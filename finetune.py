# Minimal example of training the 16B checkpoint on GPU with CPU offloading using deepspeed.

'''
apt install python3.8 python3.8-venv python3.8-dev

python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.21.1 datasets==1.16.1 deepspeed==0.7.0

deepspeed --include localhost:0,1,2,3 --master_port 1550 finetune_codegen_deepspeed.py \
    --model Salesforce/codegen-2B-mono \
    --language python \
    --epochs 10
'''

########################################################################################################
## imports

import os
import argparse
import random
import wandb

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from data.utils import load_data, construct_trainable_data

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import deepspeed

from tqdm import tqdm

########################################################################################################
## args

DEEPSPEED_CONFIG = \
{
    'fp16': {'enabled': True, 'loss_scale': 0, 'loss_scale_window': 1000, 'initial_scale_power': 12, 'hysteresis': 2, 'min_loss_scale': 1},
    'optimizer': {'type': 'AdamW', 'params': {'lr': 1e-05, 'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}},
    'scheduler': {'type': 'WarmupLR', 'params': {'warmup_min_lr': 0, 'warmup_max_lr': 1e-05, 'warmup_num_steps': 100}},
    'zero_optimization': {
        'stage': 3,
        'offload_optimizer': {'device': 'cpu', 'pin_memory': False},
        'offload_param': {'device': 'cpu', 'pin_memory': False},
        'overlap_comm': True,
        'contiguous_gradients': True,
        'sub_group_size': 1e9,
        'reduce_bucket_size': 16777216,
        'stage3_prefetch_bucket_size': 15099494.4,
        'stage3_param_persistence_threshold': 40960,
        'stage3_max_live_parameters': 1e9,
        'stage3_max_reuse_distance': 1e9,
        'stage3_gather_fp16_weights_on_model_save': True
    },
    'train_batch_size': 32,
    'train_micro_batch_size_per_gpu': 4,
    'gradient_accumulation_steps': 4,
    'gradient_clipping': 1.0,
    'steps_per_print': 8,
    'wall_clock_breakdown': False,
    'compression_training': {'weight_quantization': {'shared_parameters': {}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {}, 'different_groups': {}}}
}

def create_args(args=argparse.Namespace()):

    args.seed = 42

    args.model = "Salesforce/codegen-2B-mono"

    args.deepspeed_config = DEEPSPEED_CONFIG

    args.opt_steps_train = 1000

    args.epochs = 10

    args.language = 'python'

    return args

########################################################################################################
## preamble

def set_gpus(gpu):
    torch.cuda.set_device(gpu)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id):
    import datetime
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    return output_dir


def copy_source(file, output_dir):
    import shutil
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

########################################################################################################
## Dataset

class CodeGenDataset(Dataset):
    def __init__(self, language, tokenizer, split, max_length):
        self.data = {}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language = language
        self.input_ids = []
        self.attention_mask = []

        settings = ["cross_file_first", "cross_file_random", "in_file"]
        datasets = load_data(task="c", language=language, settings=settings)

        for data_part, dataset in zip(settings, datasets):
            self.data[data_part] = construct_trainable_data(dataset, language=language)[split]
        
        # random sample 8000, 4000, 4000 examples from cross_file_first, cross_file_random, in_file respectively
        if split == "train":
            self.data["cross_file_first"] = random.sample(self.data["cross_file_first"], 8000)
            self.data["cross_file_random"] = random.sample(self.data["cross_file_random"], 4000)
            self.data["in_file"] = random.sample(self.data["in_file"], 4000)


        # concatenate all data 
        # no need to shuffle since we will shuffle in DataLoader
        if split == "train":
            self.data = self.data["cross_file_first"] + self.data["cross_file_random"] + self.data["in_file"]
        elif split == "dev":
            # sample 60 examples from cross_file_first
            self.data = random.sample(self.data["cross_file_first"], 60)

        
        
        for i in range(len(self.data)):
            data = self.data[i]["data"] + self.data[i]["label"]
            encoding_dict = self.tokenizer(data, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
            self.input_ids.append(encoding_dict["input_ids"].squeeze())
            self.attention_mask.append(encoding_dict["attention_mask"].squeeze())


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
                "prompt": self.data[idx]["data"],
                "label": self.data[idx]["label"],
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx]
                }


########################################################################################################
## Training

def train(args):
    #######################
    ## preamble

    # set seed
    set_seed(args.seed)

    #######################
    ## model
    print('initializing model')
    config = AutoConfig.from_pretrained(args.model)
    config.gradient_checkpointing = True
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, cache_dir='cache')
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir='cache')

    tokenizer.pad_token_id = tokenizer.eos_token_id
    if "incoder" in args.model:
        PAD = "<pad>"
        tokenizer.pad_token = PAD
    tokenizer.padding_side = "left"
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    model.train()
    # TODO(enijkamp): we need to set this flag twice?
    model.gradient_checkpointing_enable()

    #######################
    ## dataset
    print('initializing dataset')
    train_dataset = CodeGenDataset(args.language, tokenizer, "train", max_length=2048)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.deepspeed_config["train_micro_batch_size_per_gpu"],
        shuffle=True
    )

    eval_dataset = CodeGenDataset(args.language, tokenizer, "dev", max_length=1950)

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.deepspeed_config["train_micro_batch_size_per_gpu"],
        shuffle=False
    )

    #######################
    ## deepspeed
    print('initializing deepspeed')
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    model_engine, optimizer, _, _ = deepspeed.initialize(
        config=args.deepspeed_config,
        model=model,
        model_parameters=model_parameters
    )

    torch.cuda.empty_cache()

    # set up wandb
    if model_engine.local_rank == 0:
        wandb.init(
            project="code-completion", 
            name= args.model.split("/")[-1],
            config={
                "learning_rate": args.deepspeed_config["optimizer"]["params"]["lr"],
                "batch_size": args.deepspeed_config["train_micro_batch_size_per_gpu"],
                "epochs": args.epochs,
                "accumulation_steps": args.deepspeed_config["gradient_accumulation_steps"],
                "num_warmup_steps": args.deepspeed_config["scheduler"]["params"]["warmup_num_steps"],
            }
            )

    #######################
    ## training

    for epoch in range(args.epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            b_input_ids = batch["input_ids"].cuda()
            b_attention_mask = batch["attention_mask"].cuda()
            b_labels = batch["input_ids"].cuda()

            loss = model_engine(b_input_ids, attention_mask=b_attention_mask, labels=b_labels).loss

            model_engine.backward(loss)
            model_engine.step()


            # log loss but we need to consider rank
            if model_engine.local_rank == 0:
                wandb.log({"loss": loss.item()})

            # eval every 2000 steps
            if step % 2000 == 0:
                eval_loss = 0

                with torch.no_grad():
                    for eval_step, eval_batch in tqdm(enumerate(eval_dataloader)):
                        eval_input_ids = eval_batch["input_ids"].cuda()
                        eval_attention_mask = eval_batch["attention_mask"].cuda()

                        batch_loss = model_engine(eval_input_ids, attention_mask=eval_attention_mask, labels=eval_input_ids).loss.item()

                        eval_loss += batch_loss

                if model_engine.local_rank == 0:
                    wandb.log({"eval_loss": eval_loss / eval_step}) 


                # save model
                save_dir = f"checkpoints/{args.model.split('/')[-1]}-java"
                os.makedirs(save_dir, exist_ok=True)

                model_engine.save_checkpoint(
                    save_dir = save_dir
                )

        



########################################################################################################
## main
def main():
    # preamble
    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id)
    # args
    args = create_args()
    args.output_dir = output_dir
    args.exp_id = exp_id
    # output
    os.makedirs(args.output_dir, exist_ok=True)
    copy_source(__file__, args.output_dir)

    # train
    train(args=args)


if __name__ == '__main__':
    main()

