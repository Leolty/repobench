"""
USAGE:

1. Load model from checkpoint
deepspeed --include localhost:0,1 --master_port 2050 inference_codegen_deepspeed.py \
    --chosen_model Salesforce/codegen-2B-mono \
    --ckpt checkpoints/codegen-2B-multi/step3000.bin \
    --language java \
    --max_new_tokens 64 \
    --batch_size 4

2. Load model from HuggingFace
deepspeed --include localhost:0,1 --master_port 2050 inference_codegen_deepspeed.py \
    --chosen_model Salesforce/codegen-2B-mono \
    --ckpt None \
    --language java \
    --max_new_tokens 64 \
    --batch_size 4
"""

import torch
import fire
import os
import deepspeed
import json
from data.utils import load_data, construct_trainable_data
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from model.utils import get_first_line_not_comment
from typing import Optional


def main(
        chosen_model:str, # the model to use, e.g., Salesforce/codegen-2B-mono
        ckpt: Optional[str], # the checkpoint to load (optional), can be None
        language:str, # the language to use, python or java
        max_new_tokens:int, # max number of tokens to generate, e.g., 64
        batch_size:int # batch size
):
    # global seed
    torch.manual_seed(42)
    if "codegen" in chosen_model:
        if language == "python":
            assert chosen_model.endswith("mono")
        elif language == "java":
            assert chosen_model.endswith("multi")
    else:
        assert language == "python" or language == "java"

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    # load model and tokenizer

    tokenizer = AutoTokenizer.from_pretrained(chosen_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if "incoder" in chosen_model:
        PAD = "<pad>"
        tokenizer.pad_token = PAD
        
    tokenizer.padding_side = "left"

    # determine whether there is a file based on the load_model path
    if os.path.exists(ckpt):
        print(f"ckpt path: {ckpt} exists, load model from ckpt")
        checkpoint = torch.load(ckpt, map_location="cpu")
        
        # load model config
        config = AutoConfig.from_pretrained(chosen_model)
        config.pad_token_id = config.eos_token_id
        config.vocab_size = len(tokenizer)

        # load model from config
        model = AutoModelForCausalLM.from_config(config)

        # load model weights
        model.load_state_dict(checkpoint, strict=False)

        # init deepspeed inference engine
        ds_model = deepspeed.init_inference(
            model=model,      # Transformers models
            mp_size=world_size, # number of gpus
            dtype=torch.float16, # dtype of the weights (fp16)
            replace_method="auto", # Lets DS autmatically identify the layer to replace
            replace_with_kernel_inject=True, # replace the model with the kernel injector
        )

    else:
        print("load model from {}".format(chosen_model))
        model = AutoModelForCausalLM.from_pretrained(
            chosen_model, torch_dtype=torch.float16
            )
        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(len(tokenizer))

        # init deepspeed inference engine
        ds_model = deepspeed.init_inference(
            model=model,      # Transformers models
            mp_size=world_size, # number of gpus
            dtype=torch.float16, # dtype of the weights (fp16)
            replace_method="auto", # Lets DS autmatically identify the layer to replace
            replace_with_kernel_inject=True, # replace the model with the kernel injector
        )
    # load datasets
    settings = ["cross_file_first", "cross_file_random", "in_file"]
    datasets = load_data("completion", language, settings)
                

    # now we can sample
    for data_part, dataset in zip(settings, datasets):
        dataset = construct_trainable_data(dataset)
        # batch load dataset
        batch_size = batch_size
        batched_dataset = []
        for i in range(0, len(dataset['test']), batch_size):
            batched_dataset.append(dataset['test'][i:i+batch_size])
        
        # inference
        for i, batch in tqdm(enumerate(batched_dataset), total=len(batched_dataset)):
            context = [d["data"] for d in batch]
            labels = [d["label"] for d in batch]
            # encode context
            encoded_input = tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=1920)

            # pass the input id and attention mask to the model
            outputs = ds_model.generate(
                    encoded_input['input_ids'].to(model.device),
                    attention_mask=encoded_input['attention_mask'].to(model.device),
                    do_sample=True,
                    max_length=max_new_tokens + len(encoded_input['input_ids'][0]),
                    top_p=0.95,
                    temperature=0.2
                    )
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                # cut off the context
                outputs = outputs[:, len(encoded_input['input_ids'][0]):]

                # batch decode
                generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                
                for _, (gen, label) in enumerate(zip(generated, labels)):
                    # save
                    res_dic = {
                            "label": label,
                            "generated": get_first_line_not_comment(gen),
                            }
                    if "codegen" in chosen_model:
                        if ckpt.endswith(".bin"):
                            directory = f"./results/completion/{chosen_model.split('/')[-1].split('-m')[0]}_{ckpt.split('step')[-1].split('.')[0]}/{language}/"
                        else:
                            directory = f"./results/completion/{chosen_model.split('/')[-1].split('-m')[0]}/{language}/"
                        os.makedirs(directory, exist_ok=True)
                    elif "incoder" in chosen_model:
                        if ckpt.endswith(".bin"):
                            directory = f"./results/completion/{chosen_model.split('/')[-1]}_{ckpt.split('step')[-1].split('.')[0]}/{language}/"
                        else:
                            directory = f"./results/completion/{chosen_model.split('/')[-1]}/{language}/"
                        os.makedirs(directory, exist_ok=True)
                    else:
                        directory = f"./results/completion/{chosen_model.split('/')[-1]}/{language}/"
                        os.makedirs(directory, exist_ok=True)

                    with open(f"{directory}/{data_part}.jsonl", "a") as f:
                        f.write(json.dumps(res_dic) + "\n")


if __name__ == "__main__":
    fire.Fire(main)


                



