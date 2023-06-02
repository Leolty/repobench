"""
USAGE:

CUDA_VISIBLE_DEVICES=0 python inference_hf.py \
    --checkpoint Salesforce/codegen-16B-mono \
    --language python \
    --max_new_tokens 64 \
    --load_in_8bit True \
    --batch_size 1
"""

import torch
import os
import json
import fire
from transformers import AutoTokenizer, AutoModelForCausalLM
from data.utils import load_data, construct_trainable_data
from model.utils import get_first_line_not_comment
from tqdm import tqdm


def main(
        checkpoint:str, # path to the checkpoint of HuggingFace model, e.g., Salesforce/codegen-16B-mono
        language:str, # language of the data, python or java
        max_new_tokens:int, # max number of tokens to generate, e.g., 64
        load_in_8bit:bool, # NOTICE: for 16B model, load_8bit=True
        batch_size:int # NOTICE: for 16B model, batch_size=1
        ):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if "incoder" in checkpoint:
        PAD = "<pad>"
        tokenizer.pad_token = PAD
    tokenizer.padding_side = "left"

    if load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16, device_map="auto")
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # load datasets
    settings = ["cross_file_first", "cross_file_random", "in_file"]
    datasets = load_data("completion", language, settings)

    # now we can sample
    for data_part, dataset in zip(settings, datasets):
        dataset = construct_trainable_data(dataset)
        batch_size = batch_size
        batched_dataset = []
        for i in range(0, len(dataset['test']), batch_size):
            batched_dataset.append(dataset['test'][i:i+batch_size])
        
        # inference
        for i, batch in tqdm(enumerate(batched_dataset), total=len(batched_dataset)):
            context = [d["data"] for d in batch]
            labels = [d["label"] for d in batch]
            # encode context
            encoded_input = tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=1950)
            outputs = model.generate(
                    encoded_input['input_ids'].to(model.device),
                    attention_mask=encoded_input['attention_mask'].to(model.device),
                    do_sample=True,
                    max_new_tokens = max_new_tokens,
                    top_p=0.95,
                    temperature=0.2
                    )

            # cut off the context
            outputs = outputs[:, len(encoded_input['input_ids'][0]):]

            # batch decode
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # save
            for j, (gen, label) in enumerate(zip(decoded_outputs, labels)):
                # save
                res_dic = {
                        "label": label,
                        "generated": get_first_line_not_comment(gen, language),
                        }
                
                if "codegen" in checkpoint:
                    directory = f"./results/completion/{checkpoint.split('/')[-1].split('-m')[0]}/{language}/"
                else:
                    directory = f"./results/completion/{checkpoint.split('/')[-1]}/{language}/"
                os.makedirs(directory, exist_ok=True)

                with open(f"{directory}/{data_part}.jsonl", "a") as f:
                    f.write(json.dumps(res_dic) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
