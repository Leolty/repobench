import torch
import os
import json
import fire
from archive_data.utils import load_data, construct_trainable_data
from model.utils import get_first_line_not_comment
from tqdm import tqdm
from hf_hub_ctranslate2 import GeneratorCT2fromHfHub
from transformers import AutoTokenizer


prefix_token = "<fim_prefix>"
suffix_token = "<fim_suffix><fim_middle>"

@torch.no_grad()
def main(
        model_name:str = "michaelfeil/ct2fast-starcoder",
        language:str = "python",
        length="2k", 
        max_new_tokens:int = 64,
        temperature: float = 0.2,
        resume_part:str = "cross_file_first",
        resume:int = 0,
        ):

    # load model
    model = GeneratorCT2fromHfHub(
        # load in int8 on CUDA
        model_name_or_path=model_name,
        device="cuda",
        compute_type="int8_float16"
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder", cache_dir="cache")

    # load datasets
    settings = ["cross_file_first", "cross_file_random", "in_file"]
    datasets = load_data(split= "test", task="completion", language=language, length=length, settings=settings)

    # now we can sample
    for data_part, dataset in zip(settings, datasets):
        dataset = construct_trainable_data(dataset, language=language)

        if resume_part == "cross_file_random" and data_part == "cross_file_first":
            continue

        if resume_part == "in_file" and data_part in ["cross_file_first", "cross_file_random"]:
            continue
        
        # inference
        for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
            # resume
            if resume_part == data_part and idx < resume:
                continue

            prompt = data["data"]
            label = data["label"]

            if "star" in model_name: # starcoder
                prompt = prefix_token + prompt + suffix_token
            

            # tokenize, get prompt length
            prompt_length = len(tokenizer.tokenize(prompt)) 

            output = model.generate(
                text=[prompt],
                max_length=max_new_tokens,
                include_prompt_in_result=False,
                sampling_temperature=temperature,
            )[0]

            # save
            res_dic = {
                    "idx": idx,
                    "label": label,
                    "generated": get_first_line_not_comment(output, language=language),
                    "prompt_length": prompt_length
                }

            if "mono" in model_name or "multi" in model_name: # codegen
                model_name = model_name.replace("-mono", "").replace("-multi", "")
                
            directory = f"./results/{model_name.split('ct2fast-')[-1]}-{length}/{language}/"
            os.makedirs(directory, exist_ok=True)

            with open(f"{directory}/{data_part}.jsonl", "a") as f:
                f.write(json.dumps(res_dic) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
