

from archive_data.utils import load_data, crop_code_lines
from retriever.retriever import retrieve
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json
import os
from model.unixcoder import UniXcoder

def main(
        language: str, # language of the data, python or java
        similarity: str, # the similarity used to retrieve, e.g., cosine, edit, jaccard
        keep_lines: list, # the lines to keep, e.g., [3, 10]
        model_name: str = "", # the model used to encode the code, e.g., microsoft/unixcoder-base
        max_length: int = 512, # max length of the code
    ):
    # load the data
    settings = ["cross_file_first", "cross_file_random"]
    data_first, data_random = load_data(split="test", task="retrieval", language=language, settings=settings)

    data_first = data_first["test"]
    data_random = data_random["test"]

    # defualt lexical retrieval, no need to load the model
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi", cache_dir="cache")
    model = None

    # if semantic retrieval
    if model_name:
        # load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")
        if "codegen" in model_name:
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = tokenizer.eos_token_id
            max_length = 2048
        elif "CodeGPT" in model_name:
            max_length = 512
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif "codebert" in model_name:
            max_length = 512
        elif "unixcoder" in model_name:
            max_length = 512
        
        if "unixcoder" in model_name:
            model = UniXcoder(model_name)
        else:
            model = AutoModel.from_pretrained(model_name, cache_dir="cache")
        model.to("cuda")
    
    
    mapping = {
        "first": data_first,
        "random": data_random
    }

    for setting, dataset in mapping.items():
        res = {}
        i = 0
        for key, dic_list in dataset.items():
            res[key] = []
            for dic in tqdm(dic_list, desc=f"running {key}"):
                res_dic = {}
                for i in keep_lines:
                    code = crop_code_lines(dic['code'], i)
                    candidates = dic['context']
                    res_dic[i] = retrieve(
                        code=code,
                        candidates=candidates, 
                        tokenizer=tokenizer,
                        model=model, 
                        max_length=max_length,
                        similarity=similarity)
                
                res_dic['ground_truth'] = dic['gold_snippet_index']
                res[key].append(res_dic)
        
        # write
        if model_name:
            os.makedirs(f'results/retrieval/{model_name.split("/")[-1]}', exist_ok=True)
            with open(f"results/retrieval/{model_name.split('/')[-1]}/{language}_{setting}.json", "w") as f:
                json.dump(res, f, indent=4)
        else:
            os.makedirs(f'results/retrieval/{similarity}', exist_ok=True)
            with open(f"results/retrieval/{similarity}/{language}_{setting}.json", "w") as f:
                json.dump(res, f, indent=4)


if __name__ == "__main__":
    import fire
    fire.Fire(main)

    
