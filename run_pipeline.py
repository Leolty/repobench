"""
USAGE example:

1. Run the baseline (random)
python run_pipeline.py \
    --language python \
    --mode random

2. Retrieve based on the preceding 3 lines, using the unixcoder model
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --language python \
    --mode top3 \
    --retriever unixcoder

"""

import tiktoken
from model.codex import query_with_retry, get_frist_line
import json
import os
import fire
from tqdm import tqdm
import random
from data.utils import load_data, comment, crop_code_lines
from retriever.retriever import retrieve
from model.unixcoder import UniXcoder
from transformers import AutoTokenizer, AutoModel


def main(
        language: str, # language of the data, python or java
        mode: str, # the mode to run, e.g., oracle-only, oracle-filled, random, baseline, topk (k is an integer, meaning the kept lines for retrieval)
        retriever: str = "" # the retriever used for topk, e.g., unixcoder, codebert, codegpt, codegen, jaccard, edit, default to be empty for oracle-only, oracle-filled, random, baseline
):
    # set the seed
    random.seed(42)

    encoding = tiktoken.get_encoding("p50k_base")
    encoding = tiktoken.encoding_for_model("code-davinci-002")

    # load datasets
    settings = ["cross_file_first", "cross_file_random", "in_file"]
    cff,cfr,ifr = load_data("pipeline", language, settings)

    mapping = {
        "cross_file_first": cff,
        "cross_file_random": cfr,
        "in_file": ifr
    }

    if mode == "oracle-only":
        for level, data in mapping.items():
            with tqdm(total=len(data), desc=f"Processing {level}") as pbar:
                for pbar_idx, dic in enumerate(data):
                    if level == "in_file":
                        gold_context = ""
                    else:
                        gold_snippet_index = dic['gold_snippet_index']

                        gold_path = dic['context'][gold_snippet_index]['path']
                        gold_context = dic['context'][gold_snippet_index]['snippet']

                        # comment the gold context
                        gold_context = comment(gold_context, language)
                        
                        # concat path and context
                        # we give 6144 token limit for the context
                        if language == "python":
                            gold_context = f"# Path: {gold_path}\n{gold_context}"
                        elif language == "java":
                            gold_context = f"// Path: {gold_path}\n{gold_context}"
                        gold_context_tokens = encoding.encode(gold_context)
                        if len(gold_context_tokens) > 6144:
                            gold_context_tokens = gold_context_tokens[-6144:]
                            gold_context = encoding.decode(gold_context_tokens)

                    # we give 1600 token limit for the code
                    import_statement = dic['import_statement']
                    code_path = dic['file_path']
                    code = dic['code']
                    code = "\n".join(code.split("\n")[-60:])
                    if language == "python":
                        code = f"# Path: {code_path}\n{import_statement}\n{code}"
                    elif language == "java":
                        # let's first see whether the first line is a package statement
                        if code.split("\n")[0].startswith("package"):
                            # insert the import statement after the package statement
                            code = "\n".join(code.split("\n")[:1] + [import_statement] + code.split("\n")[1:])
                            code = f"// Path: {code_path}\n{code}"
                        else:
                            code = f"// Path: {code_path}\n{import_statement}\n{code}"
                    
                    code_tokens = encoding.encode(code)
                    if len(code_tokens) > 1600:
                        code_tokens = code_tokens[-1600:]
                        code = encoding.decode(code_tokens)


                    next_line = dic['next_line']
                    if gold_context:
                        prompt = f"{gold_context}\n{code}"
                    else:
                        prompt = f"{code}"

                    # query the model
                    reponse = query_with_retry(prompt)
                    # get the first line of the response
                    first_line = get_frist_line(reponse)

                    res = {
                        "label": next_line,
                        "generated": first_line
                    }

                    os.makedirs(f"results/pipeline/{mode}/{language}", exist_ok=True)

                    with open(f"results/pipeline/{mode}/{language}/{level}.jsonl", "a") as f:
                        f.write(json.dumps(res) + "\n")
                    
                    if pbar_idx % 5 == 0 and pbar_idx != 0:
                        pbar.update(5)

    elif mode == "oracle-filled":
        for level, data in mapping.items():
            with tqdm(total=len(data), desc=f"Processing {level}") as pbar:
                for pbar_idx, dic in enumerate(data):
                    context = ""
                    context_length = 0
                    included_snippet_index = []
                    for i in range(len(dic['context'])):
                        if level == "in_file":
                            # random select a snippet not in the included_snippet_index
                            candidates = list(set(range(len(dic['context']))) - set(included_snippet_index))
                            random_snippet_index = random.choice(candidates)
                        else:
                            if i == 0:
                                # the first snippet is the gold snippet
                                random_snippet_index = dic['gold_snippet_index']
                            else:
                                # random select a snippet not in the included_snippet_index
                                candidates = list(set(range(len(dic['context']))) - set(included_snippet_index))
                                random_snippet_index = random.choice(candidates)
                        
                        included_snippet_index.append(random_snippet_index)
                        random_snippet_path = dic['context'][random_snippet_index]['path']
                        random_snippet_context = dic['context'][random_snippet_index]['snippet']

                        # comment the random snippet context
                        random_snippet_context = comment(random_snippet_context, language)

                        # concat path and context
                        # we give 6144 token limit for the context
                        if language == "python":
                            random_snippet_context = f"# Path: {random_snippet_path}\n{random_snippet_context}"
                        elif language == "java":
                            random_snippet_context = f"// Path: {random_snippet_path}\n{random_snippet_context}"
                        
                        random_snippet_context_tokens = encoding.encode(random_snippet_context)
                        
                        # concat the context, but we want to concat random snippet before the context
                        context = f"{random_snippet_context}\n{context}"
                        context_length += len(random_snippet_context_tokens)+1

                        # 6144 is the limit
                        if context_length > 6144:
                            # cut the context
                            context = context[-6144:]
                            break
                
                    # we give 1600 token limit for the code
                    import_statement = dic['import_statement']
                    code_path = dic['file_path']
                    code = dic['code']
                    code = "\n".join(code.split("\n")[-60:])
                    if language == "python":
                        code = f"# Path: {code_path}\n{import_statement}\n{code}"
                    elif language == "java":
                        # let's first see whether the first line is a package statement
                        if code.split("\n")[0].startswith("package"):
                            # insert the import statement after the package statement
                            code = "\n".join(code.split("\n")[:1] + [import_statement] + code.split("\n")[1:])
                            code = f"// Path: {code_path}\n{code}"
                        else:
                            code = f"// Path: {code_path}\n{import_statement}\n{code}"
                    
                    code_tokens = encoding.encode(code)
                    if len(code_tokens) > 1600:
                        code_tokens = code_tokens[-1600:]
                        code = encoding.decode(code_tokens)
                    
                    next_line = dic['next_line']
                
                    prompt = f"{context}\n{code}"

                    # query the model
                    reponse = query_with_retry(prompt)
                    # get the first line of the response
                    first_line = get_frist_line(reponse)

                    res = {
                        "label": next_line,
                        "generated": first_line
                    }

                    os.makedirs(f"results/pipeline/{mode}/{language}", exist_ok=True)

                    with open(f"results/pipeline/{mode}/{language}/{level}.jsonl", "a") as f:
                        f.write(json.dumps(res) + "\n")
                    
                    if pbar_idx % 5 == 0 and pbar_idx != 0:
                        pbar.update(5)

        
    elif mode == "random":
        for level, data in mapping.items():
            with tqdm(total=len(data), desc=f"Processing {level}") as pbar:
                for pbar_idx, dic in enumerate(data):
                    context = ""
                    context_length = 0
                    included_snippet_index = []
                    for i in range(len(dic['context'])):
                        # random select a snippet not in the included_snippet_index
                        candidates = list(set(range(len(dic['context']))) - set(included_snippet_index))
                        random_snippet_index = random.choice(candidates)

                        random_snippet_path = dic['context'][random_snippet_index]['path']
                        random_snippet_context = dic['context'][random_snippet_index]['snippet']

                        # comment the random snippet context
                        random_snippet_context = comment(random_snippet_context, language)

                        # concat path and context
                        # we give 6144 token limit for the context
                        if language == "python":
                            random_snippet_context = f"# Path: {random_snippet_path}\n{random_snippet_context}"
                        elif language == "java":
                            random_snippet_context = f"// Path: {random_snippet_path}\n{random_snippet_context}"
                        
                        random_snippet_context_tokens = encoding.encode(random_snippet_context)
                        
                        # concat the context, but we want to concat random snippet before the context
                        context = f"{random_snippet_context}\n{context}"
                        context_length += len(random_snippet_context_tokens)+1

                        # 6144 is the limit
                        if context_length > 6144:
                            # cut the context
                            context = context[-6144:]
                            break
                
                    # we give 1600 token limit for the code
                    import_statement = dic['import_statement']
                    code_path = dic['file_path']
                    code = dic['code']
                    code = "\n".join(code.split("\n")[-60:])
                    if language == "python":
                        code = f"# Path: {code_path}\n{import_statement}\n{code}"
                    elif language == "java":
                        # let's first see whether the first line is a package statement
                        if code.split("\n")[0].startswith("package"):
                            # insert the import statement after the package statement
                            code = "\n".join(code.split("\n")[:1] + [import_statement] + code.split("\n")[1:])
                            code = f"// Path: {code_path}\n{code}"
                        else:
                            code = f"// Path: {code_path}\n{import_statement}\n{code}"
                    
                    code_tokens = encoding.encode(code)
                    if len(code_tokens) > 1600:
                        code_tokens = code_tokens[-1600:]
                        code = encoding.decode(code_tokens)
                    
                    next_line = dic['next_line']
                
                    prompt = f"{context}\n{code}"

                    # query the model
                    reponse = query_with_retry(prompt)
                    # get the first line of the response
                    first_line = get_frist_line(reponse)

                    res = {
                        "label": next_line,
                        "generated": first_line
                    }

                    os.makedirs(f"results/pipeline/{mode}/{language}", exist_ok=True)

                    with open(f"results/pipeline/{mode}/{language}/{level}.jsonl", "a") as f:
                        f.write(json.dumps(res) + "\n")
                    
                    if pbar_idx % 5 == 0 and pbar_idx != 0:
                        pbar.update(5)
        
    elif mode.startswith("top"):
        top = int(mode.split("top")[-1])
        # load the retriever
        if retriever == "unixcoder":
            model = UniXcoder("microsoft/unixcoder-base")
            tokenizer = None
            sim = "cosine"
            max_length = 512
            model.to("cuda")
        elif retriever == "codebert":
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", cache_dir="cache")
            model = AutoModel.from_pretrained("microsoft/codebert-base", cache_dir="cache")
            sim = "cosine"
            max_length = 512
            model.to("cuda")
        elif retriever == "codegpt":
            if language == "python":
                tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py-adaptedGPT2", cache_dir="cache")
                model = AutoModel.from_pretrained("microsoft/CodeGPT-small-py-adaptedGPT2", cache_dir="cache")
            elif language == "java":
                tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-java-adaptedGPT2", cache_dir="cache")
                model = AutoModel.from_pretrained("microsoft/CodeGPT-small-java-adaptedGPT2", cache_dir="cache")
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = tokenizer.eos_token_id
            sim = "cosine"
            max_length = 512
            model.to("cuda")
        elif retriever == "codegen":
            if language == "python":
                tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono", cache_dir="cache")
                model = AutoModel.from_pretrained("Salesforce/codegen-350M-mono", cache_dir="cache")
            elif language == "java":
                tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi", cache_dir="cache")
                model = AutoModel.from_pretrained("Salesforce/codegen-350M-multi", cache_dir="cache")
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = tokenizer.eos_token_id
            sim = "cosine"
            max_length = 2048
            model.to("cuda")
        elif retriever == "jaccard":
            tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono", cache_dir="cache")
            model = None
            sim = "jaccard"
            max_length = 0
        elif retriever == "edit":
            tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono", cache_dir="cache")
            model = None
            sim = "edit"
            max_length = 0
        
        for level, data in mapping.items():
            with tqdm(total=len(data), desc=f"Processing {level}") as pbar:
                for pbar_idx, dic in enumerate(data):
                    # first let's crop the code
                    code_for_retrieval = crop_code_lines(dic['code'], int(top))
                    candidate_snippets = [dic['context'][i]['snippet'] for i in range(len(dic['context']))]

                    context = ""
                    context_length = 0
                    
                    rank = retrieve(code=code_for_retrieval, candidates=candidate_snippets, tokenizer=tokenizer, model=model, max_length=max_length, similarity=sim)
                    for i in rank:

                        random_snippet_path = dic['context'][i]['path']
                        random_snippet_context = dic['context'][i]['snippet']

                        # comment the random snippet context
                        random_snippet_context = comment(random_snippet_context, language)

                        # concat path and context
                        # we give 6144 token limit for the context
                        if language == "python":
                            random_snippet_context = f"# Path: {random_snippet_path}\n{random_snippet_context}"
                        elif language == "java":
                            random_snippet_context = f"// Path: {random_snippet_path}\n{random_snippet_context}"
                        
                        random_snippet_context_tokens = encoding.encode(random_snippet_context)
                        
                        # concat the context, but we want to concat random snippet before the context
                        context = f"{random_snippet_context}\n{context}"
                        context_length += len(random_snippet_context_tokens)+1

                        # 6144 is the limit
                        if context_length > 6144:
                            # cut the context
                            context = context[-6144:]
                            # crop the first line
                            context = "\n".join(context.split("\n")[1:])
                            break
                
                    # we give 1600 token limit for the code
                    import_statement = dic['import_statement']
                    code_path = dic['file_path']
                    code = dic['code']
                    code = "\n".join(code.split("\n")[-60:])
                    if language == "python":
                        code = f"# Path: {code_path}\n{import_statement}\n{code}"
                    elif language == "java":
                        # let's first see whether the first line is a package statement
                        if code.split("\n")[0].startswith("package"):
                            # insert the import statement after the package statement
                            code = "\n".join(code.split("\n")[:1] + [import_statement] + code.split("\n")[1:])
                            code = f"// Path: {code_path}\n{code}"
                        else:
                            code = f"// Path: {code_path}\n{import_statement}\n{code}"
                    
                    code_tokens = encoding.encode(code)
                    if len(code_tokens) > 1600:
                        code_tokens = code_tokens[-1600:]
                        code = encoding.decode(code_tokens)
                    
                    next_line = dic['next_line']
                
                    prompt = f"{context}\n{code}"

                    # print(prompt)

                    # import time
                    # time.sleep(100)

                    # query the model
                    reponse = query_with_retry(prompt)

                    # get the first line of the response
                    first_line = get_frist_line(reponse)

                    res = {
                        "label": next_line,
                        "generated": first_line
                    }

                    os.makedirs(f"results/pipeline/{mode}/{retriever}/{language}", exist_ok=True)

                    with open(f"results/pipeline/{mode}/{retriever}/{language}/{level}.jsonl", "a") as f:
                        f.write(json.dumps(res) + "\n")   
                    
                    if pbar_idx % 5 == 0 and pbar_idx != 0:
                        pbar.update(5)
    
    elif mode == "baseline":
        for level, data in mapping.items():
            with tqdm(total=len(data), desc=f"Processing {level}") as pbar:
                for pbar_idx, dic in enumerate(data):
                    # we give 1600 token limit for the code
                    import_statement = dic['import_statement']
                    code_path = dic['file_path']
                    code = dic['code']
                    code = "\n".join(code.split("\n")[-60:])
                    if language == "python":
                        code = f"# Path: {code_path}\n{import_statement}\n{code}"
                    elif language == "java":
                        # let's first see whether the first line is a package statement
                        if code.split("\n")[0].startswith("package"):
                            # insert the import statement after the package statement
                            code = "\n".join(code.split("\n")[:1] + [import_statement] + code.split("\n")[1:])
                            code = f"// Path: {code_path}\n{code}"
                        else:
                            code = f"// Path: {code_path}\n{import_statement}\n{code}"
                    
                    code_tokens = encoding.encode(code)
                    if len(code_tokens) > 1600:
                        code_tokens = code_tokens[-1600:]
                        code = encoding.decode(code_tokens)
                    
                    next_line = dic['next_line']
                
                    prompt = f"{code}"


                    # query the model
                    reponse = query_with_retry(prompt)

                    # get the first line of the response
                    first_line = get_frist_line(reponse)

                    res = {
                        "label": next_line,
                        "generated": first_line
                    }

                    os.makedirs(f"results/pipeline/{mode}/{language}", exist_ok=True)

                    with open(f"results/pipeline/{mode}/{language}/{level}.jsonl", "a") as f:
                        f.write(json.dumps(res) + "\n")   
                    
                    if pbar_idx % 5 == 0 and pbar_idx != 0:
                        pbar.update(5)

if __name__ == "__main__":
    fire.Fire(main)