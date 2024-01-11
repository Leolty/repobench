import torch
import tiktoken
import json
import os
import fire
from tqdm import tqdm
import random

from model.codex import query_with_retry, get_frist_line
from model.utils import get_first_line_not_comment
from archive_data.utils import load_data, crop_code_lines, comment

from retriever.retriever import retrieve
from unixcoder import UniXcoder
from transformers import AutoTokenizer, AutoModel
from hf_hub_ctranslate2 import GeneratorCT2fromHfHub
from collections import defaultdict

class TiktokenWrapper:
    def __init__(self, original_encoder):
        self.original_encoder = original_encoder

    def encode(self, text):
        return self.original_encoder.encode(text, disallowed_special=())
    
    def decode(self, ids):
        return self.original_encoder.decode(ids)
    

def main(
        language: str = "python", # python, java
        mode: str = "top3", # gold-only, gold-filled-head, gold-filled-tail, random, baseline, top3 (crop the previous 3 lines)
        retriever: str = "", # unixcoder, jaccard, only for topk
        model_name: str = "codex", # codex, starcoder
        resume_part: str = "cross_file_first",
        resume: int = 0,
        debug: bool = 0,
        print_num: int = 3,
):
    # set the seed
    random.seed(42)

    if model_name == "codex":
        encoding = tiktoken.encoding_for_model("code-davinci-002")
        encoding = TiktokenWrapper(encoding)
    elif model_name == "starcoder":
        encoding = AutoTokenizer.from_pretrained("bigcode/starcoder", cache_dir="cache")
        encoding.pad_token_id = encoding.eos_token_id
        encoding.padding_side = "left"

        if not debug:
            # load model
            model_name_or_path = "michaelfeil/ct2fast-starcoder" if language == "python" else "michaelfeil/ct2fast-starcoderbase"
            # get CUDA_VISIBLE_DEVICES
            device_num = torch.cuda.device_count()
            if device_num == 1:
                model = GeneratorCT2fromHfHub(
                    # load in int8 on CUDA
                    model_name_or_path=model_name_or_path,
                    device="cuda",
                    compute_type="int8_float16"
                )
            elif device_num == 2:
                # load model
                model = GeneratorCT2fromHfHub(
                    # load in int8 on CUDA
                    model_name_or_path=model_name_or_path,
                    device="cuda",
                    compute_type="int8_float16",
                    device_index = [1]
                )

        prefix_token = "<fim_prefix>"
        suffix_token = "<fim_suffix><fim_middle>"

    # load datasets
    settings = ["cross_file_first", "cross_file_random", "in_file"]
    cff,cfr,ifr = load_data(split="test", task="pipeline", language=language, settings=settings)

    mapping = {
        "cross_file_first": cff,
        "cross_file_random": cfr,
        "in_file": ifr
    }

    if mode == "gold-only":

        os.makedirs(f"results_new/pipeline/{model_name}/{mode}/{language}", exist_ok=True)

        for level, data in mapping.items():
            if resume_part == "cross_file_random" and level == "cross_file_first":
                continue
                
            if resume_part == "in_file" and level in ["cross_file_first", "cross_file_random"]:
                continue

            with tqdm(total=len(data), desc=f"Processing {level}") as pbar:
                for pbar_idx, dic in enumerate(data):
                    if level == resume_part and pbar_idx < resume:
                        pbar.update(1)
                        continue

                    # First, handle the code
                    import_statement = dic['import_statement']
                    code_path = dic['file_path']
                    next_line = dic['next_line']
                    code = dic['code']
                    code = "\n".join(code.split("\n")[-60:])
                    if language == "python":
                        code = f"# Path: {code_path}\n{import_statement}\n{code}"
                    elif language == "java":
                        if code.split("\n")[0].startswith("package"):
                            # insert the import statement after the package statement
                            code = "\n".join(code.split("\n")[:1] + [import_statement] + code.split("\n")[1:])
                        else:
                            code = f"// Path: {code_path}\n{import_statement}\n{code}"

                    code_tokens = encoding.encode(code)
                    if len(code_tokens) > 1600:
                        code_tokens = code_tokens[-1600:]
                        code = encoding.decode(code_tokens)

                    # Then, handle the gold context
                    if level == "in_file":
                        gold_context = ""
                    else:
                        gold_snippet_index = dic['gold_snippet_index']
                        gold_path = dic['context'][gold_snippet_index]['path']
                        gold_context = dic['context'][gold_snippet_index]['snippet']
                        gold_context = comment(gold_context, language)
                        if language == "python":
                            gold_context = f"# Path: {gold_path}\n{gold_context}"
                        elif language == "java":
                            gold_context = f"// Path: {gold_path}\n{gold_context}"
                        gold_context_tokens = encoding.encode(gold_context)

                        # Ensure that the combined length of code, gold_context, and the newline doesn't exceed 6400 tokens
                        combined_length = len(code_tokens) + len(gold_context_tokens) + 1 # +1 for the newline
                        if combined_length > 6400:
                            gold_context_tokens = gold_context_tokens[:6400 - len(code_tokens) - 1] # -1 for the newline
                            gold_context = encoding.decode(gold_context_tokens)

                    # Combine gold_context and code for the prompt
                    if gold_context:
                        prompt = f"{gold_context}\n{code}"
                    else:
                        prompt = f"{code}"

                    os.makedirs(f"results_new/pipeline/codex/gold-only/{language}", exist_ok=True)
                    with open(f"results_new/pipeline/codex/gold-only/{language}/{level}_prompt_length.jsonl", "a") as f:
                        f.write(json.dumps({
                            "idx": pbar_idx,
                            "prompt_length": len(encoding.encode(prompt))
                        }) + "\n")

                    if debug:
                        # print the first 10 prompts
                        if pbar_idx < print_num:
                            print(prompt)
                            print(f"####################\n{next_line}\n")

            
                    else:
                        # check whether we have already generated the result
                        if os.path.exists(f"results_new/pipeline/{model_name}/{mode}/{language}/{level}.jsonl"):
                            # we can use idx to check whether we have already generated the result
                            with open(f"results_new/pipeline/{model_name}/{mode}/{language}/{level}.jsonl", "r") as f:
                                lines = [json.loads(line.strip()) for line in f]
                                # get all the idx
                                idxs = [line["idx"] for line in lines]
                                if pbar_idx in idxs:
                                    pbar.update(1)
                                    continue

                        if model_name == "codex":
                            # query the model
                            reponse = query_with_retry(prompt)
                            # get the first line of the response
                            first_line = get_frist_line(reponse)
                        elif model_name == "starcoder":
                            # add prefix and suffix
                            prompt = prefix_token + prompt + suffix_token

                            # query the model
                            output = model.generate(
                                text=[prompt],
                                max_length=64,
                                include_prompt_in_result=False,
                                sampling_temperature=0.2
                            )[0]

                            first_line = get_first_line_not_comment(output, language)

                        res = {
                            "idx": pbar_idx,
                            "label": next_line,
                            "generated": first_line
                        }

                        with open(f"results_new/pipeline/{model_name}/{mode}/{language}/{level}.jsonl", "a") as f:
                            f.write(json.dumps(res) + "\n")
                        
                        if pbar_idx % 5 == 0 and pbar_idx != 0:
                            pbar.update(5)

    elif mode == "gold-filled-head":
        for level, data in mapping.items():
            if resume_part == "cross_file_random" and level == "cross_file_first":
                continue
                
            if resume_part == "in_file" and level in ["cross_file_first", "cross_file_random"]:
                continue

            with tqdm(total=len(data), desc=f"Processing {level}") as pbar:
                for pbar_idx, dic in enumerate(data):
                    if level == resume_part and pbar_idx < resume:
                        pbar.update(1)
                        continue

                    context_snippets = []

                    # Start with the gold_snippet
                    if level != "in_file":
                        gold_snippet_path = dic['context'][dic['gold_snippet_index']]['path']
                        gold_snippet_content = dic['context'][dic['gold_snippet_index']]['snippet']
                        gold_snippet_content = comment(gold_snippet_content, language)
                        if language == "python":
                            context_snippets.append(f"# Path: {gold_snippet_path}\n{gold_snippet_content}")
                        elif language == "java":
                            context_snippets.append(f"// Path: {gold_snippet_path}\n{gold_snippet_content}")

                    included_snippet_index = [dic['gold_snippet_index']] if level != "in_file" else []

                    for i in range(len(dic['context'])):
                        if i == 0 and level != "in_file":
                            continue  # we've already added the gold snippet
                        
                        candidates = list(set(range(len(dic['context']))) - set(included_snippet_index))
                        random_snippet_index = random.choice(candidates)
                        included_snippet_index.append(random_snippet_index)

                        random_snippet_path = dic['context'][random_snippet_index]['path']
                        random_snippet_content = dic['context'][random_snippet_index]['snippet']
                        random_snippet_content = comment(random_snippet_content, language)
                        if language == "python":
                            context_snippets.append(f"# Path: {random_snippet_path}\n{random_snippet_content}")
                        elif language == "java":
                            context_snippets.append(f"// Path: {random_snippet_path}\n{random_snippet_content}")
                        
                    context = '\n'.join(context_snippets)

                    # Handle code section
                    import_statement = dic['import_statement']
                    code_path = dic['file_path']
                    next_line = dic['next_line']
                    code = dic['code']
                    code = "\n".join(code.split("\n")[-60:])
                    if language == "python":
                        code = f"# Path: {code_path}\n{import_statement}\n{code}"
                    elif language == "java":
                        if code.split("\n")[0].startswith("package"):
                            code = "\n".join(code.split("\n")[:1] + [import_statement] + code.split("\n")[1:])
                            code = f"// Path: {code_path}\n{code}"
                        else:
                            code = f"// Path: {code_path}\n{import_statement}\n{code}"

                    code_tokens = encoding.encode(code)
                    if len(code_tokens) > 1600:
                        code_tokens = code_tokens[-1600:]
                        code = encoding.decode(code_tokens)

                    combined_length = len(encoding.encode(context)) + len(code_tokens) + 1  # +1 for newline

                    # Ensure the total prompt does not exceed 6400 tokens
                    if combined_length > 6400:
                        context_tokens_to_take = 6400 - len(code_tokens) - 1  # -1 for newline
                        context = encoding.decode(encoding.encode(context)[:context_tokens_to_take])

                    prompt = f"{context}\n{code}"
                
                    if debug:
                        # print the first 10 prompts
                        if pbar_idx < print_num:
                            print(prompt)
                            print(f"####################\n{next_line}\n")

                        break
            
                    else:
                        # check whether we have already generated the result
                        if os.path.exists(f"results_new/pipeline/{model_name}/{mode}/{language}/{level}.jsonl"):
                            # we can use idx to check whether we have already generated the result
                            with open(f"results_new/pipeline/{model_name}/{mode}/{language}/{level}.jsonl", "r") as f:
                                lines = [json.loads(line.strip()) for line in f]
                                # get all the idx
                                idxs = [line["idx"] for line in lines]
                                if pbar_idx in idxs:
                                    pbar.update(1)
                                    continue

                        if model_name == "codex":
                            # query the model
                            reponse = query_with_retry(prompt)
                            # get the first line of the response
                            first_line = get_frist_line(reponse)
                        elif model_name == "starcoder":
                            # add prefix and suffix
                            prompt = prefix_token + prompt + suffix_token

                            # query the model
                            output = model.generate(
                                text=[prompt],
                                max_length=64,
                                include_prompt_in_result=False,
                                sampling_temperature=0.2
                            )[0]

                            first_line = get_first_line_not_comment(output, language)

                        res = {
                            "idx": pbar_idx,
                            "label": next_line,
                            "generated": first_line
                        }

                        os.makedirs(f"results_new/pipeline/{model_name}/{mode}/{language}", exist_ok=True)

                        with open(f"results_new/pipeline/{model_name}/{mode}/{language}/{level}.jsonl", "a") as f:
                            f.write(json.dumps(res) + "\n")
                        
                        if pbar_idx % 5 == 0 and pbar_idx != 0:
                            pbar.update(5)

    elif mode == "gold-filled-tail":
        for level, data in mapping.items():
            if resume_part == "cross_file_random" and level == "cross_file_first":
                continue
                
            if resume_part == "in_file" and level in ["cross_file_first", "cross_file_random"]:
                continue

            with tqdm(total=len(data), desc=f"Processing {level}") as pbar:
                for pbar_idx, dic in enumerate(data):
                    if level == resume_part and pbar_idx < resume:
                        pbar.update(1)
                        continue

                    context_snippets = []

                    # Start with the gold_snippet
                    if level != "in_file":
                        gold_snippet_path = dic['context'][dic['gold_snippet_index']]['path']
                        gold_snippet_content = dic['context'][dic['gold_snippet_index']]['snippet']
                        gold_snippet_content = comment(gold_snippet_content, language)
                        if language == "python":
                            context_snippets.append(f"# Path: {gold_snippet_path}\n{gold_snippet_content}")
                        elif language == "java":
                            context_snippets.append(f"// Path: {gold_snippet_path}\n{gold_snippet_content}")

                    included_snippet_index = [dic['gold_snippet_index']] if level != "in_file" else []

                    for i in range(len(dic['context'])):
                        if i == 0 and level != "in_file":
                            continue  # we've already added the gold snippet
                        
                        candidates = list(set(range(len(dic['context']))) - set(included_snippet_index))
                        random_snippet_index = random.choice(candidates)
                        included_snippet_index.append(random_snippet_index)

                        random_snippet_path = dic['context'][random_snippet_index]['path']
                        random_snippet_content = dic['context'][random_snippet_index]['snippet']
                        random_snippet_content = comment(random_snippet_content, language)
                        if language == "python":
                            context_snippets.append(f"# Path: {random_snippet_path}\n{random_snippet_content}")
                        elif language == "java":
                            context_snippets.append(f"// Path: {random_snippet_path}\n{random_snippet_content}")
                    
                    # reverse the context snippets
                    context = '\n'.join(context_snippets[::-1])

                    # Handle code section
                    import_statement = dic['import_statement']
                    code_path = dic['file_path']
                    next_line = dic['next_line']
                    code = dic['code']
                    code = "\n".join(code.split("\n")[-60:])
                    if language == "python":
                        code = f"# Path: {code_path}\n{import_statement}\n{code}"
                    elif language == "java":
                        if code.split("\n")[0].startswith("package"):
                            code = "\n".join(code.split("\n")[:1] + [import_statement] + code.split("\n")[1:])
                            code = f"// Path: {code_path}\n{code}"
                        else:
                            code = f"// Path: {code_path}\n{import_statement}\n{code}"

                    code_tokens = encoding.encode(code)
                    if len(code_tokens) > 1600:
                        code_tokens = code_tokens[-1600:]
                        code = encoding.decode(code_tokens)

                    combined_length = len(encoding.encode(context)) + len(code_tokens) + 1  # +1 for newline

                    # Ensure the total prompt does not exceed 6400 tokens
                    if combined_length > 6400:
                        context_tokens_to_take = 6400 - len(code_tokens) - 1  # -1 for newline
                        # Truncate the rightmost context
                        context = encoding.decode(encoding.encode(context)[-context_tokens_to_take:])

                    prompt = f"{context}\n{code}"
                
                    if debug:
                        # print the first 10 prompts
                        if pbar_idx < print_num:
                            print(prompt)
                            print(f"####################\n{next_line}\n")

                        break
            
                    else:
                        # check whether we have already generated the result
                        if os.path.exists(f"results_new/pipeline/{model_name}/{mode}/{language}/{level}.jsonl"):
                            # we can use idx to check whether we have already generated the result
                            with open(f"results_new/pipeline/{model_name}/{mode}/{language}/{level}.jsonl", "r") as f:
                                lines = [json.loads(line.strip()) for line in f]
                                # get all the idx
                                idxs = [line["idx"] for line in lines]
                                if pbar_idx in idxs:
                                    pbar.update(1)
                                    continue

                        if model_name == "codex":
                            # query the model
                            reponse = query_with_retry(prompt)
                            # get the first line of the response
                            first_line = get_frist_line(reponse)
                        elif model_name == "starcoder":
                            # add prefix and suffix
                            prompt = prefix_token + prompt + suffix_token

                            # query the model
                            output = model.generate(
                                text=[prompt],
                                max_length=64,
                                include_prompt_in_result=False,
                                sampling_temperature=0.2
                            )[0]

                            first_line = get_first_line_not_comment(output, language)

                        res = {
                            "idx": pbar_idx,
                            "label": next_line,
                            "generated": first_line
                        }

                        os.makedirs(f"results_new/pipeline/{model_name}/{mode}/{language}", exist_ok=True)

                        with open(f"results_new/pipeline/{model_name}/{mode}/{language}/{level}.jsonl", "a") as f:
                            f.write(json.dumps(res) + "\n")
                        
                        if pbar_idx % 5 == 0 and pbar_idx != 0:
                            pbar.update(5)

    elif mode == "random":
        for level, data in mapping.items():
            if resume_part == "cross_file_random" and level == "cross_file_first":
                continue
                
            if resume_part == "in_file" and level in ["cross_file_first", "cross_file_random"]:
                continue
            with tqdm(total=len(data), desc=f"Processing {level}") as pbar:
                for pbar_idx, dic in enumerate(data):
                    if level == resume_part and pbar_idx < resume:
                        pbar.update(1)
                        continue       
         
                    # First, handle the code as you instructed
                    import_statement = dic['import_statement']
                    code_path = dic['file_path']
                    code = dic['code']
                    code = "\n".join(code.split("\n")[-60:])
                    if language == "python":
                        code = f"# Path: {code_path}\n{import_statement}\n{code}"
                    elif language == "java":
                        if code.split("\n")[0].startswith("package"):
                            code = "\n".join(code.split("\n")[:1] + [import_statement] + code.split("\n")[1:])
                            code = f"// Path: {code_path}\n{code}"
                        else:
                            code = f"// Path: {code_path}\n{import_statement}\n{code}"
                        
                    code_tokens = encoding.encode(code)
                    if len(code_tokens) > 1600:
                        code_tokens = code_tokens[-1600:]
                        code = encoding.decode(code_tokens)
                    
                    # Then, handle the context while considering the length of the code
                    context = ""
                    context_length =len(code_tokens) + 1 
                    included_snippet_index = []
                    for i in range(len(dic['context'])):
                        candidates = list(set(range(len(dic['context']))) - set(included_snippet_index))
                        random_snippet_index = random.choice(candidates)
                        included_snippet_index.append(random_snippet_index)

                        random_snippet_path = dic['context'][random_snippet_index]['path']
                        random_snippet_context = dic['context'][random_snippet_index]['snippet']

                        random_snippet_context = comment(random_snippet_context, language)
                        if language == "python":
                            random_snippet_context = f"# Path: {random_snippet_path}\n{random_snippet_context}"
                        elif language == "java":
                            random_snippet_context = f"// Path: {random_snippet_path}\n{random_snippet_context}"

                        random_snippet_context_tokens = encoding.encode(random_snippet_context)

                        context_length += len(random_snippet_context_tokens) + 1

                        if context_length > 6400:
                            # crop the context
                            random_snippet_context_tokens = random_snippet_context_tokens[:6400 - context_length - 1]
                            random_snippet_context = encoding.decode(random_snippet_context_tokens)
                            context = context + "\n" + random_snippet_context
                            break
                            
                        # Append the context
                        context = f"{context}\n{random_snippet_context}"

                    next_line = dic['next_line']

                    prompt = f"{context}\n{code}"
                
                    if debug:
                        # print the first 10 prompts
                        if pbar_idx < print_num:
                            print(prompt)
                            print(f"####################\n{next_line}\n")

                        break
            
                    else:
                        # check whether we have already generated the result
                        if os.path.exists(f"results_new/pipeline/{model_name}/{mode}/{language}/{level}.jsonl"):
                            # we can use idx to check whether we have already generated the result
                            with open(f"results_new/pipeline/{model_name}/{mode}/{language}/{level}.jsonl", "r") as f:
                                lines = [json.loads(line.strip()) for line in f]
                                # get all the idx
                                idxs = [line["idx"] for line in lines]
                                if pbar_idx in idxs:
                                    pbar.update(1)
                                    continue

                        if model_name == "codex":
                            # query the model
                            reponse = query_with_retry(prompt)
                            # get the first line of the response
                            first_line = get_frist_line(reponse)
                        elif model_name == "starcoder":
                            # add prefix and suffix
                            prompt = prefix_token + prompt + suffix_token

                            # query the model
                            output = model.generate(
                                text=[prompt],
                                max_length=64,
                                include_prompt_in_result=False,
                                sampling_temperature=0.2,
                            )[0]

                            first_line = get_first_line_not_comment(output, language)

                        res = {
                            "idx": pbar_idx,
                            "label": next_line,
                            "generated": first_line
                        }

                        os.makedirs(f"results_new/pipeline/{model_name}/{mode}/{language}", exist_ok=True)

                        with open(f"results_new/pipeline/{model_name}/{mode}/{language}/{level}.jsonl", "a") as f:
                            f.write(json.dumps(res) + "\n")
                        
                        if pbar_idx % 5 == 0 and pbar_idx != 0:
                            pbar.update(5)
        
    elif mode.startswith("top"):
        top = int(mode.split("top")[-1])
        # load the retriever
        if retriever == "unixcoder":
            retrieve_model = UniXcoder("microsoft/unixcoder-base")
            tokenizer = None
            sim = "cosine"
            max_length = 512
            # to second gpu
            retrieve_model.model.to("cuda:0")
        elif retriever == "codebert":
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", cache_dir="cache")
            retrieve_model = AutoModel.from_pretrained("microsoft/codebert-base", cache_dir="cache").to("cuda:0")
            sim = "cosine"
            max_length = 512
            retrieve_model.to("cuda:0")
        elif retriever == "codegpt":
            if language == "python":
                tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py-adaptedGPT2", cache_dir="cache")
                retrieve_model = AutoModel.from_pretrained("microsoft/CodeGPT-small-py-adaptedGPT2", cache_dir="cache")
            elif language == "java":
                tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-java-adaptedGPT2", cache_dir="cache")
                retrieve_model = AutoModel.from_pretrained("microsoft/CodeGPT-small-java-adaptedGPT2", cache_dir="cache")
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = tokenizer.eos_token_id
            sim = "cosine"
            max_length = 512
            retrieve_model.to("cuda:0")
        elif retriever == "codegen":
            if language == "python":
                tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono", cache_dir="cache")
                retrieve_model = AutoModel.from_pretrained("Salesforce/codegen-350M-mono", cache_dir="cache")
            elif language == "java":
                tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi", cache_dir="cache")
                retrieve_model = AutoModel.from_pretrained("Salesforce/codegen-350M-multi", cache_dir="cache")
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = tokenizer.eos_token_id
            sim = "cosine"
            max_length = 2048
            retrieve_model.to("cuda:0")
        elif retriever == "jaccard":
            tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono", cache_dir="cache")
            retrieve_model = None
            sim = "jaccard"
            max_length = 0
        elif retriever == "edit":
            tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono", cache_dir="cache")
            retrieve_model = None
            sim = "edit"
            max_length = 0
        
        for level, data in mapping.items():
            if resume_part == "cross_file_random" and level == "cross_file_first":
                continue
                
            if resume_part == "in_file" and level in ["cross_file_first", "cross_file_random"]:
                continue
            with tqdm(total=len(data), desc=f"Processing {level}") as pbar:
                for pbar_idx, dic in enumerate(data):
                    if level == resume_part and pbar_idx < resume:
                        pbar.update(1)
                        continue

                    # Crop the code first
                    import_statement = dic['import_statement']
                    code_path = dic['file_path']
                    code = dic['code']
                    code = "\n".join(code.split("\n")[-60:])
                    if language == "python":
                        code = f"# Path: {code_path}\n{import_statement}\n{code}"
                    elif language == "java":
                        if code.split("\n")[0].startswith("package"):
                            code = "\n".join(code.split("\n")[:1] + [import_statement] + code.split("\n")[1:])
                        else:
                            code = f"// Path: {code_path}\n{import_statement}\n{code}"
                        
                    code_tokens = encoding.encode(code)
                    if len(code_tokens) > 1600:
                        code_tokens = code_tokens[-1600:]
                        code = encoding.decode(code_tokens)

                    code_for_retrieval = crop_code_lines(dic['code'], int(top))
                    candidate_snippets = [dic['context'][i]['snippet'] for i in range(len(dic['context']))]

                    context = ""
                    context_length = len(code_tokens) + 1  # +1 for the newline between context and code
                    
                    rank = retrieve(code=code_for_retrieval, candidates=candidate_snippets, tokenizer=tokenizer, model=retrieve_model, max_length=max_length, similarity=sim)
                    for i in rank:
                        random_snippet_path = dic['context'][i]['path']
                        random_snippet_context = dic['context'][i]['snippet']
                        random_snippet_context = comment(random_snippet_context, language)
                        
                        if language == "python":
                            random_snippet_context = f"# Path: {random_snippet_path}\n{random_snippet_context}"
                        elif language == "java":
                            random_snippet_context = f"// Path: {random_snippet_path}\n{random_snippet_context}"
                        
                        random_snippet_context_tokens = encoding.encode(random_snippet_context)

                        context_length += len(random_snippet_context_tokens) + 1
                        
                        # Check the combined length
                        if context_length > 6400:
                            # crop the context
                            random_snippet_context_tokens = random_snippet_context_tokens[:6400 - context_length - 1]
                            random_snippet_context = encoding.decode(random_snippet_context_tokens)
                            context = context + "\n" + random_snippet_context
                            break

                        # Concatenate the contexts
                        context = context + "\n" + random_snippet_context
                        

                    next_line = dic['next_line']
                    prompt = f"{context}\n{code}"

                    if debug:
                        # print the first 10 prompts
                        if pbar_idx < print_num:
                            print(prompt)
                            print(f"####################\n{next_line}\n")

                        break
            
                    else:
                        # check whether we have already generated the result
                        if os.path.exists(f"results_new/pipeline/{model_name}/{retriever}/{language}/{level}.jsonl"):
                            # we can use idx to check whether we have already generated the result
                            with open(f"results_new/pipeline/{model_name}/{retriever}/{language}/{level}.jsonl", "r") as f:
                                lines = [json.loads(line.strip()) for line in f]
                                # get all the idx
                                idxs = [line["idx"] for line in lines]
                                if pbar_idx in idxs:
                                    pbar.update(1)
                                    continue
                        if model_name == "codex":
                            # query the model
                            reponse = query_with_retry(prompt)
                            # get the first line of the response
                            first_line = get_frist_line(reponse)
                        elif model_name == "starcoder":
                            # add prefix and suffix
                            prompt = prefix_token + prompt + suffix_token

                            # query the model
                            output = model.generate(
                                text=[prompt],
                                max_length=64,
                                include_prompt_in_result=False,
                                sampling_temperature=0.2
                            )[0]

                            first_line = get_first_line_not_comment(output, language)

                        res = {
                            "idx": pbar_idx,
                            "label": next_line,
                            "generated": first_line
                        }

                        os.makedirs(f"results_new/pipeline/{model_name}/{retriever}/{language}", exist_ok=True)

                        with open(f"results_new/pipeline/{model_name}/{retriever}/{language}/{level}.jsonl", "a") as f:
                            f.write(json.dumps(res) + "\n")
                        
                        if pbar_idx % 5 == 0 and pbar_idx != 0:
                            pbar.update(5)

    elif mode.startswith("reverse-top"):
        top = int(mode.split("top")[-1])
        # load the retriever
        if retriever == "unixcoder":
            retrieve_model = UniXcoder("microsoft/unixcoder-base")
            tokenizer = None
            sim = "cosine"
            max_length = 512
            # to second gpu
            retrieve_model.model.to("cuda:0")
        elif retriever == "codebert":
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", cache_dir="cache")
            retrieve_model = AutoModel.from_pretrained("microsoft/codebert-base", cache_dir="cache").to("cuda:0")
            sim = "cosine"
            max_length = 512
            retrieve_model.to("cuda:0")
        elif retriever == "codegpt":
            if language == "python":
                tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py-adaptedGPT2", cache_dir="cache")
                retrieve_model = AutoModel.from_pretrained("microsoft/CodeGPT-small-py-adaptedGPT2", cache_dir="cache")
            elif language == "java":
                tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-java-adaptedGPT2", cache_dir="cache")
                retrieve_model = AutoModel.from_pretrained("microsoft/CodeGPT-small-java-adaptedGPT2", cache_dir="cache")
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = tokenizer.eos_token_id
            sim = "cosine"
            max_length = 512
            retrieve_model.to("cuda:0")
        elif retriever == "codegen":
            if language == "python":
                tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono", cache_dir="cache")
                retrieve_model = AutoModel.from_pretrained("Salesforce/codegen-350M-mono", cache_dir="cache")
            elif language == "java":
                tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi", cache_dir="cache")
                retrieve_model = AutoModel.from_pretrained("Salesforce/codegen-350M-multi", cache_dir="cache")
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = tokenizer.eos_token_id
            sim = "cosine"
            max_length = 2048
            retrieve_model.to("cuda:0")
        elif retriever == "jaccard":
            tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono", cache_dir="cache")
            retrieve_model = None
            sim = "jaccard"
            max_length = 0
        elif retriever == "edit":
            tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono", cache_dir="cache")
            retrieve_model = None
            sim = "edit"
            max_length = 0
        
        for level, data in mapping.items():
            if resume_part == "cross_file_random" and level == "cross_file_first":
                continue
                
            if resume_part == "in_file" and level in ["cross_file_first", "cross_file_random"]:
                continue
            with tqdm(total=len(data), desc=f"Processing {level}") as pbar:
                for pbar_idx, dic in enumerate(data):
                    if level == resume_part and pbar_idx < resume:
                        pbar.update(1)
                        continue

                    # Crop the code first
                    import_statement = dic['import_statement']
                    code_path = dic['file_path']
                    code = dic['code']
                    code = "\n".join(code.split("\n")[-60:])
                    if language == "python":
                        code = f"# Path: {code_path}\n{import_statement}\n{code}"
                    elif language == "java":
                        if code.split("\n")[0].startswith("package"):
                            code = "\n".join(code.split("\n")[:1] + [import_statement] + code.split("\n")[1:])
                        else:
                            code = f"// Path: {code_path}\n{import_statement}\n{code}"
                        
                    code_tokens = encoding.encode(code)
                    if len(code_tokens) > 1600:
                        code_tokens = code_tokens[-1600:]
                        code = encoding.decode(code_tokens)

                    code_for_retrieval = crop_code_lines(dic['code'], int(top))
                    candidate_snippets = [dic['context'][i]['snippet'] for i in range(len(dic['context']))]

                    context = ""
                    context_length = len(code_tokens) + 1  # +1 for the newline between context and code
                    
                    rank = retrieve(code=code_for_retrieval, candidates=candidate_snippets, tokenizer=tokenizer, model=retrieve_model, max_length=max_length, similarity=sim)
                    for i in rank:
                        random_snippet_path = dic['context'][i]['path']
                        random_snippet_context = dic['context'][i]['snippet']
                        random_snippet_context = comment(random_snippet_context, language)
                        
                        if language == "python":
                            random_snippet_context = f"# Path: {random_snippet_path}\n{random_snippet_context}"
                        elif language == "java":
                            random_snippet_context = f"// Path: {random_snippet_path}\n{random_snippet_context}"
                        
                        random_snippet_context_tokens = encoding.encode(random_snippet_context)

                        context_length += len(random_snippet_context_tokens) + 1
                        
                        # Check the combined length
                        if context_length > 6400:
                            # crop the context
                            random_snippet_context_tokens = random_snippet_context_tokens[:6400 - context_length - 1]
                            random_snippet_context = encoding.decode(random_snippet_context_tokens)
                            context = random_snippet_context + "\n" + context
                            break

                        # Concatenate the contexts
                        context = random_snippet_context + "\n" + context
                        

                    next_line = dic['next_line']
                    prompt = f"{context}\n{code}"

                    if debug:
                        # print the first 10 prompts
                        if pbar_idx < print_num:
                            print(prompt)
                            print(f"####################\n{next_line}\n")

                        break
            
                    else:

                        # check whether we have already generated the result
                        if os.path.exists(f"results_new/pipeline/{model_name}/{retriever}-reverse/{language}/{level}.jsonl"):
                            # we can use idx to check whether we have already generated the result
                            with open(f"results_new/pipeline/{model_name}/{retriever}-reverse/{language}/{level}.jsonl", "r") as f:
                                lines = [json.loads(line.strip()) for line in f]
                                # get all the idx
                                idxs = [line["idx"] for line in lines]
                                if pbar_idx in idxs:
                                    pbar.update(1)
                                    continue

                        if model_name == "codex":
                            # query the model
                            reponse = query_with_retry(prompt)
                            # get the first line of the response
                            first_line = get_frist_line(reponse)
                        elif model_name == "starcoder":
                            # add prefix and suffix
                            prompt = prefix_token + prompt + suffix_token

                            # query the model
                            output = model.generate(
                                text=[prompt],
                                max_length=64,
                                include_prompt_in_result=False,
                                sampling_temperature=0.2
                            )[0]

                            first_line = get_first_line_not_comment(output, language)

                        res = {
                            "idx": pbar_idx,
                            "label": next_line,
                            "generated": first_line
                        }

                        os.makedirs(f"results_new/pipeline/{model_name}/{retriever}-reverse/{language}", exist_ok=True)

                        with open(f"results_new/pipeline/{model_name}/{retriever}-reverse/{language}/{level}.jsonl", "a") as f:
                            f.write(json.dumps(res) + "\n")
                        
                        if pbar_idx % 5 == 0 and pbar_idx != 0:
                            pbar.update(5)
    
    elif mode == "baseline":
        for level, data in mapping.items():
            if resume_part == "cross_file_random" and level == "cross_file_first":
                continue
                
            if resume_part == "in_file" and level in ["cross_file_first", "cross_file_random"]:
                continue
            with tqdm(total=len(data), desc=f"Processing {level}") as pbar:
                for pbar_idx, dic in enumerate(data):
                    if level == resume_part and pbar_idx < resume:
                        pbar.update(1)
                        continue

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
                    if len(code_tokens) > 6400:
                        code_tokens = code_tokens[-6400:]
                        code = encoding.decode(code_tokens)
                    
                    next_line = dic['next_line']
                
                    prompt = f"{code}"

                    if debug:
                        # print the first 10 prompts
                        if pbar_idx < print_num:
                            print(prompt)
                            print(f"####################\n{next_line}\n")

                        break
            
                    else:
                        # check whether we have already generated the result
                        if os.path.exists(f"results_new/pipeline/{model_name}/{mode}/{language}/{level}.jsonl"):
                            # we can use idx to check whether we have already generated the result
                            with open(f"results_new/pipeline/{model_name}/{mode}/{language}/{level}.jsonl", "r") as f:
                                lines = [json.loads(line.strip()) for line in f]
                                # get all the idx
                                idxs = [line["idx"] for line in lines]
                                if pbar_idx in idxs:
                                    pbar.update(1)
                                    continue
                        if model_name == "codex":
                            # query the model
                            reponse = query_with_retry(prompt)
                            # get the first line of the response
                            first_line = get_frist_line(reponse)
                        elif model_name == "starcoder":
                            # add prefix and suffix
                            prompt = prefix_token + prompt + suffix_token

                            # query the model
                            output = model.generate(
                                text=[prompt],
                                max_length=64,
                                include_prompt_in_result=False,
                                sampling_temperature=0.2
                            )[0]

                            first_line = get_first_line_not_comment(output, language)

                        res = {
                            "idx": pbar_idx,
                            "label": next_line,
                            "generated": first_line
                        }

                        os.makedirs(f"results_new/pipeline/{model_name}/{mode}/{language}", exist_ok=True)

                        with open(f"results_new/pipeline/{model_name}/{mode}/{language}/{level}.jsonl", "a") as f:
                            f.write(json.dumps(res) + "\n")
                        
                        if pbar_idx % 5 == 0 and pbar_idx != 0:
                            pbar.update(5)

if __name__ == "__main__":
    fire.Fire(main)