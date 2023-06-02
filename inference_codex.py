"""
usage:

python inference_codex.py --language python
"""


from data.utils import load_data, construct_trainable_data
from tqdm import tqdm
from model.codex import query_with_retry, get_frist_line
import json
import fire
import os

def main(
        language:str # language of the data, python or java
):
    # load datasets
    settings = ["cross_file_first", "cross_file_random", "in_file"]
    cff,cfr,iff = load_data("completion", language, settings)

    # construct trainable data
    cff = construct_trainable_data(cff)
    cfr = construct_trainable_data(cfr)
    iff = construct_trainable_data(iff)

    # iterate over cff
    for d in tqdm(cff['test']):
        # query codex
        reponse = query_with_retry(d['data'])

        # get first line not comment
        generated = get_frist_line(reponse)

        # make directory
        os.makedirs(f"results/completion/codex/{language}", exist_ok=True)

        # save to file
        with open(f"results/completion/codex/{language}/cross_file_first.jsonl", "a") as f:
            f.write(json.dumps({"label": d['label'], "generated": generated}) + "\n")
    
    # iterate over cfr
    for d in tqdm(cfr['test']):
        # query codex
        reponse = query_with_retry(d['data'])

        # get first line not comment
        generated = get_frist_line(reponse)

        # save to file
        with open(f"results/completion/codex/{language}/cross_file_random.jsonl", "a") as f:
            f.write(json.dumps({"label": d['label'], "generated": generated}) + "\n")
    
    # iterate over iff
    for d in tqdm(iff['test']):
        # query codex
        reponse = query_with_retry(d['data'])

        # get first line not comment
        generated = get_frist_line(reponse)

        # save to file
        with open(f"results/completion/codex/{language}/in_file.jsonl", "a") as f:
            f.write(json.dumps({"label": d['label'], "generated": generated}) + "\n")


if __name__ == "__main__":
    fire.Fire(main)