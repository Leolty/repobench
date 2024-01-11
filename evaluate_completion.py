from evaluation.metrics import edit_similarity_score, exact_match_score
import os
import json
import fire
from collections import defaultdict

def main(
    model = "starcoderbase-2k",
    first_n = 1e10
):
    # results in model, we drop it
    if model.startswith("results"):
        model = model.split("/")[1]
        
    output_dir = f"results/{model}"
        
    for language in ["python", "java"]:
        em, es, counts = [], [], []
        
        
        # if exist output_dir/language, we further iterate over the files
        if os.path.exists(os.path.join(output_dir, language)):
            for level in ["cross_file_first", "cross_file_random", "in_file"]:
                
                # if exist, we load the data
                filepath = os.path.join(output_dir, language, f"{level}.jsonl")
                if not os.path.exists(filepath):
                    print(f"language: {language}, level: {level}" + " not found")
                    continue
                
                with open(filepath, "r") as f:
                    data = [json.loads(line.strip()) for line in f]
                
                # compute metrics
                data = data[:int(first_n)]

                # deduplicate based on idx
                try:
                    data = {d["idx"]: d for d in data}.values()
                except:
                    data = {d["data_idx"]: d for d in data}.values()

                ground_truth = [d["label"] for d in data]
                generated = [d["generated"] for d in data]
                edit_similarity = round(edit_similarity_score(ground_truth, generated),2)
                exact_match = round(exact_match_score(ground_truth, generated)*100,2)

                em.append(exact_match)
                es.append(edit_similarity)
                counts.append(len(data))
                
                print(f"language: {language}, level: {level}, count: {len(data)}")
                print(f"edit_similarity: {edit_similarity}")
                print(f"exact_match: {exact_match}")
                print()
        
        if len(em) == 3 and len(es) == 3 and len(counts) == 3:
            print(f"language: {language}")
            total_count = sum(counts)
            mean_em = round(sum(e*m for e, m in zip(em, counts))/total_count, 2)
            mean_es = round(sum(e*s for e, s in zip(es, counts))/total_count, 2)
            print(f"{em[0]:.2f} & {es[0]:.2f} & {em[1]:.2f} & {es[1]:.2f} & {em[2]:.2f} & {es[2]:.2f} & {mean_em:.2f} & {mean_es:.2f}", end="\\\\\n")

if __name__ == "__main__":
    fire.Fire(main)
