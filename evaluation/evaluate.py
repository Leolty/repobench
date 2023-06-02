import json
from collections import defaultdict
from metrics import accuracy_at_k, edit_similarity_score, exact_match_score

def main(file_path:str):
    # if jsonl
    if file_path.endswith(".jsonl"):
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    
    # if json
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            data = json.load(f)
    
    # else
    else:
        raise NotImplementedError
    
    # if retrieval
    if "retrieval" in file_path or file_path.endswith("json"):

        # get the list of keys
        kept_lines = list(data['easy'][0].keys())

        res_dic = defaultdict(list)

        for dic in data['easy']:
            for i in kept_lines:
                res_dic[i].append(dic[i])
        
        # compute accuracy
        for i in kept_lines:
            if i == "ground_truth":
                continue

            acc_at_1 = accuracy_at_k(res_dic[i], res_dic['ground_truth'], k=1)*100
            acc_at_3 = accuracy_at_k(res_dic[i], res_dic['ground_truth'], k=3)*100

            print(f"Kept {i} lines, acc@1: {acc_at_1:.2f}, acc@3: {acc_at_3:.2f}")
        
        # get the list of keys
        kept_lines = list(data['hard'][0].keys())

        res_dic = defaultdict(list)

        for dic in data['hard']:
            for i in kept_lines:
                res_dic[i].append(dic[i])
        
        # compute accuracy
        for i in kept_lines:
            if i == "ground_truth":
                continue

            acc_at_1 = accuracy_at_k(res_dic[i], res_dic['ground_truth'], k=1)*100
            acc_at_3 = accuracy_at_k(res_dic[i], res_dic['ground_truth'], k=3)*100
            acc_at_5 = accuracy_at_k(res_dic[i], res_dic['ground_truth'], k=5)*100

            print(f"Kept {i} lines, acc@1: {acc_at_1:.2f}, acc@3: {acc_at_3:.2f}, acc@5: {acc_at_5:.2f}")


    # if completion or pipeline
    elif "completion" in file_path or "pipeline" in file_path or file_path.endswith("jsonl"):

        preds, labels = zip(*[(d['generated'], d['label']) for d in data])

        # compute exact match
        print(f"Exact match: {exact_match_score(preds, labels)*100:.2f}")
        print(f"Edit similarity: {edit_similarity_score(preds, labels):.2f}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)

            
        
