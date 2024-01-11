from evaluation.metrics import accuracy_at_k
import json
import random
import os
import fire

random.seed(42)


def main(
        dir = "results/unixcoder-base",
        print_random = False # whether to print the random baseline (sample 100 times)
    ):
    
    # get all the json files under the dir
    json_files = [pos_json for pos_json in os.listdir(dir) if pos_json.endswith('.json')]

    record = {
        'python': {
            'first': {
                'easy': {},
                'hard': {}
            },
            'random': {
                'easy': {},
                'hard': {}
            }
        },
        'java': {
            'first': {
                'easy': {},
                'hard': {}
            },
            'random': {
                'easy': {},
                'hard': {}
            }
        }
    }

    # loop through the files
    for file in json_files:
        # open the file and load the data
        with open(os.path.join(dir, file), "r") as f:
            res = json.load(f)
        
        # get the language, and subset
        language = file.split("_")[0]
        subset = file.split("_")[-1].split(".")[0]
        
        # test the easy retrieval
        gt, pred_3, pred_5, pred_10, pred_20, pred_30, pred_60, pred_120 = [], [], [], [], [], [], [], []
        # create a list with 10 empty lists
        rd = [[] for _ in range(100)]      
        for res_dic in res['easy']:
            gt.append(res_dic['ground_truth'])
            if print_random:
                for i in range(100):
                    rdm_get = list(range(len(res_dic['3'])))
                    random.shuffle(rdm_get)
                    rd[i].append(rdm_get)
            if '3' in res_dic:
                pred_3.append(res_dic['3'])
            if '5' in res_dic:
                pred_5.append(res_dic['5'])
            if '10' in res_dic:
                pred_10.append(res_dic['10'])
            if '20' in res_dic:
                pred_20.append(res_dic['20'])
            if '30' in res_dic:
                pred_30.append(res_dic['30'])
            if '60' in res_dic:
                pred_60.append(res_dic['60'])
            if '120' in res_dic:
                pred_120.append(res_dic['120'])
            
        # # average of the 10 random lists
        if print_random:
            easy_rd_acc_at_1, easy_rd_acc_at_3 = [], []
            for i in range(100):
                easy_rd_acc_at_1.append(accuracy_at_k(prediction_list=rd[i], golden_index_list=gt, k=1)*100)
                easy_rd_acc_at_3.append(accuracy_at_k(prediction_list=rd[i], golden_index_list=gt, k=3)*100)

            # # average of the 10 random lists
            easy_rd_acc_at_1 = sum(easy_rd_acc_at_1)/len(easy_rd_acc_at_1)
            easy_rd_acc_at_3 = sum(easy_rd_acc_at_3)/len(easy_rd_acc_at_3)
            record[language][subset]['easy']['rd'] = [easy_rd_acc_at_1, easy_rd_acc_at_3]

        if '3' in res_dic:
            easy_3_acc_at_1 = accuracy_at_k(prediction_list=pred_3, golden_index_list=gt, k=1)*100
            easy_3_acc_at_3 = accuracy_at_k(prediction_list=pred_3, golden_index_list=gt, k=3)*100
            record[language][subset]['easy']['3'] = [easy_3_acc_at_1, easy_3_acc_at_3]
        if '5' in res_dic:
            easy_5_acc_at_1 = accuracy_at_k(prediction_list=pred_5, golden_index_list=gt, k=1)*100
            easy_5_acc_at_3 = accuracy_at_k(prediction_list=pred_5, golden_index_list=gt, k=3)*100
            record[language][subset]['easy']['5'] = [easy_5_acc_at_1, easy_5_acc_at_3]
        if '10' in res_dic:
            easy_10_acc_at_1 = accuracy_at_k(prediction_list=pred_10, golden_index_list=gt, k=1)*100
            easy_10_acc_at_3 = accuracy_at_k(prediction_list=pred_10, golden_index_list=gt, k=3)*100
            record[language][subset]['easy']['10'] = [easy_10_acc_at_1, easy_10_acc_at_3]
        if '20' in res_dic:
            easy_20_acc_at_1 = accuracy_at_k(prediction_list=pred_20, golden_index_list=gt, k=1)*100
            easy_20_acc_at_3 = accuracy_at_k(prediction_list=pred_20, golden_index_list=gt, k=3)*100
            record[language][subset]['easy']['20'] = [easy_20_acc_at_1, easy_20_acc_at_3]
        if '30' in res_dic:
            easy_30_acc_at_1 = accuracy_at_k(prediction_list=pred_30, golden_index_list=gt, k=1)*100
            easy_30_acc_at_3 = accuracy_at_k(prediction_list=pred_30, golden_index_list=gt, k=3)*100
            record[language][subset]['easy']['30'] = [easy_30_acc_at_1, easy_30_acc_at_3]
        if '60' in res_dic:
            easy_60_acc_at_1 = accuracy_at_k(prediction_list=pred_60, golden_index_list=gt, k=1)*100
            easy_60_acc_at_3 = accuracy_at_k(prediction_list=pred_60, golden_index_list=gt, k=3)*100
            record[language][subset]['easy']['60'] = [easy_60_acc_at_1, easy_60_acc_at_3]
        if '120' in res_dic:
            easy_120_acc_at_1 = accuracy_at_k(prediction_list=pred_120, golden_index_list=gt, k=1)*100
            easy_120_acc_at_3 = accuracy_at_k(prediction_list=pred_120, golden_index_list=gt, k=3)*100
            record[language][subset]['easy']['120'] = [easy_120_acc_at_1, easy_120_acc_at_3]
    
        # test the hard dataset
        gt, pred_3, pred_5, pred_10, pred_20, pred_30, pred_60, pred_120 = [], [], [], [], [], [], [], []
        rd = [[] for _ in range(100)]
        for res_dic in res['hard']:
            gt.append(res_dic['ground_truth'])
            if print_random:
                for i in range(100):
                    rdm_get = list(range(len(res_dic['3'])))
                    random.shuffle(rdm_get)
                    rd[i].append(rdm_get)
            if '3' in res_dic:
                pred_3.append(res_dic['3'])
            if '5' in res_dic:
                pred_5.append(res_dic['5'])
            if '10' in res_dic:
                pred_10.append(res_dic['10'])
            if '20' in res_dic:
                pred_20.append(res_dic['20'])
            if '30' in res_dic:
                pred_30.append(res_dic['30'])
            if '60' in res_dic:
                pred_60.append(res_dic['60'])
            if '120' in res_dic:
                pred_120.append(res_dic['120'])
        
        # # average of the 10 random lists

        if print_random:
            hard_rd_acc_at_1, hard_rd_acc_at_3, hard_rd_acc_at_5 = [], [], []
            for i in range(100):
                hard_rd_acc_at_1.append(accuracy_at_k(prediction_list=rd[i], golden_index_list=gt, k=1)*100)
                hard_rd_acc_at_3.append(accuracy_at_k(prediction_list=rd[i], golden_index_list=gt, k=3)*100)
                hard_rd_acc_at_5.append(accuracy_at_k(prediction_list=rd[i], golden_index_list=gt, k=5)*100)


            hard_rd_acc_at_1 = sum(hard_rd_acc_at_1)/len(hard_rd_acc_at_1)
            hard_rd_acc_at_3 = sum(hard_rd_acc_at_3)/len(hard_rd_acc_at_3)
            hard_rd_acc_at_5 = sum(hard_rd_acc_at_5)/len(hard_rd_acc_at_5)
            record[language][subset]['hard']['rd'] = [hard_rd_acc_at_1, hard_rd_acc_at_3, hard_rd_acc_at_5]

        if '3' in res_dic:
            hard_3_acc_at_1 = accuracy_at_k(prediction_list=pred_3, golden_index_list=gt, k=1)*100
            hard_3_acc_at_3 = accuracy_at_k(prediction_list=pred_3, golden_index_list=gt, k=3)*100
            hard_3_acc_at_5 = accuracy_at_k(prediction_list=pred_3, golden_index_list=gt, k=5)*100
            record[language][subset]['hard']['3'] = [hard_3_acc_at_1, hard_3_acc_at_3, hard_3_acc_at_5]
        if '5' in res_dic:
            hard_5_acc_at_1 = accuracy_at_k(prediction_list=pred_5, golden_index_list=gt, k=1)*100
            hard_5_acc_at_3 = accuracy_at_k(prediction_list=pred_5, golden_index_list=gt, k=3)*100
            hard_5_acc_at_5 = accuracy_at_k(prediction_list=pred_5, golden_index_list=gt, k=5)*100
            record[language][subset]['hard']['5'] = [hard_5_acc_at_1, hard_5_acc_at_3, hard_5_acc_at_5]
        if '10' in res_dic:
            hard_10_acc_at_1 = accuracy_at_k(prediction_list=pred_10, golden_index_list=gt, k=1)*100
            hard_10_acc_at_3 = accuracy_at_k(prediction_list=pred_10, golden_index_list=gt, k=3)*100
            hard_10_acc_at_5 = accuracy_at_k(prediction_list=pred_10, golden_index_list=gt, k=5)*100
            record[language][subset]['hard']['10'] = [hard_10_acc_at_1, hard_10_acc_at_3, hard_10_acc_at_5]
        if '20' in res_dic:
            hard_20_acc_at_1 = accuracy_at_k(prediction_list=pred_20, golden_index_list=gt, k=1)*100
            hard_20_acc_at_3 = accuracy_at_k(prediction_list=pred_20, golden_index_list=gt, k=3)*100
            hard_20_acc_at_5 = accuracy_at_k(prediction_list=pred_20, golden_index_list=gt, k=5)*100
            record[language][subset]['hard']['20'] = [hard_20_acc_at_1, hard_20_acc_at_3, hard_20_acc_at_5]
        if '30' in res_dic:
            hard_30_acc_at_1 = accuracy_at_k(prediction_list=pred_30, golden_index_list=gt, k=1)*100
            hard_30_acc_at_3 = accuracy_at_k(prediction_list=pred_30, golden_index_list=gt, k=3)*100
            hard_30_acc_at_5 = accuracy_at_k(prediction_list=pred_30, golden_index_list=gt, k=5)*100
            record[language][subset]['hard']['30'] = [hard_30_acc_at_1, hard_30_acc_at_3, hard_30_acc_at_5]
        if '60' in res_dic:
            hard_60_acc_at_1 = accuracy_at_k(prediction_list=pred_60, golden_index_list=gt, k=1)*100
            hard_60_acc_at_3 = accuracy_at_k(prediction_list=pred_60, golden_index_list=gt, k=3)*100
            hard_60_acc_at_5 = accuracy_at_k(prediction_list=pred_60, golden_index_list=gt, k=5)*100
            record[language][subset]['hard']['60'] = [hard_60_acc_at_1, hard_60_acc_at_3, hard_60_acc_at_5]
        if '120' in res_dic:
            hard_120_acc_at_1 = accuracy_at_k(prediction_list=pred_120, golden_index_list=gt, k=1)*100
            hard_120_acc_at_3 = accuracy_at_k(prediction_list=pred_120, golden_index_list=gt, k=3)*100
            hard_120_acc_at_5 = accuracy_at_k(prediction_list=pred_120, golden_index_list=gt, k=5)*100
            record[language][subset]['hard']['120'] = [hard_120_acc_at_1, hard_120_acc_at_3, hard_120_acc_at_5]
        
    # print latex
    print("Python")
    # print rd as baseline
    if 'rd' in record['python']['first']['easy']:
        print(f" rd & {record['python']['first']['easy']['rd'][0]:.2f} & {record['python']['first']['easy']['rd'][1]:.2f} & {record['python']['random']['easy']['rd'][0]:.2f} & {record['python']['random']['easy']['rd'][1]:.2f} & {record['python']['first']['hard']['rd'][0]:.2f} & {record['python']['first']['hard']['rd'][1]:.2f} & {record['python']['first']['hard']['rd'][2]:.2f}& {record['python']['random']['hard']['rd'][0]:.2f} & {record['python']['random']['hard']['rd'][1]:.2f} & {record['python']['random']['hard']['rd'][2]:.2f} \\\\")
    for i in [3,5,10,20,30,60,120]:
        try:
            print(f"&{i} & {record['python']['first']['easy'][str(i)][0]:.2f} & {record['python']['first']['easy'][str(i)][1]:.2f} & {record['python']['random']['easy'][str(i)][0]:.2f} & {record['python']['random']['easy'][str(i)][1]:.2f} & {record['python']['first']['hard'][str(i)][0]:.2f} & {record['python']['first']['hard'][str(i)][1]:.2f} & {record['python']['first']['hard'][str(i)][2]:.2f}& {record['python']['random']['hard'][str(i)][0]:.2f} & {record['python']['random']['hard'][str(i)][1]:.2f} & {record['python']['random']['hard'][str(i)][2]:.2f} \\\\")
        except:
            pass
    
    print("Java")
    # print rd as baseline
    if 'rd' in record['java']['first']['easy']:
        print(f" rd & {record['java']['first']['easy']['rd'][0]:.2f} & {record['java']['first']['easy']['rd'][1]:.2f} & {record['java']['random']['easy']['rd'][0]:.2f} & {record['java']['random']['easy']['rd'][1]:.2f} & {record['java']['first']['hard']['rd'][0]:.2f} & {record['java']['first']['hard']['rd'][1]:.2f} & {record['java']['first']['hard']['rd'][2]:.2f}& {record['java']['random']['hard']['rd'][0]:.2f} & {record['java']['random']['hard']['rd'][1]:.2f} & {record['java']['random']['hard']['rd'][2]:.2f} \\\\")
    for i in [3,5,10,20,30,60,120]:
        try:
            print(f"&{i} & {record['java']['first']['easy'][str(i)][0]:.2f} & {record['java']['first']['easy'][str(i)][1]:.2f} & {record['java']['random']['easy'][str(i)][0]:.2f} & {record['java']['random']['easy'][str(i)][1]:.2f} & {record['java']['first']['hard'][str(i)][0]:.2f} & {record['java']['first']['hard'][str(i)][1]:.2f} & {record['java']['first']['hard'][str(i)][2]:.2f}& {record['java']['random']['hard'][str(i)][0]:.2f} & {record['java']['random']['hard'][str(i)][1]:.2f} & {record['java']['random']['hard'][str(i)][2]:.2f} \\\\")
        except:
            pass
if __name__ == '__main__':
    fire.Fire(main)
