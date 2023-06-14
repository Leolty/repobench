import gzip
import pickle
from tqdm import tqdm
from typing import Union
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

def load_data(task:str, language:str, settings: Union[str, list]):
    """
    Load data from the specified task and language.

    Args:
        task: the task to load
        language: the language to load
        settings: the settings to load
    
    Returns:
        data: the loaded data
    """

    if task.lower() == 'r':
        task = 'retrieval'
    elif task.lower() == 'c':
        task = 'completion'
    elif task.lower() == 'p':
        task = 'pipeline'
    
    if language.lower() == 'py':
        language = 'python'
    
    if isinstance(settings, str):
        settings = [settings]
    
    for i, setting in enumerate(settings):
        if setting.lower() == 'xf-f':
            settings[i] = 'cross_file_first'
        elif setting.lower() == 'xf-r':
            settings[i] = 'cross_file_random'
        elif setting.lower() == 'if':
            settings[i] = 'in_file'
        

    # some assertions
    assert task.lower() in ['r', 'c', 'p', 'retrieval', 'completion', 'pipeline'], \
        "task must be one of R, C, or P"
    

    assert language.lower() in ['python', 'java', 'py'], \
        "language must be one of python, java"

    
    if task == "retrieval":
        assert all([setting.lower() in ['cross_file_first', 'cross_file_random'] for setting in settings]), \
            "For RepoBench-R, settings must be one of xf-f or xf-r"
    else:
        assert all([setting.lower() in ['cross_file_first', 'cross_file_random', 'in_file'] for setting in settings]), \
            "Settings must be one of xf-f, xf-r, or if"
    

    # load data
    data = {}
    # We split retrieval data into shards due to the github file size limit
    if task == "retrieval":
        for setting in tqdm(settings, desc=f"Loading data"):
            # we only further split the cross_file_first setting for java
            if setting == "cross_file_first" and language == "java":
                dic = {
                    "train": {},
                    "test": {}
                }
                with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_easy_1.gz", 'rb') as f:
                    train_easy_1 = pickle.load(f)
                with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_easy_2.gz", 'rb') as f:
                    train_easy_2 = pickle.load(f)
                dic['train']['easy'] = train_easy_1['easy'] + train_easy_2['easy']

                with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_hard_1.gz", 'rb') as f:
                    train_hard_1 = pickle.load(f)
                with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_hard_2.gz", 'rb') as f:
                    train_hard_2 = pickle.load(f)
                dic['train']['hard'] = train_hard_1['hard'] + train_hard_2['hard']

                with gzip.open(f"{ROOT}/{task}/{language}/{setting}_test.gz", 'rb') as f:
                    test = pickle.load(f)
                dic['test'] = test['test']
        
            
                data[setting] = dic
        
            else:
                dic = {
                    "train": {},
                    "test": {}
                }
                with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_easy.gz", 'rb') as f:
                    train_easy = pickle.load(f)
                dic['train']['easy'] = train_easy['easy']

                with gzip.open(f"{ROOT}/{task}/{language}/{setting}_train_hard.gz", 'rb') as f:
                    train_hard = pickle.load(f)
                dic['train']['hard'] = train_hard['hard']

                with gzip.open(f"{ROOT}/{task}/{language}/{setting}_test.gz", 'rb') as f:
                    test = pickle.load(f)
                dic['test'] = test['test']
        
                data[setting] = dic

    else:
        for setting in tqdm(settings, desc=f"Loading data"):
            with gzip.open(f"{ROOT}/{task}/{language}/{setting}.gz", 'rb') as f:
                data[setting] = pickle.load(f)
    
    if len(settings) == 1:
        return data[settings[0]]
    else:
        return [data[setting] for setting in settings]
    
def construct_trainable_data(dic: dict, language: str = "python"):
    """
    construct the data that can be used for training

    Args:
        dic: the dictionary of data from load_data
        language: the language of the data
    
    Returns:
        return_data: the data that can be used for training (a list of dictionaries)
    """
    # data is a dictionary with keys: file_path, context, import_statement, code, next_line
    
    # check if the language is valid
    assert language in ["python", "java"], "language must be one of [python, java]"

    return_data = {}

    for key, data_list in dic.items():

        return_data[key] = []

        for data in data_list:
            # check if context is a string
            assert isinstance(data['context'], str), "context must be a string instead of snippet list"

            if language == "python":
                path = f"# Path: {data['file_path']}"
            
            elif language == "java":
                path = f"// Path: {data['file_path']}"

            d = f"""{data['context']}
{path}
{data['import_statement']}

{data['code']}"""

            label = data['next_line']

            return_data[key].append({ 'data': d, 'label': label })
        
    return return_data


# crop lines given a threshold and tokenized code
def crop_code_lines(code: Union[str, list], threshold: int):
    """
    Crop the code to the last threshold lines.

    Args:
        code: the code to crop (either a string or a list of tokens)
        threshold: the number of lines to keep
    
    Returns:
        code: the cropped code
    """

    # if the code is string, meaning it is not tokenized
    if isinstance(code, str):
        code = code.split('\n')

        # return the last threshold lines if the code is longer than the threshold
        if len(code) > threshold:
            return "\n".join(code[-threshold:])
        else:
            return "\n".join(code)
    
    # if the code is tokenized
    elif isinstance(code, list):
        # set the current number of lines to -1, since the last line is always a newline
        cur_lines = -1
        # iterate over the code list from the end to the beginning
        for i in range(len(code)-1, -1, -1):
            # "Ċ" is the token for newline 
            if "Ċ" in code[i]:
                cur_lines += 1
            
            # if the current number of lines reaches the threshold, 
            # return the code from the current index to the end (do not include the newline token)
            if cur_lines == threshold:
                return code[i+1:]
        
        # if the code is shorter than the threshold, return the whole code
        return code

# comment code
def comment(code: str, language: str):
    """
    Comment the code.

    Args:
        code: the code to comment
        language: the language of the code
    
    Returns:
        code: the commented code
    """
    if language == "python":
        return "\n".join([f"# {line}" for line in code.split("\n")])
    elif language == "java":
        return "\n".join([f"// {line}" for line in code.split("\n")])
    else:
        raise ValueError("language must be one of [python, java]")
        

