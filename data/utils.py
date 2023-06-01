import gzip
import pickle
from tqdm import tqdm
from typing import Union

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

    for setting in tqdm(settings, desc=f"Loading data"):
        with gzip.open(f"data/{task}/{language}/{setting}.gz", 'rb') as f:
            data[setting] = pickle.load(f)
    
    return [data[setting] for setting in settings]
    
    
        

