# Baseline

We provide the baseline for RepoBench. We provide the role of each python file, the exact usage can be seen in the comments of the files.

## RepoBench-R

- [run_retriever](run_retriever.py): the python script to run the baseline for RepoBench-R.

## RepoBench-C

- [finetune.py](finetune.py): the python script to finetune the code models for RepoBench-C.
- [inference_deepspeed.py](inference_deepspeed.py): the python script to inference models or checkpoints for RepoBench-C with DeepSpeed. (For parallel)
- [inference_hf.py](inference_hf.py): the python script to inference quantized 8bit models for RepoBench-C with HuggingFace Transformers. (For codegen-16B)
- [inference_codex.py](inference_codex.py): the python script to query code-davinci-002 for RepoBench-C.

## RepoBench-P

- [run_pipeline.py](run_pipeline.py): the python script to run pipeline by querying code-davinci-002 for RepoBench-P.

# Evaluation

Run the following command to evaluate the results. The `file_path` is the path to the result file, for RepoBench-R, it is a `.json` file, for RepoBench-C and RepoBench-P, it is a `.jsonl` file. Check how we save the results in each baseline file mentioned above.


```bash
python repobench/evaluation/evaluate.py --file_path /home/leo/repobench/results/retrieval/unixcoder/cross_file_first.json
```


