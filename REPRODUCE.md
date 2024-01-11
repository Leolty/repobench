<p align="center">
  <a href="https://github.com/Leolty/repobench#gh-light-mode-only">
    <img src="assets/repobench_dark.png" width="318px" alt="repobench logo" />
  </a>
  <a href="https://github.com/Leolty/repobench#gh-dark-mode-only">
    <img src="assets/repobench_light.png" width="318px" alt="repobench logo" />
  </a>

<p align="center">
  <a href="https://arxiv.org/abs/2306.03091">
    RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems
  </a>
</p>

<hr>

## 🚀 Reproducing the Results of RepoBench

This markdown file contains the instructions for reproducing the results of RepoBench. 

### ⚒️ Installation of Dependencies

```bash
pip install -r requirements.txt
```

> [!NOTE] 
> If you runninng into any issues regarding the dependencies, please follow the error message to install the missing packages accordingly.

### 📥 Downloading the Data

Follow the instructions in [README.md](./README.md) to download the data.

### 📊 Reproducing the Results

#### 🛍️ RepoBench-R (Retrieval)

Below is an example command for reproducing the results of RepoBench-R:

```bash
python run_repobench_r.py \
    --language python \
    --similarity cosine \
    --keep_lines 3 \
    --model_name microsoft/unixcoder-base \
    --max_length 512
```

**Arguments**:
- `--language`: The programming language of the dataset, can only be `python` or `java`.
- `--similarity`: The similarity metric used for ranking, can only be `cosine`, `jaccard`, or `edit`.
- `--keep_lines`: The number of lines to keep for matching code snippets, in the paper, we use `3`.
- `--model_name`: The name of the model to use, use model names on 🤗 HuggingFace, e.g, microsoft/unixcoder-base, microsoft/codebert-base, etc. 
    > If similarity is `edit` or `jaccard`, no need to specify the model name.
- `--max_length`: The maximum length of the input sequence. This should be related to the maximum length of the model. For example, `microsoft/unixcoder-base` has a maximum length of `512`.

#### 📝 RepoBench-C (Completion)

Below is an example command for reproducing the results of RepoBench-C, using [CTranslate2](https://github.com/OpenNMT/CTranslate2):

```bash
python inference_repobench_c_ctranslate.py \
    --model_name michaelfeil/ct2fast-starcoder \
    --language python \
    --length 2k \
    --max_new_tokens 64 \
    --temperature 0.2 \
    --resume_part cross_file_first \
    --resume 0
```

**Arguments**:
- `--model_name`: The name of the model to use, use ct2fast model names on [🤗 HuggingFace](https://huggingface.co/michaelfeil), e.g, michaelfeil/ct2fast-starcoder, michaelfeil/ct2fast-codegen-16B-mono, etc.
- `--language`: The programming language of the dataset, can only be `python` or `java`.
- `--length`: The length version of the dataset, can only be `2k` or `8k`.
- `--max_new_tokens`: The maximum number of tokens to generate, in the paper, we use `64` for efficiency.
- `--temperature`: The temperature for sampling, in the paper, we use `0.2`. 
    > [This paper](https://arxiv.org/abs/2107.03374) shows 0.2 is a good choice for code generation.
- `--resume_part`: The part of the dataset to resume from, default is `cross_file_first`, which means starting from cross-file-first setting. You can also choose `cross_file_random` or `in_file` to skip to the corresponding part. 
- `--resume`: The index of the data point to resume from, default is `0`, which means starting from the beginning of the dataset. 
    > `resume_part` and `resume` are used for resuming, since the process might be interrupted due to various reasons. 

> [!IMPORTANT] 
> Codex models were turned off on January 4th, 2024. So we will no longer be able to reproduce the results using Codex models.

#### 🔮 RepoBench-P (Pipeline)

> [!WARNING] 
> We apologize for the extensive repetition in the code of RepoBench-P and acknowledge that many areas require optimization. We plan to work on these issues as time permits.

Below is an example command for reproducing the results of RepoBench-P:

```bash
python inference_repobench_p.py \
    --language python \
    --mode top3 \
    --retriever unixcoder \
    --model_name starcoder \
    --resume_part cross_file_first \
    --resume 0
```

**Arguments**:
- `--language`: The programming language of the dataset, can only be `python` or `java`.
- `--mode`: The mode of the pipeline, we provide the following options:
    - `top3`: Use the previous 3 lines to retrieve and fill the cross-file context (ranked by similarity from high to low).
    - `reverse-top3`: Same as `top3`, but the cross-file context is filled from low to high.
        > The `3` in `top3` and `reverse-top3` can be changed to any positive integer.
    - `gold-only`: Only use the gold code snippet as cross-file context.
    - `gold-filled-head`: Put the gold code snippet at the beginning of the cross-file context, and fill the rest with randomly sampled code snippets.
    - `gold-filled-tail`: Put the gold code snippet at the end of the cross-file context, and fill the rest with randomly sampled code snippets.
    - `random`: Fill the cross-file context with randomly sampled code snippets.
    - `baseline`: No cross-file context is used.
- `--retriever`: The retriever to use, only support:
    - `unixcoder`
    - `codebert`
    - `codegpt`
    - `codegen`
    - `jaccard`
    - `edit`
- `--model_name`: The name of the model to use, it should support starcoder and codex, but since codex models were turned off now, we only support starcoder.
- `--resume_part` and `--resume`: Same as RepoBench-C, check the arguments above for details.

### 📊 Evaluating the Results

Simply input the path of the generated results to the `evaluate_completion.py` for completion and pipeline tasks, and `evaluate_retrieval.py` for retrieval task.
