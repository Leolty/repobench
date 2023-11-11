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


## 🛠️ Installation

```bash
git clone https://github.com/Leolty/repobench.git
cd repobench
```

> **Note**: There is no `requirements.txt` for this project. You can simply follow the error message to install the missing packages.

## ⚙️ Description of Settings

As discussed in the paper, we have three settings for each task:

- `cross_file_first`: Masks the line where a module from a different file is used for the first time.
- `cross_file_random`: Masks a random line where a module from a different file is used (not the first usage).
- `in_file`: Masks a random line that has no cross-file dependency.


## 📥 Load Data

To load the desired dataset, use the `load_data` utility function from the `data.utils` module.

### Download Data via Google Drive

1. Download the test data from [Google Drive](https://drive.google.com/file/d/1HvFFnOybTKEJCrEypWh4ftmW6DZBaiK_/view?usp=sharing), or simply run the following command:
   ```bash
    gdown --id '1HvFFnOybTKEJCrEypWh4ftmW6DZBaiK_' --output ./archive_data/test.zip
    unzip ./archive_data/test.zip -d ./archive_data/
    rm ./archive_data/test.zip
    ```

> **Note**: If you download through the browser, please make sure to unzip the file and place the `test` folder under `archive_data`, i.e., `archive_data/test`.
  
2. (Optional) If you also want to download the training data, it can be found [here](https://drive.google.com/file/d/179TXJBfMMbP9FDC_hsdpGLQPmN6iB4vY/view?usp=sharing). Similarly, you can run the following command:
   ```bash
    gdown --id '179TXJBfMMbP9FDC_hsdpGLQPmN6iB4vY' --output ./archive_data/train.zip
    unzip ./archive_data/train.zip -d ./archive_data/
    rm ./archive_data/train.zip
    ```

### How to Use

1. Import the `load_data` function:
   ```python
   from archive_data.utils import load_data
   ```

2. Call the function with the desired parameters:
   ```python
   data = load_data(split, task, language, settings, length)
   ```

**Parameters**:

- `split`: Specify whether you want the `train` or `test` split. 
- `task`: Choose between `retrieval`, `completion` and `pipeline`.
- `language`: Select the programming language, either `python` or `java`.
- `settings`: Choose between `cross_file_first`, `cross_file_random`, or `in_file`. You can also provide a list combining these settings.
- `length`: (Optional) For the `completion` task, please specify the length as either `2k` or `8k`.

**Return**:
- If a single setting is provided, the function returns the loaded data for that setting.
- If multiple settings are specified, the function returns a list containing data for each setting.

### Example

Load `completion` data (the `8k` version) for `Python` with `cross_file_first`, `cross_file_random`, and `in_file` settings:

```python
data = load_data(split='test', task='completion', language='python', settings=['cross_file_first', 'cross_file_random', 'in_file'], length='8k')
```


## 🚨 RepoBench Test Data Alert

🤯 **Data Leakage/Memorization:** The test data under `archive_data` folder holds code sourced from GitHub, created between February 9, 2023, and August 3, 2023. Please be vigilant: if your model's training data includes code from this timeframe, there's a risk of data leakage and memorization. This can jeopardize the integrity and trustworthiness of your model's evaluation results.

📅 **Stay Updated with RepoBench:** Our commitment/idea is to regularly update the RepoBench dataset. If you're looking for the most up-to-date code samples, keep an eye out for our releases. And if you feel the need to expedite the process or suggest a collaboration, don't hesitate! Feel free to raise an issue or drop us an email to give us a nudge. Collaborative efforts are always welcomed and appreciated.

> **Give us your knowledge cut-off, and we can provide the newest data.**

🔔 **A Note on Benchmarking:** We aim to provide the best, but due to computational constraints, we can't benchmark every data version. If you decide to use our dataset, be ready to benchmark against other competitors.

⚖️ **Licensing and Ownership:** While we strive for accuracy, licenses for the included repositories may not have been meticulously verified individually. If you identify license discrepancies or if your code is included but you'd prefer it wasn't, please reach out to us immediately.


## 📊 Baseline

In an effort to streamline our codebase, we've removed the baseline running code since model inference should ideally be uncomplicated. For improved efficiency, we suggest leveraging libraries like [CTranslate2](https://github.com/OpenNMT/CTranslate2) and [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ). If you require the removed code or have further inquiries, kindly open an issue or reach out to me via email.

## 📝 Citation

If you use RepoBench in your research, please consider citing us:

```bibtex
@misc{liu2023repobench,
      title={RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems}, 
      author={Tianyang Liu and Canwen Xu and Julian McAuley},
      year={2023},
      eprint={2306.03091},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```






