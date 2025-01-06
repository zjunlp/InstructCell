<h1 align="center"> 🎨 InstructCell </h1>
<h3 align="center"> A Multi-Modal AI Copilot for Single-Cell Analysis with Instruction Following </h3>

[![Awesome](https://awesome.re/badge.svg)](https://github.com/zjunlp/InstructCell) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/zjunlp/InstructCell?color=green) 

## Table of Contents
- 🗞️ [Overview](#1)
- 🗝️ [Quick start](#2)
- 🚀 [How to run](#3)
- 🌻 [Acknowledgement](#4)
- 🔖 [Citation](#5)


<h2 id="1">🗞️ Overview</h2>

**InstructCell** is a multi-modal AI copilot that integrates natural language with single-cell RNA sequencing data, enabling researchers to perform tasks like cell type annotation, pseudo-cell generation, and drug sensitivity prediction through intuitive text commands. 
By leveraging a specialized multi-modal architecture and our multi-modal single-cell instruction dataset, InstructCell reduces technical barriers and enhances accessibility for single-cell analysis.

**InstructCell** has two versions:

1. **Chat Version**: Supports generating both detailed textual answers and single-cell data, offering comprehensive and context-rich outputs.
2. **Instruct Version**: Supports generating only the answer portion without additional explanatory text, providing concise and task-specific outputs.
   
Both versions of the model are available for download from Hugging Face ([zjunlp/InstructCell-chat](https://huggingface.co/zjunlp/InstructCell-chat) and [zjunlp/InstructCell-instruct](https://huggingface.co/zjunlp/InstructCell-instruct)).

<img width="1876" alt="image" src="https://github.com/user-attachments/assets/3fefe71c-3c00-4c21-b388-cf2300fb9f90" />


<h2 id="2">🗝️ Quick start</h2>

### 🪜 Requirements
- python 3.10 and above are recommended
- CUDA 11.7 and above are recommended

We provide a simple example for quick reference. This demonstrates a basic **cell type annotation** workflow. 

Make sure to specify the paths for `H5AD_PATH` and `GENE_VOCAB_PATH` appropriately:
- `H5AD_PATH`: Path to your `.h5ad` single-cell data file (e.g., `H5AD_PATH = "path/to/your/data.h5ad"`).
- `GENE_VOCAB_PATH`: Path to your gene vocabulary file (e.g., `GENE_VOCAB_PATH = "path/to/your/gene_vocab.npy"`).

```python
from mmllm.module import InstructCell
import anndata
import numpy as np
from utils import unify_gene_features

# Load the pre-trained InstructCell model from HuggingFace
model = InstructCell.from_pretrained("zjunlp/InstructCell-chat")

# Load the single-cell data (H5AD format) and gene vocabulary file (numpy format)
adata = anndata.read_h5ad(H5AD_PATH)
gene_vocab = np.load(GENE_VOCAB_PATH)
adata = unify_gene_features(adata, gene_vocab, force_gene_symbol_uppercase=False)

# Select a random single-cell sample and extract its gene counts and metadata
k = np.random.randint(0, len(adata)) 
gene_counts = adata[k, :].X.toarray()
sc_metadata = adata[k, :].obs.iloc[0].to_dict()

# Define the model prompt with placeholders for metadata and gene expression profile
prompt = (
    "Can you help me annotate this single cell from a {species}? " 
    "It was sequenced using {sequencing_method} and is derived from {tissue}. " 
    "The gene expression profile is {input}. Thanks!"
)

# Use the model to generate predictions
for key, value in model.predict(
    prompt, 
    gene_counts=gene_counts, 
    sc_metadata=sc_metadata, 
    do_sample=True, 
    top_p=0.95,
    top_k=50,
    max_new_tokens=256,
).items():
    # Print each key-value pair
    print(f"{key}: {value}")
```

For more detailed explanations and additional examples, please refer to the Jupyter notebook [demo.ipynb](https://github.com/zjunlp/InstructCell/blob/main/demo.ipynb).
  
<h2 id="3">🚀 How to run</h2>

Assume your current directory path is `DIR_PATH`. 

### 🧫 Collecting Raw Single-Cell Datasets

<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/b2002629-a2dc-4009-976e-f63fa6d4aec6" />
</div>

The datasets used in the paper are all publicly available. 
Detailed instructions and dataset links are provided in the Jupyter notebooks: [`HumanUnified.ipynb`](https://github.com/zjunlp/InstructCell/blob/main/HumanUnified.ipynb) and [`MouseUnified.ipynb`](https://github.com/zjunlp/InstructCell/blob/main/MouseUnified.ipynb). Below is a summary of the datasets and their corresponding details:


|Dataset|Species|Task|Data Repository|Download Link|
|:-------:|:-------:|:----:|:---------------:|:-------------:|
|Xin-2016|human|cell type annotation|GEO|https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114297|
|Segerstolpe-2016|human|cell type annotation|BioStudies|https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-5061|
|He-2020|human|cell type annotation|GEO|https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE159929|
|PBMC68K|human|conditional pseudo cell generation|Figshare|https://figshare.com/s/49b29cb24b27ec8b6d72|
|GSE117872|human|drug sensitivity prediction|GEO|https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117872|
|GSE149383|human|drug sensitivity predictio|Github|https://github.com/OSU-BMBL/scDEAL|
|Ma-2020|mouse|cell type annotation|GEO|https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140203|
|Bastidas-Ponce-2019|mouse|cell type annotation|GEO|https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132188|
|GSE110894|mouse|drug sensitivity predictio|GEO|https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE110894|
|Mouse-Atlas|mouse|conditional pseudo cell generation|GEO|https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4505404|

🔗 Please Note:

For the **He-2020** dataset, the cell type annotation file is sourced from the GitHub repository [scRNA-AHCA](https://github.com/bei-lab/scRNA-AHCA/tree/master/Cell_barcode_and_corresponding_cell_types_of_AHCA) 👈. 



### ⚙️ Installation Guide

Follow these steps to set up InstructCell:

1. Clone the repository:
```sh
git clone https://github.com/zjunlp/InstructCell.git
```
2. Set up a virtual environment and install the dependencies:
```sh
conda create -n instructcell python=3.10
conda activate instructcell
cd InstructCell
pip install -r requirements.txt
```

### 🌐 Downloading Pre-trained Language Models 
The pre-trained language model used in this project is **T5-base**. You can download it from 🤗 [Hugging Face](https://huggingface.co/google-t5/t5-base) and place the corresponding model directory under `DIR_PATH`.

Alternatively, you can use the provided script to automate the download process:
```sh
python download_script.py --repo_id google-t5/t5-base --parent_dir ..
```

### 🛠️ Single Cell Data Preprocessing
Navigate to the parent directory `DIR_PATH` and organize your data by creating a main data folder and three task-specific subfolders:
```sh
cd ..
mkdir data 
cd data
mkdir cell_type_annotation 
mkdir drug_sensitivity_prediction 
mkdir conditional_pseudo_cell_generation
cd ..
```

For dataset preprocessing, refer to the previously mentioned Jupyter notebooks:
- [HumanUnified.ipynb](https://github.com/zjunlp/InstructCell/blob/main/HumanUnified.ipynb) for human datasets.
- [MouseUnified.ipynb](https://github.com/zjunlp/InstructCell/blob/main/MouseUnified.ipynb) for mouse datasets.



> [!NOTE]
> Matching orthologous genes between mouse and human is based on [pybiomart](https://github.com/jrderuiter/pybiomart/tree/develop) and [pyensembl](https://github.com/openvax/pyensembl). Before preprocessing mouse datasets, ensure the corresponding Ensembl data are downloaded by running:
```sh
pyensembl install --release 100 --species mus_musculus
```

After completing the preprocessing steps, split each dataset and build a gene vocabulary using the following command: 
```sh
cd InstructCell
python preprocess.py --n_top_genes 3600 
```
To customize the size of the gene vocabulary, adjust the `n_top_genes` parameter as needed. For instance, setting it to 2000 will generate a smaller vocabulary.


### 🧺 Instruction-Response Template Construction

The instruction-response templates used in the projects are stored in this [folder](https://github.com/zjunlp/InstructCell/tree/main/exp_log/exp_templates).

<div align="center">

<img width="800" alt="image" src="https://github.com/user-attachments/assets/a58e5c62-c6dd-4fac-8677-c47c4cb7c093" />

</div>

The construction of instruction-response templates is divided into four stages:
1. **Motivation and personality generation**: In this stage, the large language model is prompted to generate potential motivations for each task and corresponding personalities. This step is implemented in the `data_synthesis.py` script.
2. **Template synthesis via parallel API calls**: Multiple APIs are run in parallel to synthesize templates, with each API invoked a specified number of times per task. This process is also implemented in the `data_synthesis.py` script.
3. **Merging synthesized templates**: The generated templates are consolidated into a unified collection using the `merge_templates.py` script.
4. **Filtering and splitting templates**: Finally, the templates are filtered for quality and divided into specific datasets using the `split_templates.py` script.


To execute all four stages in sequence, use the `run_data_synthesis.sh` script:
```sh
bash run_data_synthesis.sh  
```

> [!NOTE]
> Before executing `run_data_synthesis.sh`, ensure the parameters in the script are configured correctly. Update the API keys and base URL as needed, specify the model for template synthesis (`model` in the script), and adjust the number of API calls per task (`num_templates_for_task` in the script).


### 🚀 Training InstructCell 

<div align="center">
     <img width="650" alt="image" src="https://github.com/user-attachments/assets/82ed82c4-5d9d-4e84-9ce2-dc11fc4e560e" />
</div>

To train InstructCell, use the following command: 
```sh
torchrun --nproc_per_node=8 mm_train.py \
    --epochs 160 \
    --save_freq 20 \
    --batch_size 64 \
    --train_template_dir ../output/train_templates \
    --valid_template_dir ../output/valid_templates \
    --save_dir ../checkpoints \
    --best_model_dir ../trained_models \ 
    --train_no_extra_output_ratio 1.0 \
    --eval_no_extra_output_ratio 1.0
```
- To obtain the chat version of InstructCell, set both `train_no_extra_output_ratio` and `eval_no_extra_output_ratio` to 0. 
- To resume training from a specific checkpoint (`YOUR_CHECKPOINT_PATH`), include the flags `--resume True` and `--resume_path YOUR_CHECKPOINT_PATH`.
- For training on a single task and dataset, modify the `TASKS` parameter in `metadata.py`, retain only one dataset directory in the corresponding task folder, and set `--unify_gene False`.
- You can customize the architecture of InstructCell (e.g., the number of query tokens in Q-Former or the latent variable dimensions in the VAE) by modifying the `MODEL_PARAMETERS` in `metadata.py`.



### 📑 Evaluation
To evaluate the performance of InstructCell on conditional pseudo-cell generation, run:
```sh
python evaluate.py \
    --best_model_path ../trained_models/best_mm_model.pkl \
    --task_type "conditional pseudo cell generation" \
    --template_dir_name ../output/test_templates \
    --no_extra_output_ratio 1.0 
```
- To evaluate InstructCell on other tasks, modify the `task_type` parameter to `"cell type annotation"` or `"drug sensitivity prediction"` accordingly.
- To test InstructCell’s robustness to different task descriptions, add the flag `--evaluate_single_prompt True`. By default, 20 different task descriptions are used. To increase this number (e.g., to 40), include `--num_single_prompt 40`.
- If you want to evaluate only test templates that contain options, add `--provide_choices True`. By default, all test templates are evaluated.
- To evaluate the **chat** version of InstructCell, set the `no_extra_output_ratio` parameter to 0.0.  This will generate content formatted for xFinder’s JSON input requirements.  For detailed evaluation procedures using xFinder, please visit the [xFinder repository](https://github.com/IAAR-Shanghai/xFinder) 👈. Alternatively, you can refer to the [README](https://github.com/zjunlp/InstructCell/blob/main/xfinder/README.md) provided in [our repository](https://github.com/zjunlp/InstructCell/tree/main/xfinder) for additional guidance.

<!-- ## 🧬 Extracting Marker Genes -->

<!-- ## 🌠 Visualization --> 

<!-- ## 🎬 Demo  --> 

<h2 id="4">🌻 Acknowledgement</h2>

We would like to express our sincere gratitude to the excellent work [ALBEF](https://github.com/salesforce/ALBEF), [scvi-tools](https://github.com/scverse/scvi-tools).


<h2 id="5">🔖 Citation</h2>

## ✨ Contributors

<a href="https://github.com/zjunlp/InstructCell/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zjunlp/InstructCell" />
</a>

We will offer long-term maintenance to fix bugs and solve issues. So if you have any problems, please put issues to us.
