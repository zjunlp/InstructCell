<h1 align="center"> 🎨 InstructCell </h1>
<b>InstructCell: A Multimodal AI Assistant for Natural Language-Driven Single-cell Analysis</b>

[![Awesome](https://awesome.re/badge.svg)](https://github.com/zjunlp/InstructCell) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/zjunlp/InstructCell?color=green) 

## Table of Contents
- 🗞️[Overview](#1)
- 🗝️[Quick Start](#2)
- 🔖[Citation](#3)

## 🪜 Requirements
- python 3.10 and above are recommended
- CUDA 11.7 and above are recommended 


<h2 id="1">🗞️ Overview</h2>
InstructCell is a multimodal AI assistant that integrates natural language with single-cell RNA sequencing (scRNA-seq) data, enabling researchers to perform tasks like cell type annotation, pseudo-cell generation, and drug sensitivity prediction through intuitive text commands. 
By leveraging a specialized multimodal architecture and our multimodal single-cell instruction dataset, InstructCell reduces technical barriers and enhances accessibility for single-cell analysis.

<h2 id="2">🗝️ Quick Start</h2>
Assume your current directory path is `DIR_PATH`. 
### 🧫 Collecting Raw Single Cell Datasets
The datasets used in the paper are all publicly available. The Jupyter notebooks, `HumanUnified.ipynb` and `MouseUnified.ipynb`, provide links to each dataset. The information for each dataset is as follows. 

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
|Tabular-Muris|mouse|conditional pseudo cell generation|Figshare|https://figshare.com/articles/dataset/Single-cell_RNA-seq_data_from_microfluidic_emulsion_v2_/5968960|

Note that, for He-2020, the cell type annotation file is downloaded from Github repository [scRNA-AHCA](https://github.com/bei-lab/scRNA-AHCA/tree/master/Cell_barcode_and_corresponding_cell_types_of_AHCA) 👈. 

### ⚙️ Installation 
Download the code: 
```
git clone https://github.com/zjunlp/InstructCell.git
```
Create a virtual environment and install the dependencies:
```
conda create -n instructcell python=3.10
conda activate instructcell
cd InstructCell
pip install -r requirements.txt
```

### 🌐 Downloading Pretrained Language Models 
The language model used in the paper is T5-base, you can download it from 🤗 [Hugging Face](https://huggingface.co/google-t5/t5-base) and place the corresponding model directory under `DIR_PATH`.

You can use `download_script.py` to download the model: 
```
python download_script.py --repo_id google-t5/t5-base --parent_dir ..
```

### 🛠️ Single Cell Data Preprocessing
Navigate back to the parent directory `DIR_PATH`, then create a data folder and three task-specific folders within it to organize your data：
```
cd ..
mkdir data 
cd data
mkdir cell_type_annotation 
mkdir drug_sensitivity_prediction 
mkdir conditional_pseudo_cell_generation
cd ..
```
We provide two Jupyter notebooks as references for preprocessing datasets. `HumanUnified.ipynb` is used for preprocessing human datasets, while `MouseUnified.ipynb` is used for preprocessing mouse datasets.

> [!NOTE]
> Matching orthologous genes between mouse and human is based on [pybiomart](https://github.com/jrderuiter/pybiomart/tree/develop) and [pyensembl](https://github.com/openvax/pyensembl). Therefore, before preprocessing mouse datasets, make sure the corresponding Ensembl data are downloaded. You can run the following command to download them: 
```
pyensembl install --release 100 --species mus_musculus
```

After preprocessing, split each dataset and build a gene vocabulary: 
```
cd InstructCell
python preprocess.py --n_top_genes 3600 
```
You can adjust the parameter `n_top_genes`, for example, to 2000, which will give you a smaller gene vocabulary.

### 🧺 Instruction-Response Templates Construction
The template construction process consists of four parts. The first part involves prompting the large language model to list possible motivations for each task and personalities (handled in `data_synthesis.py`). The second part is to run multiple APIs in parallel, where each API is called a certain number of times to synthesize some templates (handled in `data_synthesis.py`). The third part merges all the synthesized templates together (handled in `merge_templates.py`). The fourth part further filters the synthesized templates and then splits them (handled in `split_templates.py`). The `run_data_synthesis.sh` script covers these four steps, and you only need to execute the following command:
```
bash run_data_synthesis.sh  
```

> [!NOTE]
> Before running `run_data_synthesis.sh`, you need to modify the parameters in the file, such as the API keys and base URL being used. You can also adjust the model used for template synthesis (`model` in the script) and the number of times each API key is called for each task (`num_templates_for_task` in the script).

### 🚀 Training InstructCell 
You can run the following command to train InstructCell: 
```
torchrun --nproc_per_node=4 mm_train.py \
    --epochs 250 \
    --save_freq 50 \
    --batch_size 128 \
    --train_template_dir ../output/train_templates \
    --valid_template_dir ../output/valid_templates \
    --save_dir ../checkpoints \
    --best_model_dir ../trained_models \ 
    --train_no_extra_output_ratio 1.0 \
    --eval_no_extra_output_ratio 1.0
```
- You can get **chat** version of InstructCell by setting the parameters `train_no_extra_output_ratio` and `eval_no_extra_output_ratio` to 0. 
- To resume training from a specific checkpoint (path: `YOUR_CHECKPOINT_PATH`), add `--resume True` and `--resume_path YOUR_CHECKPOINT_PATH`.
- You can train InstructCell on a single task and a single dataset by modifying the `TASKS` in `metadata.py`, keeping only one dataset directory in the corresponding task directory, and adding `--unify_gene False`.
- You can adjust the architecture of InstructCell, such as the number of query tokens in Q-Former or the dimension of the latent variables in the VAE, simply by modifying the `MODEL_PARAMETERS` in `metadata.py`.

### 📑 Evaluation
Run the following command to evaluate performance of InstructCell on conditional pseudo cell generation: 
```
python evaluate.py \
    --best_model_path ../trained_models/best_mm_model.pkl \
    --task_type "conditional pseudo cell generation" \
    --template_dir_name ../output/test_templates \
    --no_extra_output_ratio 1.0 
```
- To evaluate the performance of InstructCell on other tasks, simply change the parameter `task_type`.
- To evaluate the robustness of InstructCell for task descriptions, you can add `--evaluate_single_prompt True`. By default, 20 different task descriptions are used. If you want to change the number, for example to 40, you can add `--num_single_prompt 40`.
- When evaluating InstructCell, you can adjust whether to just use the test templates which contain options. By default, all templates are used. You can add `--provide_choices True` to indicate that only test templates providing options should be used.
- To evaluate the **chat** version of InstructCell, set the parameter `no_extra_output_ratio` to 0.0. This will generate content in a format that matches the JSON file required by xFinder. Then, use xFinder for the evaluation. For usage details, please refer to the [xFinder repository](https://github.com/IAAR-Shanghai/xFinder) 👈. 

<!-- ## 🧬 Extracting Marker Genes -->

<!-- ## 🌠 Visualization --> 

<!-- ## 🎬 Demo  --> 


<h2 id="3">🔖 Citation</h2>

## ✨ Contributors

<a href="https://github.com/zjunlp/InstructCell/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zjunlp/InstructCell" /></a>

We will offer long-term maintenance to fix bugs and solve issues. So if you have any problems, please put issues to us.
