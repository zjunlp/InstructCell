import os 
import anndata 
import scanpy as sc 
from collections import Counter
import numpy as np 
from pathlib import Path
import pickle 
from metadata import (
    TASKS, 
    CELL_TYPE_DIR, 
    CELL_GENERATION_DIR, 
    CTA, 
    DSP,
    CELL_LABEL, 
    RESPONSE_LABEL, 
    DATASET_IDENTITY,
    TRAIN_SIZE, 
    SEED,
    GENE_VOCAB_DIR, 
    OPTION_DIR, 
    OPTION_FILE_NAME, 
    UNDEFINED, 
    COUNT_DATA_FILE_NAME, 
    SPECIES,   
)
import pandas as pd 
from utils import find_duplicates_uppercase, str2bool
import argparse
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_top_genes", type=int, default=3600)
    parser.add_argument("--random_state", type=int, default=SEED)
    parser.add_argument("--force_gene_symbol_uppercase", type=str2bool, default=False)

    args = parser.parse_args()
    n_top_genes = args.n_top_genes
    random_state = np.random.RandomState(args.random_state)
    force_gene_symbol_uppercase = args.force_gene_symbol_uppercase

    # construct gene vocabulary
    options = {} 
    counter = Counter() 
    gene2species = {} 
    for task_dir in TASKS.values():
        # the order of os.listdir is random, so we need to sort the subdirectories
        for sub_dir in sorted(os.listdir(task_dir)):
            path = os.path.join(task_dir, sub_dir)
            if os.path.isdir(path):
                adata = anndata.read_h5ad(os.path.join(path, COUNT_DATA_FILE_NAME))
                print(f"Processing {sub_dir} in {task_dir} ...")
                if (adata.X.astype(np.int32) != adata.X).sum() != 0:
                    # get the raw count data 
                    print("The count data are not integers. Use the raw count data instead.")
                    var2index = {var_name: i for i, var_name in enumerate(adata.raw.var_names)}
                    X = adata.raw.X[:, [var2index[var_name] for var_name in adata.var_names if var_name in var2index]]
                    assert X.shape == adata.X.shape, "The raw count data should have the same genes as the count data."
                    adata.X = X
                # drop duplicates 
                # TO DO: consider those that share the same gene symbol but have different ids. 
                if force_gene_symbol_uppercase:
                    adata = adata[:, find_duplicates_uppercase(adata.var_names)].copy()
                sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
                for gene in adata.var.index[adata.var["highly_variable"].values]:
                    if force_gene_symbol_uppercase:
                        counter[gene.upper()] += 1
                    else:
                        counter[gene] += 1
                    # consider copies of the same gene symbol
                    current_species = adata.var.loc[gene, SPECIES]
                    if isinstance(current_species, pd.Series):
                        current_species = current_species.values[0]
                    if gene not in gene2species:
                        gene2species[gene] = current_species
                    else:
                        if gene2species[gene] != current_species:
                            print(f"Warning: {gene} has multiple species: {gene2species[gene]} and {current_species}")
                            assert_mouse = False 
                            # some gene symbols like H19 are shared by both human and mouse
                            # but it is not in pybiomart Dataset 
                            for ch in gene:
                                if ch.isalpha() and 'a' <= ch <= 'z':
                                    assert_mouse = True 
                                    break
                            if assert_mouse:
                                gene2species[gene] = "mouse"
                            else:
                                gene2species[gene] = "human"

                tag = DATASET_IDENTITY[sub_dir]["tag"]
                # for those classification tasks, we need to save the corresponding options
                if tag == DSP:
                    options[sub_dir] = np.unique(adata.obs[RESPONSE_LABEL].values).tolist()
                elif tag == CTA:
                    options[sub_dir] = [
                        label for label in np.unique(adata.obs[CELL_LABEL].values).tolist() if label != UNDEFINED
                    ]
                if task_dir in [CELL_TYPE_DIR, CELL_GENERATION_DIR]:
                    adata = adata[adata.obs[CELL_LABEL] != UNDEFINED, :]
                indexer = random_state.permutation(len(adata))
                adata = adata[indexer]
                n_train = int(len(adata) * TRAIN_SIZE) 
                n_test = int(len(adata) * (1 - TRAIN_SIZE)) // 2
                adata[: n_train].write_h5ad(os.path.join(path, "train_" + COUNT_DATA_FILE_NAME))
                adata[n_train: n_train + n_test].write_h5ad(os.path.join(path, "valid_" + COUNT_DATA_FILE_NAME))
                adata[n_train + n_test: ].write_h5ad(os.path.join(path, "test_" + COUNT_DATA_FILE_NAME))
                print("Done!")
    
    # save the gene vocabulary
    # sort the genes so we can reproduce the result 
    gene_vocab = np.array([gene for gene in sorted(counter.keys())])
    species = np.array([gene2species[gene] for gene in gene_vocab])
    print(f"Number of genes in the vocabulary: {len(gene_vocab)}")
    dir_name = Path(GENE_VOCAB_DIR)
    dir_name.mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(dir_name, "gene_vocab.npy"), gene_vocab)
    np.save(os.path.join(dir_name, "gene_species.npy"), species)

    # save options used in classification tasks
    dir_name = Path(OPTION_DIR)
    dir_name.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(dir_name, OPTION_FILE_NAME), "wb") as f:
        pickle.dump(options, f)
