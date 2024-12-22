import os 
import anndata 
import scanpy as sc 
from collections import Counter
import numpy as np 
from pathlib import Path
import pickle 
from metadata import (
    TASKS, 
    CTA, 
    DSP,
    CELL_LABEL, 
    RESPONSE_LABEL, 
    TRAIN_SIZE, 
    SEED,
    GENE_VOCAB_DIR, 
    OPTION_DIR, 
    OPTION_FILE_NAME, 
    UNDEFINED, 
    COUNT_DATA_FILE_NAME, 
    SPECIES,   
    TOTAL_SUM, 
    BASE, 
)
import pandas as pd 
from scipy.sparse import vstack  
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors  
from utils import (
    find_duplicates_uppercase, 
    str2bool, 
    unify_gene_features, 
) 
import argparse
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_top_genes", type=int, default=3600)
    parser.add_argument("--random_state", type=int, default=SEED)
    parser.add_argument("--force_gene_symbol_uppercase", type=str2bool, default=False)
    parser.add_argument("--drop_duplicates", type=str2bool, default=True) 

    args = parser.parse_args()
    n_top_genes = args.n_top_genes
    random_state = np.random.RandomState(args.random_state)
    force_gene_symbol_uppercase = args.force_gene_symbol_uppercase
    drop_duplicates = args.drop_duplicates 

    # construct gene vocabulary
    counter = Counter() 
    gene2species = {} 
    task_list, path_list, adata_list = [], [], []   
    for task, task_dir in TASKS.items():
        # the order of os.listdir is random, so we need to sort the subdirectories
        for sub_dir in sorted(os.listdir(task_dir)):
            path = os.path.join(task_dir, sub_dir)
            if os.path.isdir(path):
                adata = anndata.read_h5ad(os.path.join(path, COUNT_DATA_FILE_NAME))
                flavor = "seurat_v3" 
                print(f"Processing {sub_dir} in {task_dir} ...")
                if (adata.X.astype(np.int32) != adata.X).sum() != 0:
                    # get the raw count data  
                    var2index = {var_name: i for i, var_name in enumerate(adata.raw.var_names)}
                    X = adata.raw.X[:, [var2index[var_name] for var_name in adata.var_names if var_name in var2index]]
                    assert X.shape == adata.X.shape, "The raw count data should have the same genes as the count data."
                    if (X.astype(np.int32) != X).sum() == 0:
                        print("The count data are not integers. Use the raw count data instead.") 
                        adata.X = X
                    else:
                        flavor = "seurat"
                # drop duplicates 
                # TO DO: consider those that share the same gene symbol but have different ids
                if force_gene_symbol_uppercase:
                    adata = adata[:, find_duplicates_uppercase(adata.var_names)].copy()
                adata_ = adata.copy() 
                if flavor == "seurat":
                    sc.pp.normalize_total(adata_, target_sum=TOTAL_SUM)
                    sc.pp.log1p(adata_, base=BASE)
                    print("Seurat is used.")  
                sc.pp.highly_variable_genes(adata_, n_top_genes=n_top_genes, flavor=flavor)
                for gene in adata_.var.index[adata_.var["highly_variable"].values]:
                    if force_gene_symbol_uppercase:
                        counter[gene.upper()] += 1
                    else:
                        counter[gene] += 1
                    # consider copies of the same gene symbol
                    current_species = adata_.var.loc[gene, SPECIES]
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
                task_list.append(task)
                path_list.append(path)
                adata_list.append(adata)
                print("Done!")
    
    # save the gene vocabulary
    # sort the genes so we can reproduce the result 
    gene_vocab = np.array([gene for gene in sorted(counter.keys())])
    # species = np.array([gene2species[gene] for gene in gene_vocab])
    print(f"Number of genes in the vocabulary: {len(gene_vocab)}")
    dir_name = Path(GENE_VOCAB_DIR)
    dir_name.mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(dir_name, "gene_vocab.npy"), gene_vocab)
    # np.save(os.path.join(dir_name, "gene_species.npy"), species)

    if drop_duplicates:
        count_matrix = [] 
        for adata in adata_list:
            new_adata = unify_gene_features(
                adata, 
                gene_vocab, 
                force_gene_symbol_uppercase=force_gene_symbol_uppercase 
            )
            sc.pp.normalize_total(new_adata, target_sum=TOTAL_SUM)
            sc.pp.log1p(new_adata, base=BASE)
            count_matrix.append(new_adata.X)
        count_matrix = vstack(count_matrix) 
        pca = TruncatedSVD(n_components=5, n_iter=20, random_state=SEED)
        samples = pca.fit_transform(count_matrix)

    options = {}  
    # consider the samples are repeated at least 3 times 
    # deduplicate samples and split the data 
    start_idx = 0 
    for task, path, adata in zip(task_list, path_list, adata_list):
        sub_dir = os.path.split(path)[-1] 
        if drop_duplicates: 
            num_samples = len(adata) 
            end_idx = start_idx + num_samples 
            ne_pre = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
                samples[: start_idx] if start_idx > 0 else np.zeros((1, samples.shape[1]))  
            ) 
            current_samples = samples[start_idx: end_idx]    
            ne_cur = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(current_samples)
            distances, indices = ne_cur.kneighbors(current_samples)
            is_duplicated = distances[:, 1] == 0
            indices = indices[is_duplicated] 
            sample_ids = np.arange(num_samples)[is_duplicated] 
            is_duplicated[sample_ids[indices[indices != sample_ids[:, None]] > sample_ids]] = False  
            if (t := is_duplicated.sum()) > 0:
                print(f"For {sub_dir}, {t} samples are duplicated.")
            distances, _ = ne_pre.kneighbors(current_samples)
            is_duplicated_ = distances[:, 0] == 0
            if (t := is_duplicated_.sum()) > 0:
                print(f"For {sub_dir}, {t} samples are duplicated with the previous samples.")
            is_duplicated |= is_duplicated_
            adata_ = adata[~is_duplicated, :].copy() 
            print(f"For {sub_dir}, there are {num_samples} samples in total.", end=' ')
            print(f"{num_samples - len(adata_)} samples are removed now.")
        else:
            adata_ = adata.copy()
        if task != DSP:
            label_key = CELL_LABEL
        else:
            label_key = RESPONSE_LABEL 
        adata_ = adata_[adata_.obs[label_key] != UNDEFINED, :].copy()
        # for those classification tasks, we need to save the corresponding options 
        if task in [CTA, DSP]:
            options[sub_dir] = np.unique(adata_.obs[label_key].values).tolist()
        indexer = random_state.permutation(len(adata_))
        adata_ = adata_[indexer].copy() 
        n_train = int(len(adata_) * TRAIN_SIZE) 
        n_test = int(len(adata_) * (1 - TRAIN_SIZE)) // 2
        adata_train = adata_[: n_train].copy() 
        adata_train.write_h5ad(os.path.join(path, "train_" + COUNT_DATA_FILE_NAME))
        adata_valid = adata_[n_train: n_train + n_test].copy() 
        adata_valid.write_h5ad(os.path.join(path, "valid_" + COUNT_DATA_FILE_NAME))
        adata_test = adata_[n_train + n_test: ].copy() 
        adata_test.write_h5ad(os.path.join(path, "test_" + COUNT_DATA_FILE_NAME))
        start_idx = end_idx 

    # save options used in classification tasks
    dir_name = Path(OPTION_DIR)
    dir_name.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(dir_name, OPTION_FILE_NAME), "wb") as f:
        pickle.dump(options, f)
