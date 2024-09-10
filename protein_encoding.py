import os    
import numpy as np 
from metadata import GENE_VOCAB_DIR 
import mygene 
from tqdm import tqdm 
import argparse 
from typing import Iterable, Optional 
from multiprocessing import Pool

def get_protein_encoding(
    gene_vocab: Iterable[str], 
    desc: Optional[str] = None, 
) -> np.ndarray:
    """Check if the gene is protein coding or not."""
    species = "human"
    mg = mygene.MyGeneInfo() 
    is_protein_coding = np.array([False] * len(gene_vocab))

    for i in tqdm(range(len(gene_vocab)), desc=desc):
        gene = gene_vocab[i] 
        res = mg.query(gene, species=species, fields="type_of_gene")
        if res.get("hits") is not None and len(res["hits"]) > 0:
            is_protein_coding[i] = res["hits"][0].get("type_of_gene") == "protein-coding"
    
    return is_protein_coding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene_vocab_path", type=str, required=True, help="path to the gene vocabulary") 
    parser.add_argument("--n_jobs", type=int, default=10, help="number of jobs to run in parallel")
    args = parser.parse_args()
    gene_vocab_path = os.path.join(GENE_VOCAB_DIR, f"{args.gene_vocab_path}.npy")
    gene_vocab = np.load(gene_vocab_path)
    
    n_jobs, num_genes = args.n_jobs, len(gene_vocab)
    intervals = np.linspace(0, num_genes, n_jobs + 1).astype(int)
    pool = Pool(processes=n_jobs)
    inputs = [
        (gene_vocab[intervals[i]: intervals[i + 1]], f"[{intervals[i]}, {intervals[i + 1]})") for i in range(n_jobs)
    ]
    results = pool.starmap(get_protein_encoding, inputs)
    pool.close()
    pool.join()
    is_protein_coding = np.concatenate(results, axis=0)
    np.save(
        os.path.join(GENE_VOCAB_DIR, f"is_protein_coding.npy"), 
        is_protein_coding
    )

    