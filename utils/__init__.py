import argparse 
import pandas as pd 
import numpy as np 
from dataclasses import dataclass
from typing import (
    Iterable, 
    List, 
    Optional, 
    Dict, 
    Any, 
) 
from scipy.sparse import lil_matrix
from collections import defaultdict 
import anndata 
import json 
from importlib.util import find_spec
import os 
import warnings 

@dataclass(frozen=True)
class Template:
    """A data structure representing input and optional output templates."""

    input: str
    output: Optional[str] = None

def str2bool(v: str) -> bool:
    """Convert a string to a boolean value."""
    if isinstance(v, bool):
       return v
    if v.lower() in ("yes", "true", 't', 'y', '1'):
        return True
    elif v.lower() in ("no", "false", 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
def read_templates(
    template_dir_name: Optional[str] = None,  
    task_type: Optional[str] = None,  
) -> Dict[str, List[Template]]:
    """
    Read templates from a directory, structured by task type, and return a dict of template sets.

    The function looks for subdirectories within the given directory, each representing a different task type.
    Each subdirectory should contain a 'templates.json' file that stores template data with input instructions 
    and corresponding output responses. The templates are parsed and returned as a dictionary where the keys 
    are the task types and the values are lists of Template objects. If no directory is provided, the current 
    working directory is used. The function also allows filtering by a specific task type, so only that 
    subdirectory's templates are processed.

    Parameters
    ----------
    template_dir_name: str, optional, default None
        The name of the directory containing task subdirectories with 'templates.json' files. If not provided,
        the current directory is used by default.
    task_type: str, optional, default None 
        The specific task type to filter for. If provided, only the templates from the corresponding subdirectory 
        are read.

    Returns
    -------
    template_set: dict of str to list of Template
        A dict where keys are task types (subdirectory names) and values are lists of Template objects 
        containing input instructions and optional output responses.
    """
    if template_dir_name is None:
        warnings.warn(
            "No template directory is provided. Use the current directory.", 
            UserWarning
        )
        template_dir_name = '.'
    template_set = defaultdict(list)
    have_invalid_template = False

    for sub_dir in os.listdir(template_dir_name):
        if task_type is not None and sub_dir != task_type:
            continue 
        path = os.path.join(template_dir_name, sub_dir)
        if os.path.isdir(path):
            template_file_name = os.path.join(path, "templates.json")
            if os.path.exists(template_file_name):
                with open(template_file_name, 'r') as f:
                    templates = json.load(f)
                for template in templates:
                    if "instruction" not in template or "response" not in template:
                        have_invalid_template = True
                    input = template.get("instruction", "{input}")
                    output = template.get("response", "{output}")
                    template_set[sub_dir].append(Template(input, output=output))

    if have_invalid_template:
        warnings.warn(
            "Some templates are invalid. Please ensure that all templates have 'instruction' and 'response' keys. "
            "Those invalid templates will be replaced with the default template, namely '{input}' and '{output}'.", 
            UserWarning
        )
    return template_set

def find_duplicates_uppercase(inputs: Iterable[str]) -> List[bool]:
    """Find duplicates in the input list after converting all elements to uppercase."""
    visited = set() 
    res = []

    for x in inputs:
        x_ = x.upper()
        if x_ in visited:
            res.append(False)
        else:
            res.append(True)
            visited.add(x_)
    
    return res

def unify_gene_features(
    adata: anndata.AnnData, 
    gene_features: Iterable[str], 
    force_gene_symbol_uppercase: bool = True, 
) -> anndata.AnnData:
    """
    Unify the gene features of the AnnData object to match the provided gene features.

    This function adjusts the gene features in an AnnData object, ensuring that only the specified 
    gene features remain. Gene features present in the list but not in the AnnData object will be 
    added with empty data. Optionally, gene symbols can be converted to uppercase. 

    Parameters
    ----------
    adata: anndata.AnnData
        The AnnData object whose gene features need to be unified.
    gene_features: a sequence of str 
        A sequence of genes to unify within the AnnData object.
    force_gene_symbol_uppercase: bool, default True
        If True, both the gene symbols in the AnnData object and the input gene features will be 
        converted to uppercase. If False, case sensitivity will be maintained.

    Returns
    -------
    new_adata: anndata.AnnData
        A new AnnData object with unified gene features.

    Notes
    -----
    - Duplicates may arise if case conversion leads to identical gene symbols. In such cases, 
      a ValueError will be raised unless ``force_gene_symbol_uppercase`` is set to False.
    - Metadata in the AnnData object will not be altered, and the function does not operate 
      in-place, creating a new object instead.

    Examples
    --------
    >>> adata = anndata.AnnData(
    ...     X=np.array([[1, 2], [3, 4]]), 
    ...     var=pd.DataFrame(index=["GeneA", "GeneB"]), 
    ...     obs=pd.DataFrame({"state": ["normal", "cancer"]}, index=["CellA', 'CellB"]),
    ... )
    >>> gene_features = ["GENEA", "GENEC"]
    >>> new_adata = unify_gene_features(adata, gene_features, force_gene_symbol_uppercase=True)
    >>> new_adata.X.toarray()
    array([[1., 0.],
           [3., 0.]])
    >>> new_adata.var_names 
    Index(['GENEA', 'GENEC'], dtype='object')
    >>> new_adata.obs.to_dict()
    {'state': {'CellA': 'normal', 'CellB': 'cancer'}}
    """
    # drop duplicates 
    # TO DO: consider gene copies with different esmbl ids
    if force_gene_symbol_uppercase:
        adata = adata[:, find_duplicates_uppercase(adata.var_names)].copy()
        gene_features = [gene.upper() for gene in gene_features]
        if len(set(gene_features)) != len(gene_features):
            raise ValueError("Duplicates occur when gene symbols are capitalized. " + \
                              "To avert it, please remove duplicates in the gene features or set 'force_gene_symbol_uppercase' to False")
    
    new_var_names = pd.Index(gene_features)
    # lil_matrix is efficient for constructing sparse matrices incrementally
    new_X = lil_matrix((len(adata), len(new_var_names)))   
    
    # drop genes that are not in the gene vocabulary and order the genes
    var_features = {
        gene.upper() if force_gene_symbol_uppercase else gene: i for i, gene in enumerate(adata.var_names)
    }
    gene_features = np.array(gene_features)
    indexer = np.arange(len(gene_features))
    src_indexer = indexer[np.vectorize(lambda x: gene_features[x] in var_features)(indexer)]
    tgt_indexer = np.vectorize(lambda x: var_features[x])(gene_features[src_indexer])
    new_X[:, src_indexer] = adata.X[:, tgt_indexer]
    
    # convert the lil_matrix to csr_matrix
    # we don't change the metadata of the AnnData object
    return anndata.AnnData(
        X=new_X.tocsr(), 
        obs=adata.obs, 
        var=pd.DataFrame(index=new_var_names)
    ) 

def set_global_random_seed(seed: int, libraries: Optional[Iterable[str]] = None) -> None:
    """Set the global random seed for reproducibility."""
    if libraries is None:
        libraries = ["numpy", "torch", "random"]
    
    for lib in libraries:
        if lib == "numpy":
            if find_spec("numpy") is not None:
                import numpy as np
                np.random.seed(seed)
        elif lib == "torch":
            if find_spec("torch") is not None:
                import torch
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        elif lib == "random":
            import random
            random.seed(seed)
        else:
            raise ValueError(f"Library {lib} is not supported for setting the global seed.")
    
    return 

def parse_parameters(parameters: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Parse the model parameters."""
    model_parameters = {}
    for key, value in parameters.items():
        name, sub_key = key.split("::", 1)
        if name not in model_parameters:
            model_parameters[name] = {} 
        model_parameters[name][sub_key] = value
    return model_parameters
