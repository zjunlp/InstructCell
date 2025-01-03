from mmllm.module import InstructCell
import anndata
import numpy as np
from utils import unify_gene_features


model = InstructCell.from_pretrained("zjunlp/InstructCell-chat") 
adata = anndata.read_h5ad("./exp_log/demo_data/cell_type_annotation/He-2020-Liver/rna.h5ad")
gene_vocab = np.load("./exp_log/gene_vocab.npy")
adata = unify_gene_features(adata, gene_vocab, force_gene_symbol_uppercase=False)
k = np.random.randint(0, len(adata)) 
gene_counts = adata[k, :].X.toarray()
sc_metadata = adata[k, :].obs.iloc[0].to_dict()
prompt = (
    "Can you help me annotate this single cell from a {species}? " 
    "It was sequenced using {sequencing_method} and is derived from {tissue}. " 
    "The gene expression profile is {input}. Thanks!"
)

for key, value in model.predict(
    prompt, 
    gene_counts=gene_counts, 
    sc_metadata=sc_metadata, 
    do_sample=True, 
    top_p=0.95,
    top_k=50,
    max_new_tokens=256,
).items():
    print(f"{key}: {value}")
  

# CPCG 
# adata = anndata.read_h5ad("./exp_log/demo_data/conditional_pseudo_cell_generation/PBMC68K/rna.h5ad")
# gene_vocab = np.load("./exp_log/gene_vocab.npy")
# adata = unify_gene_features(adata, gene_vocab, force_gene_symbol_uppercase=False)
# k = np.random.randint(0, len(adata)) 
# sc_metadata = adata[k, :].obs.iloc[0].to_dict()
# # prompt = (
# #     "Hey, can you whip up a single-cell gene profile for the given specs: cell type is {cell_type}, species is {species}, tissue is {tissue}, and sequencing method is {sequencing_method}? "
# # )

# prompt = (
#     "Hey, can you whip up a single-cell gene profile for the given specs: cell type is CD56+ NK, species is human, tissue is peripheral blood, and sequencing method is 10xGenomics (GemCode Technology Platform)? "
# )


# for key, value in model.predict(
#     prompt, 
#     # gene_counts=gene_counts, 
#     sc_metadata=sc_metadata, 
#     do_sample=True, 
#     top_p=0.95,
#     top_k=50,
#     max_new_tokens=256,
# ).items():
#     if not isinstance(value, str): 
#         value = '\n'.join(
#             f"{gene_vocab[idx]}: {int(value[idx])}" for idx in np.argsort(-value)[: 20]
#         )
#     print(f"{key}:\n{value}")
    
    
# DSP 
adata = anndata.read_h5ad("./exp_log/demo_data/drug_sensitivity_prediction/GSE110894/rna.h5ad")
gene_vocab = np.load("./exp_log/gene_vocab.npy")
adata = unify_gene_features(adata, gene_vocab, force_gene_symbol_uppercase=False)
k = np.random.randint(0, len(adata)) 
gene_counts = adata[k, :].X.toarray()
sc_metadata = adata[k, :].obs.iloc[0].to_dict()
prompt = (
    "Given {sequencing_method}, can you predict the response of the single cell {input} from {species} when exposed to {drug} in {tissue}?"
)

for key, value in model.predict(
    prompt, 
    gene_counts=gene_counts, 
    sc_metadata=sc_metadata, 
    do_sample=True, 
    top_p=0.95,
    top_k=50,
    max_new_tokens=256,
).items():
    print(f"{key}: {value}")