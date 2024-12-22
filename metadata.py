import os 
from collections import OrderedDict 

DATA_DIR = "../data"
# count data file name 
COUNT_DATA_FILE_NAME = "rna.h5ad"

CTA = "cell type annotation"
DSP = "drug sensitivity prediction"
CPCG = "conditional pseudo cell generation"

CELL_TYPE_DIR = os.path.join(DATA_DIR, '_'.join(CTA.split()))
CELL_GENERATION_DIR = os.path.join(DATA_DIR, '_'.join(CPCG.split()))
DRUG_RESPONSE_DIR = os.path.join(DATA_DIR, '_'.join(DSP.split()))

TASKS = OrderedDict(
    [
        (CTA, CELL_TYPE_DIR),
        (DSP, DRUG_RESPONSE_DIR),
        (CPCG, CELL_GENERATION_DIR),
    ]
)

# fields for the metadata 
SPECIES = "species"
SEQUENCING_METHOD = "sequencing_method"
TISSUE = "tissue"
REFERENCE = "reference"
DRUG = "drug"
CHOICES = "choices"

MODEL_PARAMETERS = {
    "language_model::model_path": "../t5-base",
    "feature_decoder::use_layer_norm": "both",
    "feature_decoder::use_batch_norm": "none",
    "feature_decoder::n_latent": 256,
    "feature_decoder::condition_input_dim": 256, 
    "feature_decoder::log_variational": True, 
    "feature_decoder::n_layers": 4, 
    "feature_decoder::n_hidden": 1024, 
    "feature_decoder::dropout_rate": 0.1, 
    "feature_decoder::adaptive_library": True, 
    "feature_encoder::is_q_former_encoder": True, 
    "feature_encoder::cross_attention_frequency": 1,
    "feature_encoder::num_hidden_layers": 4,
    "feature_encoder::num_key_value_tokens": 6,
    "feature_encoder::num_blocks": 3,
    "feature_encoder::num_query_tokens": 8,
    "feature_encoder::hidden_dropout_prob": 0.1, 
}

TOTAL_SUM = 1e4
BASE = 10
MIN_GENES = 200
MIN_CELLS = 8   

UNDEFINED = "Undefined"
CELL_LABEL = "cell_type"
ORIGINAL_LABEL = "annotation"

# the minimum number of cells in a cluster for cell type annotation
MIN_CLUSTER_SIZE = 20 

# for drug sensitivity classification 
RESPONSE_LABEL = "drug_response"

# splitting
TRAIN_SIZE = 0.8  
SEED = 42

# gene features 
GENE_VOCAB_DIR = "../gene_vocab"

# options 
OPTION_DIR = "../choices"
OPTION_FILE_NAME = "choices.pkl"
