import os 

DATA_DIR = "../data"
# count data file name 
COUNT_DATA_FILE_NAME = "rna.h5ad"

CTA = "cell type annotation"
DSP = "drug sensitivity prediction"
CPCG = "conditional pseudo cell generation"

CELL_TYPE_DIR = os.path.join(DATA_DIR, '_'.join(CTA.split()))
CELL_GENERATION_DIR = os.path.join(DATA_DIR, '_'.join(CPCG.split()))
DRUG_RESPONSE_DIR = os.path.join(DATA_DIR, '_'.join(DSP.split()))

TASKS = {
    CTA: CELL_TYPE_DIR,
    DSP: DRUG_RESPONSE_DIR,
    CPCG: CELL_GENERATION_DIR,
}

# fields for the metadata 
SPECIES = "species"
SEQUENCING_METHOD = "sequencing_method"
TISSUE = "tissue"
REFERENCE = "reference"
DRUG = "drug"

DATASET_IDENTITY = {
    "Segerstolpe-2016": {
        "species": "human",
        "tissue": "pancreas (pancreatic islet)",
        "sequencing_method": "Smart-seq2",
        "tag": CTA, 
        "reference": "Segerstolpe Å, Palasantza A, Eliasson P, et al. Single-cell transcriptome profiling of human pancreatic islets in health and type 2 diabetes[J]. Cell metabolism, 2016, 24(4): 593-607."
    },
    "Xin-2016": {
        "species": "human",
        "tissue": "pancreas (pancreatic islet)",
        "sequencing_method": "SMARTer",
        "tag": CTA,  
        "reference": "Xin Y, Kim J, Okamoto H, et al. RNA sequencing of single human islet cells reveals type 2 diabetes genes[J]. Cell metabolism, 2016, 24(4): 608-615."
    }, 
    "He-2020-Liver": {
        "species": "human", 
        "tissue": "liver",
        "sequencing_method": "HiSeq X Ten System",
        "tag": CTA,
        "reference": "He S, Wang L H, Liu Y, et al. Single-cell transcriptome profiling of an adult human cell atlas of 15 major organs[J]. Genome biology, 2020, 21: 1-34."
    }, 
    "He-2020-Heart": {
        "species": "human",
        "tissue": "heart",
        "sequencing_method": "HiSeq X Ten System",
        "tag": CTA, 
        "reference": "He S, Wang L H, Liu Y, et al. Single-cell transcriptome profiling of an adult human cell atlas of 15 major organs[J]. Genome biology, 2020, 21: 1-34."
    }, 
    "GSE149383": {
        "species": "human",
        "tissue": "lung",
        "sequencing_method": "Drop-seq",
        "drug": "Erlotinib", 
        "tag": DSP, 
        "reference": "Aissa A F, Islam A B, Ariss M M, et al. Single-cell transcriptional changes associated with drug tolerance and response to combination therapies in cancer[J]. Nature communications, 2021, 12(1): 1628."
    },
    "GSE117872": {
        "species": "human", 
        "tissue": "oral cavity", 
        "sequencing_method": "Fluidigm C1",
        "drug": "Cisplatin", 
        "tag": DSP, 
        "reference": "Sharma A, Cao E Y, Kumar V, et al. Longitudinal single-cell RNA sequencing of patient-derived primary cells reveals drug-induced infidelity in stem cell hierarchy[J]. Nature communications, 2018, 9(1): 4931."
    }, 
    "PBMC68K": {
        "species": "human", 
        "tissue": "peripheral blood",
        "sequencing_method": "10xGenomics (GemCode Technology Platform)",
        "tag": CPCG, 
        "reference": "Zheng G X Y, Terry J M, Belgrader P, et al. Massively parallel digital transcriptional profiling of single cells[J]. Nature communications, 2017, 8(1): 14049."
    },
    "Tabular-Sapiens-Spleen": {
        "species": "human", 
        "tissue": "spleen",
        "sequencing_method": "10xGenomics (Chromium Next GEM Single Cell 3′ Kit v3.1)",
        "tag": CPCG, 
        "reference": "The Tabula Sapiens Consortium*, Jones R C, Karkanias J, et al. The Tabula Sapiens: A multiple-organ, single-cell transcriptomic atlas of humans[J]. Science, 2022, 376(6594): eabl4896."
    },
    "Tabular-Sapiens-Blood": {
        "species": "human", 
        "tissue": "peripheral blood",
        "sequencing_method": "10xGenomics (Chromium Next GEM Single Cell 3′ Kit v3.1)",
        "tag": CPCG, 
        "reference": "The Tabula Sapiens Consortium*, Jones R C, Karkanias J, et al. The Tabula Sapiens: A multiple-organ, single-cell transcriptomic atlas of humans[J]. Science, 2022, 376(6594): eabl4896."
    }, 
    "Tabular-Sapiens-Thymus": {
        "species": "human", 
        "tissue": "thymus",
        "sequencing_method": "10xGenomics (Chromium Next GEM Single Cell 3′ Kit v3.1)",
        "tag": CPCG, 
        "reference": "The Tabula Sapiens Consortium*, Jones R C, Karkanias J, et al. The Tabula Sapiens: A multiple-organ, single-cell transcriptomic atlas of humans[J]. Science, 2022, 376(6594): eabl4896."
    },
    "Tabular-Sapiens-Vasculature": {
        "species": "human", 
        "tissue": "vasculature",
        "sequencing_method": "10xGenomics (Chromium Next GEM Single Cell 3′ Kit v3.1)",
        "tag": CPCG, 
        "reference": "The Tabula Sapiens Consortium*, Jones R C, Karkanias J, et al. The Tabula Sapiens: A multiple-organ, single-cell transcriptomic atlas of humans[J]. Science, 2022, 376(6594): eabl4896."
    }, 
    "Tabular-Sapiens-Bladder": {
        "species": "human", 
        "tissue": "bladder", 
        "sequencing_method": "10xGenomics (Chromium Next GEM Single Cell 3′ Kit v3.1)",
        "tag": CPCG, 
        "reference": "The Tabula Sapiens Consortium*, Jones R C, Karkanias J, et al. The Tabula Sapiens: A multiple-organ, single-cell transcriptomic atlas of humans[J]. Science, 2022, 376(6594): eabl4896."
    }, 
    "Ma-2020": { 
        "species": "mouse", 
        "tissue": "skin (dorsal)", 
        "sequencing_method": "SHARE-seq",
        "tag": CTA, 
        "reference": "Ma S, Zhang B, LaFave L M, et al. Chromatin potential identified by shared single-cell profiling of RNA and chromatin[J]. Cell, 2020, 183(4): 1103-1116. e20."
    }, 
    "Bastidas-Ponce-2019": {
        "species": "mouse", 
        "tissue": "pancreas", 
        "sequencing_method": "10xGenomics (Chromium Single Cell 3' Library & Gel Bead Kit v2)",
        "tag": CTA, 
        "reference": "Bastidas-Ponce A, Tritschler S, Dony L, et al. Comprehensive single cell mRNA profiling reveals a detailed roadmap for pancreatic endocrinogenesis[J]. Development, 2019, 146(12): dev173849."
    }, 
    "GSE110894": {
        "species": "mouse",
        "tissue": "bone marrow",
        "sequencing_method": "Cel-Seq2",
        "drug": "BET inhibitor (I-BET-762)", 
        "tag": DSP, 
        "reference": "Bell C C, Fennell K A, Chan Y C, et al. Targeting enhancer switching overcomes non-genetic drug resistance in acute myeloid leukaemia[J]. Nature communications, 2019, 10(1): 2723."
    },
    "Mouse-Atlas-Liver": {
        "species": "mouse", 
        "tissue": "liver",
        "sequencing_method": "10xGenomics (GemCode Single-Cell 3' Gel Bead and Library V2 Kit)", 
        "tag": CPCG, 
        "reference": "A single-cell transcriptomic atlas characterizes ageing tissues in the mouse[J]. Nature, 2020, 583(7817): 590-595."
    }, 
    "Mouse-Atlas-Spleen": {
        "species": "mouse", 
        "tissue": "spleen",
        "sequencing_method": "10xGenomics (GemCode Single-Cell 3' Gel Bead and Library V2 Kit)", 
        "tag": CPCG, 
        "reference": "A single-cell transcriptomic atlas characterizes ageing tissues in the mouse[J]. Nature, 2020, 583(7817): 590-595."
    }, 
    "Mouse-Atlas-Thymus": {
        "species": "mouse", 
        "tissue": "thymus",
        "sequencing_method": "10xGenomics (GemCode Single-Cell 3' Gel Bead and Library V2 Kit)", 
        "tag": CPCG,  
        "reference": "A single-cell transcriptomic atlas characterizes ageing tissues in the mouse[J]. Nature, 2020, 583(7817): 590-595."
    }, 
    "Mouse-Atlas-Bladder": {
        "species": "mouse", 
        "tissue": "bladder",
        "sequencing_method": "10xGenomics (GemCode Single-Cell 3' Gel Bead and Library V2 Kit)", 
        "tag": CPCG, 
        "reference": "A single-cell transcriptomic atlas characterizes ageing tissues in the mouse[J]. Nature, 2020, 583(7817): 590-595."
    }, 
    "Mouse-Atlas-Lung": {
        "species": "mouse", 
        "tissue": "lung",
        "sequencing_method": "10xGenomics (GemCode Single-Cell 3' Gel Bead and Library V2 Kit)", 
        "tag": CPCG, 
        "reference": "A single-cell transcriptomic atlas characterizes ageing tissues in the mouse[J]. Nature, 2020, 583(7817): 590-595."
    }, 
    "Tabular-Muris-Lung": {
        "species": "mouse", 
        "tissue": "lung", 
        "sequencing_method": "10xGenomics (GemCode Single-Cell 3′ Gel Bead and Library V2 Kit)",
        "tag": CPCG, 
        "reference": "Schaum N, Karkanias J, Neff N F, et al. Single-cell transcriptomics of 20 mouse organs creates a Tabula Muris: The Tabula Muris Consortium[J]. Nature, 2018, 562(7727): 367."
    }, 
    "Tabular-Muris-Spleen": {
        "species": "mouse", 
        "tissue": "spleen", 
        "sequencing_method": "10xGenomics (GemCode Single-Cell 3′ Gel Bead and Library V2 Kit)",
        "tag": CPCG, 
        "reference": "Schaum N, Karkanias J, Neff N F, et al. Single-cell transcriptomics of 20 mouse organs creates a Tabula Muris: The Tabula Muris Consortium[J]. Nature, 2018, 562(7727): 367."
    }, 
    "Tabular-Muris-Tongue": {
        "species": "mouse", 
        "tissue": "tongue", 
        "sequencing_method": "10xGenomics (GemCode Single-Cell 3′ Gel Bead and Library V2 Kit)",
        "tag": CPCG, 
        "reference": "Schaum N, Karkanias J, Neff N F, et al. Single-cell transcriptomics of 20 mouse organs creates a Tabula Muris: The Tabula Muris Consortium[J]. Nature, 2018, 562(7727): 367."
    },
    "Tabular-Muris-Trachea": {
        "species": "mouse", 
        "tissue": "trachea", 
        "sequencing_method": "10xGenomics (GemCode Single-Cell 3′ Gel Bead and Library V2 Kit)",
        "tag": CPCG, 
        "reference": "Schaum N, Karkanias J, Neff N F, et al. Single-cell transcriptomics of 20 mouse organs creates a Tabula Muris: The Tabula Muris Consortium[J]. Nature, 2018, 562(7727): 367."
    },
}

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
MIN_CELLS_RATIO = 0.01
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

# gene token 
GENE_VOCAB_DIR = "../gene_vocab"

# options 
OPTION_DIR = "../choices"
OPTION_FILE_NAME = "choices.pkl"
