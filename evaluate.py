import torch.nn as nn 
import os    
import numpy as np 
from torch.utils.data import ( 
    SequentialSampler, 
    DataLoader, 
) 
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
    Blip2QFormerConfig, 
) 
from metadata import (
    TASKS, 
    CTA, 
    DSP,
    TOTAL_SUM, 
    BASE, 
    GENE_VOCAB_DIR, 
    CELL_LABEL, 
    RESPONSE_LABEL, 
    SEED, 
    OPTION_DIR, 
    OPTION_FILE_NAME, 
    MODEL_PARAMETERS, 
) 
import json 
from data_utils import TextCellDataset, TextCellCollator
from mmllm import prepare_cell_text_llm
from mmllm.module import (
    Generator, 
    CellTextLLM, 
    SCQFormer, 
)
from scvi.utils import init_library_size
from scipy.sparse import csr_matrix 
import torch 
import scanpy as sc 
import anndata 
import pickle 
from sklearn.decomposition import TruncatedSVD
from metrics import (
    compute_biased_mmd_rbf, 
    measure_bio_preservation, 
    measure_simulation, 
    measure_classification_accuracy_text, 
    measure_classification_f1_score_text, 
)
from copy import deepcopy 
from collections import defaultdict 
from utils import str2bool, parse_parameters 
from umap import UMAP 
from tqdm import tqdm 
import argparse 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_model_path", required=True, type=str, help="the file name of the best model")
    parser.add_argument("--task_type", required=True, type=str, help="the type of task")
    parser.add_argument("--output_path_suffix", default="all-outputs", type=str, help="the suffix of the output path")
    parser.add_argument("--device_id", default=0, type=int, help="The id of gpu to use")    
    parser.add_argument("--modality_tag", default="CELL", type=str, help="the name of added modality")
    parser.add_argument("--num_signal_tokens", default=1, type=int, help="the number of signal tokens")
    parser.add_argument("--gene_vocab_file_name", default="gene_vocab.npy", type=str, help="the gene vocabulary file name")
    parser.add_argument(
        "--force_gene_symbol_uppercase", 
        default=False, 
        type=str2bool, 
        help="whether to force gene symbols to be uppercase or not"
    )
    parser.add_argument(
        "--no_extra_output_ratio", 
        default=1.0, 
        type=float,
        help="the ratio of test samples without extra text outputs"
    )
    parser.add_argument("--provide_choices", default=None, type=str2bool, help="whether to provide choices or not")
    parser.add_argument("--unify_gene", default=True, type=str2bool, help="whether to unify gene symbols or not")
    parser.add_argument("--template_dir_name", default=None, type=str, help="the directory of evaluation templates")
    parser.add_argument("--batch_size", default=128, type=int, help="the batch size of the dataloader")
    parser.add_argument(
        "--evaluate_single_prompt",
        default=False,
        type=str2bool,
        help="whether to evaluate a single prompt or not"
    )
    parser.add_argument(
        "--num_single_prompt", 
        default=20,
        type=int, 
        help="the number of single prompts to evaluate"
    )
    args = parser.parse_args() 

    modality_tag = args.modality_tag
    num_signal_tokens = args.num_signal_tokens
    task_type = args.task_type
    force_gene_symbol_uppercase = args.force_gene_symbol_uppercase   
    no_extra_output_ratio = args.no_extra_output_ratio 
    provide_choices = args.provide_choices
    unify_gene = args.unify_gene 
    random_state = np.random.default_rng(SEED) 

    model_parameters = parse_parameters(MODEL_PARAMETERS)
    model_path = model_parameters["language_model"]["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    is_encoder_decoder = model.config.is_encoder_decoder 
    gene_vocab = np.load(os.path.join(GENE_VOCAB_DIR, args.gene_vocab_file_name)) if unify_gene else None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ignore_index = -100 if not hasattr(model.config, "ignore_index") else None
    pad_token_id = tokenizer.pad_token_id if not hasattr(model.config, "pad_token_id") else None
    template_dir_name = args.template_dir_name 

    assert task_type in TASKS, f"Task type {task_type} is not supported."

    dataset = TextCellDataset(
        dir_name=TASKS[task_type], 
        tokenizer=tokenizer, 
        task_type=task_type, 
        template_dir_name=template_dir_name, 
        split="test", 
        gene_vocab=gene_vocab, 
        modality=modality_tag, 
        num_signal_tokens=num_signal_tokens, 
        force_gene_symbol_uppercase=force_gene_symbol_uppercase, 
        provide_choices=provide_choices, 
        no_extra_output_ratio=no_extra_output_ratio, 
        is_encoder_decoder=is_encoder_decoder, 
        random_state=random_state, 
    )
    count_matrix = dataset.count_data.X
    count_dim = count_matrix.shape[1]

    # CVAE
    condition_input_dim = model_parameters["feature_decoder"]["condition_input_dim"]
    use_layer_norm = model_parameters["feature_decoder"]["use_layer_norm"]
    use_batch_norm = model_parameters["feature_decoder"]["use_batch_norm"]
    n_latent = model_parameters["feature_decoder"]["n_latent"]
    # if True, the library size is used as an observed covariate
    use_observed_lib_size = False  
    # to inject the conditional embedding into the encoder
    encode_covariates = True 
    deeply_inject_covariates = False 
    log_variational = model_parameters["feature_decoder"]["log_variational"]
    n_layers = model_parameters["feature_decoder"]["n_layers"]
    n_hidden = model_parameters["feature_decoder"]["n_hidden"]
    dropout_rate = model_parameters["feature_decoder"]["dropout_rate"]
    adaptive_library = model_parameters["feature_decoder"]["adaptive_library"]
    library_log_means, library_log_vars = init_library_size(count_matrix)

    best_model_path = args.best_model_path 
    is_q_former_encoder = model_parameters["feature_encoder"]["is_q_former_encoder"]
    if is_q_former_encoder:
        cross_attention_frequency = model_parameters["feature_encoder"]["cross_attention_frequency"]
        num_hidden_layers = model_parameters["feature_encoder"]["num_hidden_layers"]
        config = Blip2QFormerConfig(
            vocab_size=0, 
            hidden_size=model.config.hidden_size,
            hidden_dropout_prob=model_parameters["feature_encoder"]["hidden_dropout_prob"], 
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=model.config.num_attention_heads,
            intermediate_size=model.config.hidden_size * 4,
            pad_token_id=model.config.pad_token_id, 
            cross_attention_frequency=cross_attention_frequency, 
            encoder_hidden_size=model.config.hidden_size,
        )
        num_key_value_tokens = model_parameters["feature_encoder"]["num_key_value_tokens"]
        num_blocks = model_parameters["feature_encoder"]["num_blocks"]
        num_query_tokens = model_parameters["feature_encoder"]["num_query_tokens"]
        feature_encoder = SCQFormer(
            count_dim, 
            num_query_tokens, 
            num_key_value_tokens, 
            config, 
            num_hidden_layers=num_blocks,
        )
    else:
        feature_encoder = nn.Sequential(
            nn.Linear(count_dim, (count_dim + model.config.hidden_size) // 2),
            nn.GELU(),
            nn.Linear((count_dim + model.config.hidden_size) // 2, model.config.hidden_size),
            nn.Dropout(model_parameters["feature_encoder"]["hidden_dropout_prob"]),
        )
    feature_decoder = Generator(
        count_dim, 
        condition_dim=model.config.hidden_size,
        condition_input_dim=condition_input_dim,
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_latent=n_latent,
        dropout_rate=dropout_rate,
        use_layer_norm=use_layer_norm,
        use_batch_norm=use_batch_norm,
        encode_covariates=encode_covariates,
        deeply_inject_covariates=deeply_inject_covariates,
        log_variational=log_variational,
        adaptive_library=adaptive_library,
        use_observed_lib_size=use_observed_lib_size,
        library_log_means=library_log_means, 
        library_log_vars=library_log_vars, 
    )
    model, tokenizer = prepare_cell_text_llm(
        model, 
        tokenizer, 
        modality_tag=modality_tag,
        num_signal_tokens=num_signal_tokens, 
        ignore_index=ignore_index, 
        pad_token_id=pad_token_id,
        pad_to_multiple_of=8, 
    )
    collator = TextCellCollator(
        tokenizer, 
        pad_to_multiple_of=8, 
        model=model, 
    )
    batch_size = args.batch_size 
    mm_model = CellTextLLM(
        model, 
        tokenizer, 
        feature_encoder=feature_encoder, 
        feature_decoder=feature_decoder
    )

    device_id = args.device_id
    mm_model.load_state_dict(torch.load(best_model_path, map_location="cpu")) 
    mm_model = mm_model.to(f"cuda:{device_id}")
    # close the dropout layers in feature encoder 
    mm_model.eval()

    # ground truth 
    test_adata = dataset.count_data.copy()
    # if we evaluate a single prompt's performance across all test samples
    if args.evaluate_single_prompt:
        templates = dataset.templates[: args.num_single_prompt]
    else:
        templates = np.array(['#'])

    all_outputs = defaultdict(lambda: defaultdict(list))
    dataset_sources = np.unique(test_adata.obs["_source"].values)
    if task_type in [CTA, DSP]:
        if task_type == CTA:
            targets = test_adata.obs[CELL_LABEL].values
        else:
            targets = test_adata.obs[RESPONSE_LABEL].values.astype(str)
        choices_path = os.path.join(OPTION_DIR, OPTION_FILE_NAME)
        with open(choices_path, "rb") as f:
            choices = pickle.load(f) 
        choices = {source: choices.get(source) for source in dataset_sources} 
        for template_id, template in enumerate(tqdm(templates)):
            if template == '#':
                dataloader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    collate_fn=collator,
                    num_workers=8, 
                    sampler=SequentialSampler(dataset),
                )
            else:
                dataset_copy = deepcopy(dataset) 
                # we just simply replace the templates in the dataset 
                dataset_copy.templates[:] = template
                dataloader = DataLoader(
                    dataset_copy, 
                    batch_size=batch_size, 
                    collate_fn=collator,
                    num_workers=8, 
                    sampler=SequentialSampler(dataset_copy),
                )
            res = []  
            pointer = 0 
            for batch in dataloader:
                batch = {
                    key: value.to(device=next(mm_model.parameters()).device) if value is not None else value for key, value in batch.items()
                }
                # greedy decoding 
                outputs = mm_model.generate(
                    batch["input_ids"],
                    batch["input_counts"], 
                    do_sample=False,
                    max_new_tokens=512,
                )
                if no_extra_output_ratio == 1.0:
                    res.append(np.array(outputs["texts"]))
                else:
                    output_instances = outputs["texts"] 
                    input_instances = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                    # remove the prefix and suffix of the input instances
                    input_instances = [input_instance[6: -11] for input_instance in input_instances]
                    # save results in the format supported by xFinder 
                    for index in range(len(output_instances)):
                        source = test_adata.obs["_source"].values[pointer]
                        res.append(
                            {
                                "question": input_instances[index],
                                "llm_output": output_instances[index],
                                "model_name": "InstructCell", 
                                "key_answer_type": "categorical label",
                                "correct_answer": targets[pointer],
                                "dataset": source, 
                                "standard_answer_range": choices[source],
                            }
                        )
                        pointer += 1   
            if no_extra_output_ratio == 1.0:
                res = np.concatenate(res, axis=0)
                for source in dataset_sources: 
                    source_mask = test_adata.obs["_source"] == source
                    metric_dict = {
                        "accuracy": measure_classification_accuracy_text(res[source_mask], targets[source_mask]),
                        "average_f1": measure_classification_f1_score_text(res[source_mask], targets[source_mask], average="macro"),
                        "weighted_f1": measure_classification_f1_score_text(res[source_mask], targets[source_mask], average="weighted"),
                    }
                    for metric_name in metric_dict:
                        all_outputs[source][metric_name].append(metric_dict[metric_name])
            else:
                source_outputs = {
                    key: [] for key in choices
                }
                for item in res:
                    source_outputs[item["dataset"]].append(item)
                for source in source_outputs:
                    all_outputs[source][template_id] = source_outputs[source]
    else:
        dataset_sources = np.unique(test_adata.obs["_source"].values)
        sc.pp.normalize_total(test_adata, target_sum=TOTAL_SUM)
        sc.pp.log1p(test_adata, base=BASE)
        pca = TruncatedSVD(n_components=50, n_iter=20, random_state=SEED)
        estimator = UMAP(n_neighbors=40, random_state=SEED) 
        test_adata.obsm["X_pca"] = pca.fit_transform(test_adata.X)
        test_adata.obsm["X_umap"] = estimator.fit_transform(test_adata.obsm["X_pca"])

        k_list = [5, 10, 25, 50]

        for template in tqdm(templates):
            if template == '#':
                dataloader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    collate_fn=collator,
                    num_workers=8, 
                    sampler=SequentialSampler(dataset),
                )
            else:
                dataset_copy = deepcopy(dataset) 
                # we just simply replace the templates in the dataset 
                dataset_copy.templates[:] = template
                dataloader = DataLoader(
                    dataset_copy, 
                    batch_size=batch_size, 
                    collate_fn=collator,
                    num_workers=8, 
                    sampler=SequentialSampler(dataset_copy),
                )
            fake_samples = [] 
            for batch in dataloader:
                batch = {
                    key: value.to(device=next(mm_model.parameters()).device) if value is not None else value for key, value in batch.items()
                }
                outputs = mm_model.generate(
                    batch["input_ids"],
                    batch["input_counts"], 
                    do_sample=False,
                    max_new_tokens=512,
                )
                fake_samples.append(
                    np.stack(
                        [output_cell if output_cell is not None else np.full(test_adata.shape[1], 0.0) for output_cell in outputs["cells"]]
                    )
                )
            fake_samples = csr_matrix(np.concatenate(fake_samples, axis=0))
            fake_adata = anndata.AnnData(
                X=fake_samples, 
                obs=test_adata.obs, 
                var=test_adata.var
            )
            sc.pp.normalize_total(fake_adata, target_sum=TOTAL_SUM)
            sc.pp.log1p(fake_adata, base=BASE)
            fake_adata.obsm["X_pca"] = pca.transform(fake_adata.X)
            fake_adata.obsm["X_umap"] = estimator.transform(fake_adata.obsm["X_pca"]) 

            for source in dataset_sources:
                fake_source_adata = fake_adata[fake_adata.obs["_source"].str.startswith(source)]
                test_source_adata = test_adata[test_adata.obs["_source"].str.startswith(source)]
                metric_dict = {
                    "MMD": compute_biased_mmd_rbf(
                        fake_source_adata.obsm["X_umap"], 
                        test_source_adata.obsm["X_umap"],
                        n_neighbours=25, 
                    ), 
                }
                for k in k_list:
                    for metric_name, func in zip(
                        [f"sKNN ({k})", f"pKNN ({k})", f"sKNN ({k}) for Real Data"], 
                        [measure_bio_preservation, measure_simulation, measure_bio_preservation]
                    ): 
                        if not metric_name.endswith("Data"):
                            metric_dict[metric_name] = func(
                                predictions=fake_source_adata.obsm["X_umap"],
                                targets=test_source_adata.obsm["X_umap"], 
                                prediction_labels=fake_source_adata.obs[CELL_LABEL].values, 
                                target_labels=test_source_adata.obs[CELL_LABEL].values,
                                k=k,
                            )
                        else:
                            metric_dict[metric_name] = func(
                                predictions=test_source_adata.obsm["X_umap"],
                                prediction_labels=test_source_adata.obs[CELL_LABEL].values,
                                k=k,
                            )
                for metric_name in metric_dict:
                    all_outputs[source][metric_name].append(metric_dict[metric_name])

    for source in all_outputs:
        with open(f"{source}-{args.output_path_suffix}.json", 'w') as f:
            json.dump(all_outputs[source], f, indent=4) 
