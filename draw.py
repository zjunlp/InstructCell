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
    MODEL_PARAMETERS, 
) 
import matplotlib.pyplot as plt
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
from sklearn.decomposition import TruncatedSVD
from utils import str2bool, parse_parameters 
from utils.plotting import *
from umap import UMAP 
import warnings
import argparse

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_model_path", required=True, type=str, help="the file name of the best model")
    parser.add_argument("--task_type", required=True, type=str, help="the type of task")
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
    parser.add_argument("--unify_gene", default=True, type=str2bool, help="whether to unify gene symbols or not")
    parser.add_argument("--template_dir_name", default=None, type=str, help="the directory of evaluation templates")
    parser.add_argument("--batch_size", default=128, type=int, help="the batch size of the dataloader")
    args = parser.parse_args() 

    modality_tag = args.modality_tag
    num_signal_tokens = args.num_signal_tokens
    task_type = args.task_type
    force_gene_symbol_uppercase = args.force_gene_symbol_uppercase   
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
        provide_choices=None, 
        no_extra_output_ratio=1.0, 
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
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collator,
        num_workers=8, 
        sampler=SequentialSampler(dataset),
    )
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
    sc.pp.normalize_total(test_adata, target_sum=TOTAL_SUM)
    sc.pp.log1p(test_adata, base=BASE) 
    pca = TruncatedSVD(n_components=50, n_iter=20, random_state=SEED)
    estimator = UMAP(n_neighbors=40, random_state=SEED) 
    test_adata.obsm["X_pca"] = pca.fit_transform(test_adata.X)
    test_adata.obsm["X_umap"] = estimator.fit_transform(test_adata.obsm["X_pca"])
    res = [] 
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
        if task_type in [CTA, DSP]:
            res.append(np.array(outputs["texts"]))
        else:
            res.append(
                np.stack(
                    [output_cell if output_cell is not None else np.full(test_adata.shape[1], 0.0) for output_cell in outputs["cells"]]
                )
            )

    if task_type in [CTA, DSP]:
        if task_type == CTA:
            targets = test_adata.obs[CELL_LABEL].values
        else:
            targets = test_adata.obs[RESPONSE_LABEL].values.astype(str)
        predictions = np.concatenate(res, axis=0) 

        # plot confusion matrix
        if task_type == DSP: 
            cmap = "Blues"
        else:
            cmap = "BuPu"
        default_kwargs = {
            "cmap": cmap,
            "cbar": False,
            "square": True,
            "fmt": ".2f",
        }
        for source in test_adata.obs["_source"].unique():
            indices = test_adata.obs["_source"] == source
            source_predictions = predictions[indices]
            source_targets = targets[indices]
            labels = np.unique(source_targets) 
            if task_type == DSP:
                figsize = (4, 4)
                dpi = 180 
            elif source != "Ma-2020":
                figsize = (10, 10)
                dpi = 200
            else:
                figsize = (20, 20)
                dpi = 300
            if source != "Bastidas-Ponce-2019":
                default_kwargs["annot_kws"] = {"fontsize": 18}
            else:
                default_kwargs["annot_kws"] = {"fontsize": 14}
            if task_type == DSP:
                xticklabels_kwargs = {"fontsize": 15, "rotation": "horizontal"} 
                yticklabels_kwargs = {"fontsize": 15, "rotation": "vertical"}
            else:
                xticklabels_kwargs = {"fontsize": 18, "rotation": "vertical"}
                yticklabels_kwargs = {"fontsize": 18, "rotation": "horizontal"} 
            fig = plot_confusion_matrix(
                source_predictions, 
                source_targets, 
                labels=labels, 
                normalize=True, 
                figsize=figsize, 
                dpi=dpi, 
                xticklabels_kwargs=xticklabels_kwargs,
                yticklabels_kwargs=yticklabels_kwargs,
                heatmap_kwargs=default_kwargs,
            )
            fig.savefig(
                f"confusion_matrix_{source}.pdf", 
                bbox_inches="tight",
                dpi=200,
                format="pdf", 
            )
        
        # plot label distribution derived from predictions 
        if task_type == DSP:
            default_kwargs = {
                's': 9.0, 
                "alpha": 0.75,
            }
            palette = {
                "Resistant": "#E26982", 
                "Sensitive": "#37A3B3", 
                "Holiday": "#85658E", 
            }
            for source in test_adata.obs["_source"].unique():
                indices = test_adata.obs["_source"] == source
                xy = test_adata.obsm["X_umap"][indices] 
                source_targets = targets[indices]
                source_predictions = predictions[indices]
                fig = plot_label_distribution(
                    xy,
                    source_predictions,  
                    source_targets, 
                    figsize=(12, 6), 
                    dpi=300,
                    palette=palette, 
                    truth_title="Ground Truth",
                    pred_title="Predictions",
                    label_legend_kwargs={
                        "loc": "lower center", 
                        "title": "Response Label",
                        "ncols": len(np.unique(source_targets)),  
                    }, 
                    scatter_kwargs=default_kwargs,
                )
                fig.savefig(
                    f"drug_response_umap_pred_tgt_{source}.pdf", 
                    bbox_inches="tight", 
                    dpi=300, 
                    format="pdf", 
                )
        else:
            for source in test_adata.obs["_source"].unique():
                indices = test_adata.obs["_source"] == source
                xy = test_adata.obsm["X_umap"][indices]
                source_targets = targets[indices]
                source_predictions = predictions[indices]
                if source == "Ma-2020":
                    palette = "gnuplot"
                elif source == "Bastidas-Ponce-2019":
                    palette = "plasma"
                else:
                    palette = "gist_earth"
                default_kwargs = {
                    's': 5, 
                    "alpha": 0.75 if source != "Xin-2016" else 0.5, 
                }
                fig = plot_label_distribution(
                    xy,
                    source_predictions,  
                    source_targets, 
                    figsize=(12, 6), 
                    dpi=300,
                    palette=palette, 
                    truth_title="Ground Truth",
                    pred_title="Predictions",
                    label_legend_kwargs={
                        "loc": "lower center", 
                        "title": "Cell Type", 
                        "ncols": 4,
                    }, 
                    scatter_kwargs=default_kwargs,
                )
                if source == "Ma-2020":
                    fig.subplots_adjust(bottom=0.25)
                elif source == "Bastidas-Ponce-2019":
                    fig.subplots_adjust(bottom=0.15)
                fig.savefig(
                    f"cell_type_annotation_umap_pred_tgt_{source}.pdf", 
                    bbox_inches="tight", 
                    dpi=300, 
                    format="pdf", 
                )
    else:
        fake_samples = csr_matrix(np.concatenate(res, axis=0))
        fake_adata = anndata.AnnData(
            X=fake_samples, 
            obs=test_adata.obs, 
            var=test_adata.var
        )
        sc.pp.normalize_total(fake_adata, target_sum=TOTAL_SUM)
        sc.pp.log1p(fake_adata, base=BASE)
        fake_adata.obsm["X_pca"] = pca.transform(fake_adata.X)
        fake_adata.obsm["X_umap"] = estimator.transform(fake_adata.obsm["X_pca"])
        adata_all = anndata.concat([test_adata, fake_adata], axis=0)
        adata_all.obs["batch"] = ["Real"] * test_adata.shape[0] + ["Generated"] * fake_adata.shape[0]

        # visualize UMAP projections of real single-cell data
        # generated single-cell data, and their distribution differences 
        figures = plot_real_vs_generated_umap(
            test_adata, 
            fake_adata, 
            label_key=CELL_LABEL, 
            source_key="_source", 
            figsize=(8, 8), 
            dpi=300, 
            palette_for_real_generated=("#bf6c60", "#7298d1"), 
            palette_for_labels="icefire", 
            titles=('', '', ''), 
            umap_kwargs={
                "alpha": 0.4,
                "size": 5,
            }, 
        )
        for fig, filename in zip(
            figures, 
            [   
                "real_vs_fake.pdf", 
                "real_cells.pdf", 
                "generated_cells.pdf",
            ]
        ):
            fig.savefig(
                filename,
                dpi=300,
                bbox_inches="tight",
                format="pdf",
            )
        
        # visualize gene expression patterns 
        ori_global_font_size = plt.rcParams["font.size"]
        for source in test_adata.obs["_source"].unique():
            test_source_adata = test_adata[test_adata.obs["_source"] == source]
            fake_source_adata = fake_adata[fake_adata.obs["_source"] == source]
            cell_types = np.unique(test_source_adata.obs[CELL_LABEL].values)
            change_font_size = len(cell_types) > 24 
            if change_font_size:
                cell_types = cell_types[: 24] 
                plt.rcParams["font.size"] = 20
            if len(cell_types) > 20:
                figsize = (38, 17)
                dpi = 300
            else:
                figsize = (15, 10)
                dpi = 200
            dp_real, dp_fake = plot_gene_expression_patterns(
                test_source_adata, 
                fake_source_adata, 
                n_genes=3, 
                label_key=CELL_LABEL, 
                figsize=figsize, 
                dpi=dpi, 
                selected_labels=cell_types,
                dotplot_kwargs={
                    "standard_scale": "var",
                    "color_map": "coolwarm",
                    "dendrogram": False,
                },
            )
            if change_font_size:
                for size_legend in [
                    dp_real.ax_dict["size_legend_ax"], 
                    dp_fake.ax_dict["size_legend_ax"]
                ]:
                    for text_obj in size_legend.get_xticklabels():
                        text_obj.set_fontsize(12)
            dp_real.fig.savefig(
                f"real_gene_pattern_{source}.pdf", 
                bbox_inches="tight",
                dpi=dpi, 
                format="pdf", 
            )
            dp_fake.fig.savefig(
                f"fake_gene_pattern_{source}.pdf", 
                bbox_inches="tight",
                dpi=dpi, 
                format="pdf", 
            )
            if change_font_size:
                plt.rcParams["font.size"] = ori_global_font_size 