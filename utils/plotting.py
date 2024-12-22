import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib.patches as mpatches 
from sklearn.metrics import confusion_matrix
import scanpy as sc 
from scanpy.get import rank_genes_groups_df 
from scanpy.pl import DotPlot 
import anndata  
from collections import OrderedDict 
import warnings 
from typing import (
    Tuple, 
    Optional, 
    Dict, 
    Any, 
    Iterable, 
) 

def _check_values(values: np.ndarray, valid_values: Iterable[Any]) -> bool:
    """Check if the values are valid."""
    return np.sum(np.vectorize(lambda value: value in valid_values)(values)) == len(values)

def plot_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,  
    labels: Optional[np.ndarray] = None,
    normalize: bool = True, 
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 200, 
    xticklabels_kwargs: Dict[str, Any] = {},
    yticklabels_kwargs: Dict[str, Any] = {},
    heatmap_kwargs: Dict[str, Any] = {},
) -> plt.Figure: 
    """Visualize the confusion matrix."""
    matrix = confusion_matrix(targets, predictions, labels=labels)
    if normalize:
        matrix = matrix / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    heatmap_kwargs = {
        **heatmap_kwargs, 
        "annot": matrix,        
        "xticklabels": labels,
        "yticklabels": labels,
        "ax": ax, 
    } 
    sns.heatmap(matrix, **heatmap_kwargs)

    for text_obj in ax.get_xticklabels():
        text_obj.set(**xticklabels_kwargs)
    for text_obj in ax.get_yticklabels():
        text_obj.set(**yticklabels_kwargs)

    return fig

def plot_label_distribution(
    xy: np.ndarray,
    predictions: np.ndarray, 
    targets: np.ndarray,
    sources: Optional[np.ndarray] = None, 
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 200, 
    palette: Dict[str, Any] | str = "plasma", 
    markers: Optional[Dict[str, str]] = None, 
    truth_title: str = "Ground Truth",
    pred_title: str = "Predictions",
    label_legend_kwargs: Dict[str, Any] = {}, 
    marker_legend_kwargs: Dict[str, Any] = {},
    scatter_kwargs: Dict[str, Any] = {},
) -> plt.Figure:
    """Visualize the predicted label distribution and ground truth distribution."""
    if xy.shape[1] != 2:
        raise ValueError("The shape of `xy` must be (n_samples, 2).")
    if len(targets) != len(xy):
        raise ValueError("The length of `targets` must be equal to the number of samples.")
    if len(predictions) != len(xy):
        raise ValueError("The length of `predictions` must be equal to the number of samples.")
    if sources is not None:
        if len(sources) != len(xy):
            raise ValueError("The length of `sources` must be equal to the number of samples.")
        if markers is None:
            raise ValueError("Please provide `markers` if `sources` are provided.")
        if not _check_values(sources, markers):
            raise ValueError("Please ensure each source has a marker.")
    
    label_set = set(targets)
    indices = np.vectorize(lambda label: label in label_set)(predictions)
    num_valid_predictions = indices.sum()
    if num_valid_predictions < len(predictions):
        warnings.warn(
            f"Found {len(predictions) - num_valid_predictions} invalid predictions.", 
            UserWarning
        )
        xy = xy[indices]
        targets = targets[indices]
        predictions = predictions[indices]
        if sources is not None:
            sources = sources[indices]
    label_set = sorted(label_set)
    if isinstance(palette, str):
        palette = sns.color_palette(palette, len(label_set)) 
        palette = {label: palette[i] for i, label in enumerate(label_set)}
    else:
        if not _check_values(label_set, palette):
            raise ValueError("Please ensure each label has a color.")
            
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    # a dummy plot to generate the legend for markers 
    if sources is not None:
        ref_ax = plt.subplots(figsize=(figsize[0] // 2, figsize[1]), dpi=dpi)[1]
    else:
        ref_ax = None 

    xs, ys = xy[:, 0], xy[:, 1]
    main_dim = label_set if ref_ax is None else sorted(np.unique(sources))
    for class_type in main_dim:
        if ref_ax is None:
            indices = targets == class_type
            basic_scatter_kwargs = {
                **scatter_kwargs,
                'x': xs[indices],
                'y': ys[indices],
                "color": palette[class_type],
                "label": class_type, 
            }
            axes[0].scatter(**basic_scatter_kwargs)
            indices = predictions == class_type
            basic_scatter_kwargs['x'] = xs[indices]
            basic_scatter_kwargs['y'] = ys[indices]
            axes[1].scatter(**basic_scatter_kwargs)
        else:
            indices = sources == class_type
            basic_scatter_kwargs = {
                **scatter_kwargs,
                'x': xs[indices],
                'y': ys[indices],
                'c': np.vectorize(lambda label: palette[label])(targets[indices]),
                "marker": markers[class_type],
            }
            axes[0].scatter(**basic_scatter_kwargs)
            basic_scatter_kwargs['c'] = np.vectorize(lambda label: palette[label])(predictions[indices])
            axes[1].scatter(**basic_scatter_kwargs)
            basic_scatter_kwargs['c'] = ["none"] * indices.sum()
            basic_scatter_kwargs["edgecolors"] = "black"
            basic_scatter_kwargs["label"] = class_type
            ref_ax.scatter(**basic_scatter_kwargs)
    axes[0].set_title(truth_title)
    axes[1].set_title(pred_title)

    for ax in axes:
        ax.axis("off")
        ax.set_xlabel('')
        ax.set_ylabel('')
        for spine in ax.spines.values():
            spine.set_visible(False)
    if ref_ax is not None:
        ref_ax.legend()
        handles, labels = ref_ax.get_legend_handles_labels()
        axes[0].legend(handles, labels, **marker_legend_kwargs)
        # generate each patch 
        patches = [
            mpatches.Patch(
                facecolor=palette[label],  
                edgecolor="none", 
                label=label, 
            ) for label in palette
        ]
        axes[1].legend(handles=patches, **label_legend_kwargs)
    else:
        axes[0].legend()
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].get_legend().remove()
        fig.legend(handles, labels, **label_legend_kwargs)

    return fig 

def plot_real_vs_generated_umap(
    real_adata: anndata.AnnData, 
    generated_adata: anndata.AnnData,
    label_key: str = "cell_type",
    source_key: Optional[str] = None, 
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 200,
    palette_for_real_generated: Tuple[str, str] = ("#bf6c60", "#7298d1"), 
    palette_for_labels: str = "icefire",
    titles: Tuple[str, str, str] = ('', '', ''),
    umap_kwargs: Dict[str, Any] = {}, 
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Visualize UMAP projections of real single-cell data, generated single-cell data, 
    and their distribution differences."""
    adata_all = anndata.concat([real_adata, generated_adata], axis=0)
    adata_all.obs["batch"] = ["Real"] * real_adata.shape[0] + ["Generated"] * generated_adata.shape[0]

    # visualize the difference between real and generated data 
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    kwargs = {
        **umap_kwargs,
        "palette": {
            "Real": palette_for_real_generated[0], 
            "Generated": palette_for_real_generated[1],  
        },
        "ax": ax,
        "color": ["batch"],
    }
    sc.pl.umap(adata_all, **kwargs)
    figures, axes = [fig], [ax]  

    # visualize the real data and generated data repsectively
    for adata in [real_adata, generated_adata]:
        adata = adata.copy() 
        if source_key is not None:
            adata.obs[label_key] = adata.obs.apply(
                lambda item: f"{item[label_key]} ({item[source_key]})", 
                axis=1
            )
        classes = adata.obs[label_key].unique()
        colors = sns.color_palette(palette_for_labels, len(classes)) 
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        kwargs = {
            **umap_kwargs,
            "palette": {classes[i]: colors[i] for i in range(len(classes))},
            "ax": ax,
            "color": [label_key],
        }
        sc.pl.umap(adata, **kwargs)
        figures.append(fig)
        axes.append(ax)
    
    for ax, title in zip(axes, titles):
        ax.set_title(title) 
    
    return tuple(figures) 
    
def plot_gene_expression_patterns(
    real_adata: anndata.AnnData,
    generated_adata: anndata.AnnData,
    n_genes: int = 3, 
    label_key: str = "cell_type",
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 200,
    selected_labels: Optional[Iterable[str]] = None,
    dotplot_kwargs: Dict[str, Any] = {},
) -> Tuple[DotPlot, DotPlot]: 
    """Visualize the gene expression pattern of real and generated single-cell data."""
    if selected_labels is None:
        selected_labels = np.unique(real_adata.obs[label_key].values) 
    elif not _check_values(selected_labels, set(real_adata.obs[label_key].values)):
        raise ValueError("Please ensure each label in `selected_labels` is valid.")

    sc.tl.rank_genes_groups(
        real_adata, 
        label_key, 
        n_genes=len(real_adata.var_names)
    )
    var_names = OrderedDict()
    for label in selected_labels:
        df = rank_genes_groups_df(
            real_adata,
            label,
            key="rank_genes_groups",
        )
        genes_list = df.names[df.names.notnull()].tolist()
        genes_list = genes_list[: n_genes]
        var_names[label] = genes_list

    _, ax_1 = plt.subplots(figsize=figsize, dpi=dpi)
    kwargs = { 
        **dotplot_kwargs,
        "var_names": var_names,
        "groupby": label_key,
        "return_fig": True,
        "ax": ax_1,
    }
    dp_1 = sc.pl.dotplot(real_adata, **kwargs)
    dp_1.make_figure() 
 
    _, ax_2 = plt.subplots(figsize=figsize, dpi=dpi)
    kwargs["ax"] = ax_2
    dp_2 = sc.pl.dotplot(generated_adata, **kwargs)
    dp_2.make_figure()

    return dp_1, dp_2