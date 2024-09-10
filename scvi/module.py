import torch 
from torch.distributions import (
    Distribution, 
    Normal, 
)
from torch.nn.functional import one_hot
import numpy as np 
from typing import (
    Literal, 
    Callable, 
    Dict,
    List, 
    Tuple, 
) 
from .base import (
    EmbeddingModuleMixin, 
    BaseModuleClass, 
    LossOutput, 
) 

class CVAE(EmbeddingModuleMixin, BaseModuleClass):
    """
    Conditional variational auto-encoder for scRNA-seq data. 

    Adapted from https://github.com/scverse/scvi-tools. 

    Parameters
    ----------
    n_input: int 
        Number of input features.
    n_batch: int, default 0  
        Number of batches. If 0, no batch correction is performed.
    n_labels: int, default 0
        Number of labels.
    n_hidden: int, default 128
        Number of nodes per hidden layer. Passed into ``scvi.nn.Encoder`` and ``scvi.nn.DecoderSCVI``.
    n_latent: int, default 10 
        Dimensionality of the latent space.
    n_layers: int, default 1 
        Number of hidden layers. Passed into ``scvi.nn.Encoder`` and ``scvi.nn.DecoderSCVI``.
    n_continuous_cov: int, default 0 
        Number of continuous covariates.
    n_cats_per_cov: list of int, optional, default None 
        A list of integers containing the number of categories for each categorical covariate.
    dropout_rate: float, default 0.1 
        Dropout rate. Passed into ``scvi.nn.Encoder`` but not ``scvi.nn.DecoderSCVI``.
    dispersion: {"gene", "gene-batch", "gene-label", "gene-cell"}, default "gene"
        Flexibility of the dispersion parameter when ``gene_likelihood`` is either "nb" or
        "zinb". One of the following:
        * "gene": parameter is constant per gene across cells.
        * "gene-batch": parameter is constant per gene per batch.
        * "gene-label": parameter is constant per gene per label.
        * "gene-cell": parameter is constant per gene per cell.
    log_variational: bool, default True
        If True, use ``torch.log1p`` on input data before encoding for numerical stability
        (not normalization).
    gene_likelihood: {"zinb", "nb", "poisson"}, default "zinb"
        Distribution to use for reconstruction in the generative process. One of the following:
        * "nb": ``scvi.distributions.NegativeBinomial``.
        * "zinb": ``scvi.distributions.ZeroInflatedNegativeBinomial``.
        * "poisson": ``scvi.distributions.Poisson``.
    latent_distribution: {"normal", "ln"}, default "normal"
        Distribution to use for the latent space. One of the following:
        * "normal": isotropic normal.
        * "ln": logistic normal with normal params N(0, 1).
    encode_covariates: bool, default False
        If True, covariates are concatenated to gene expression prior to passing through
        the encoder(s). Else, only gene expression is used.
    deeply_inject_covariates: bool, default True
        If True and ``n_layers > 1``, covariates are concatenated to the outputs of hidden
        layers in the encoder(s) (if ``encoder_covariates`` is True) and the decoder prior to
        passing through the next layer.
    batch_representation: {"one-hot", "embedding"}, default "one-hot"
        ``EXPERIMENTAL`` Method for encoding batch information. One of the following:
        * "one-hot": represent batches with one-hot encodings.
        * "embedding": represent batches with continuously-valued embeddings using ``scvi.nn.Embedding``.
        Note that batch representations are only passed into the encoder(s) if
        ``encode_covariates`` is True.
    use_batch_norm: {"encoder", "decoder", "none", "both"}, default "both"
        Specifies where to use ``torch.nn.BatchNorm1d`` in the model. One of the following:
        * "none": don't use batch norm in either encoder(s) or decoder.
        * "encoder": use batch norm only in the encoder(s).
        * "decoder": use batch norm only in the decoder.
        * "both": use batch norm in both encoder(s) and decoder.
        Note: if ``use_layer_norm`` is also specified, both will be applied (first
        ``torch.nn.BatchNorm1d``, then ``torch.nn.LayerNorm``).
    use_layer_norm: {"encoder", "decoder", "none", "both"}, default "none"
        Specifies where to use ``torch.nn.LayerNorm`` in the model. One of the following:
        * "none": don't use layer norm in either encoder(s) or decoder.
        * "encoder": use layer norm only in the encoder(s).
        * "decoder": use layer norm only in the decoder.
        * "both": use layer norm in both encoder(s) and decoder.
        Note: if ``use_batch_norm`` is also specified, both will be applied (first
        ``torch.nn.BatchNorm1d``, then ``torch.nn.LayerNorm``).
    use_size_factor_key: bool, default False
        If True, use the ``anndata.AnnData.obs`` column as defined by the
        ``size_factor_key`` parameter in the model's ``setup_anndata`` method as the scaling
        factor in the mean of the conditional distribution. Takes priority over
        ``use_observed_lib_size``.
    use_observed_lib_size: bool, default True
        If True, use the observed library size for RNA as the scaling factor in the mean of the
        conditional distribution.
    library_log_means: np.ndarray, optional, default None
        ``numpy.ndarray`` of shape (1, n_batch) of means of the log library sizes that
        parameterize the prior on library size if ``use_size_factor_key`` is False and
        ``use_observed_lib_size`` is False.
    library_log_vars: np.ndarray, optional, default None
        ``numpy.ndarray`` of shape (1, n_batch) of variances of the log library sizes
        that parameterize the prior on library size if ``use_size_factor_key`` is False and
        ``use_observed_lib_size`` is False.
    var_activation: Callable, default None
        Callable used to ensure positivity of the variance of the variational distribution. Passed
        into ``scvi.nn.Encoder``. Defaults to ``torch.exp``.
    extra_encoder_kwargs: dict, optional, default None
        Additional keyword arguments passed into ``scvi.nn.Encoder``.
    extra_decoder_kwargs: dict, optional, default None
        Additional keyword arguments passed into ``scvi.nn.DecoderSCVI``.
    batch_embedding_kwargs: dict, optional, default None
        Keyword arguments passed into ``scvi.nn.Embedding`` if ``batch_representation`` is
        set to "embedding".
    adaptive_library: bool, default True
        If True, the model will adaptively decode the library size based on the conditional information 
        instead of using batch information only. Note that this is more flexible because it can take 
        both batch information and other conditional information into account. This is typically used
        when conditional information is provided by an external model.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: List[int] | None = None,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: np.ndarray | None = None,
        library_log_vars: np.ndarray | None = None,
        var_activation: Callable[[torch.Tensor], torch.Tensor] = None,
        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,
        batch_embedding_kwargs: dict | None = None,
        adaptive_library: bool = True, 
    ) -> "CVAE":
        from scvi.nn import DecoderSCVI, Encoder

        super().__init__()

        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "`dispersion` must be one of 'gene', 'gene-batch', 'gene-label', 'gene-cell'."
            )

        self.batch_representation = batch_representation
        if self.batch_representation == "embedding":
            self.init_embedding('batch_ids', n_batch, **(batch_embedding_kwargs or {}))
            batch_dim = self.get_embedding('batch_ids').embedding_dim
        elif self.batch_representation != "one-hot":
            raise ValueError("`batch_representation` must be one of 'one-hot', 'embedding'.")

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        if self.batch_representation == "embedding":
            n_input_encoder += batch_dim * encode_covariates
            cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        else:
            cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)

        encoder_cat_list = cat_list if encode_covariates else None
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        n_input_decoder = n_latent + n_continuous_cov
        if self.batch_representation == "embedding":
            n_input_decoder += batch_dim

        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        self.decoder = DecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
            **_extra_decoder_kwargs,
        )

        # make the library size adaptively
        self.adaptive_library = adaptive_library
        self.library_decoder = lambda x: x 
        if encode_covariates and adaptive_library:
            assert n_continuous_cov > 0, "The number of continuous covariates should be greater than 0" + \
                "so the model can adaptively decode the library size"
            n_input_library_decoder = 1 + n_continuous_cov
            self.library_decoder = torch.nn.Sequential(
                torch.nn.Linear(n_input_library_decoder, n_input_library_decoder // 2), 
                torch.nn.GELU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(n_input_library_decoder // 2, 1)
            )
        else:
            self.adaptive_library = False

    def _get_inference_input(
        self,
        tensors: Dict[str, torch.Tensor | None],
    ) -> Dict[str, torch.Tensor | None]:
        """Get input tensors for the inference process."""

        return {
            'gene_counts': tensors['gene_counts'],
            'batch_ids': tensors['batch_ids'],
            'cont_covs': tensors.get('cont_covs', None),
            'cat_covs': tensors.get('cat_covs', None),
        }

    def _get_generative_input(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[str, torch.Tensor | Distribution | None],
    ) -> Dict[str, torch.Tensor | None]:
        """Get input tensors for the generative process."""
        size_factor = tensors.get('size_factor', None)
        if size_factor is not None:
            size_factor = torch.log(size_factor)

        return {
            'z': inference_outputs['z'],
            'library': inference_outputs['library'],
            'batch_ids': tensors['batch_ids'],
            'y': tensors['y'],
            'cont_covs': tensors.get('cont_covs', None),
            'cat_covs': tensors.get('cat_covs', None),
            'size_factor': size_factor,
        }

    def _compute_local_library_params(
        self,
        batch_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes local library parameters.

        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        from torch.nn.functional import linear

        n_batch = self.library_log_means.shape[1]
        local_library_log_means = linear(
            one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_means
        )

        local_library_log_vars = linear(
            one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_vars
        )

        return local_library_log_means, local_library_log_vars

    def inference(
        self,
        gene_counts: torch.Tensor,
        batch_ids: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> Dict[str, torch.Tensor | Distribution | None]:
        """Run the regular inference process."""
        gene_counts_ = gene_counts
        if self.use_observed_lib_size:
            library = torch.log(gene_counts.sum(1)).unsqueeze(1)
        if self.log_variational:
            gene_counts_ = torch.log1p(gene_counts_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((gene_counts_, cont_covs), dim=-1)
        else:
            encoder_input = gene_counts_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
        
        if self.batch_representation == "embedding" and self.encode_covariates:
            batch_rep = self.compute_embedding('batch_ids', batch_ids)
            encoder_input = torch.cat([encoder_input, batch_rep], dim=-1)
            qz, z = self.z_encoder(encoder_input, *categorical_input)
        else:
            qz, z = self.z_encoder(encoder_input, batch_ids, *categorical_input)

        ql = None
        if not self.use_observed_lib_size:
            if self.batch_representation == "embedding":
                ql, library_encoded = self.l_encoder(encoder_input, *categorical_input)
            else:
                ql, library_encoded = self.l_encoder(
                    encoder_input, batch_ids, *categorical_input
                )
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.sample((n_samples,))

        return {
            'z': z,                                 # the latent variable z sampled from the posterior distribution
            'q_z': qz,                              # the posterior distribution of the latent variable z
            'q_library': ql,                        # the posterior distribution of the library size
            'library': library,                     # the library size sampled from the posterior distribution
        }

    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_ids: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        transform_batch: torch.Tensor | None = None,
    ) -> Dict[str, Distribution | None]:
        """Run the generative process."""
        from torch.nn.functional import linear

        from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial

        # TODO: refactor forward function to not rely on y
        # Likelihood distribution
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            # for generative process, cont_covs is concatenated to z
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if transform_batch is not None:
            batch_ids = torch.ones_like(batch_ids) * transform_batch

        if not self.use_size_factor_key:
            # MODIFIED: adjust the size_factor based on the conditional information adaptively
            size_factor = library
            if self.adaptive_library and cont_covs is not None:
                size_factor = self.library_decoder(torch.cat([size_factor, cont_covs], dim=-1))

        if self.batch_representation == "embedding":
            batch_rep = self.compute_embedding('batch_ids', batch_ids)
            decoder_input = torch.cat([decoder_input, batch_rep], dim=-1)
            px_scale, px_r, px_rate, px_dropout = self.decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                *categorical_input,
                y, 
            )
        else:
            px_scale, px_r, px_rate, px_dropout = self.decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                batch_ids,
                *categorical_input,
                y, 
            )

        if self.dispersion == "gene-label":
            px_r = linear(
                one_hot(y.squeeze(-1), self.n_labels).float(), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = linear(one_hot(batch_ids.squeeze(-1), self.n_batch).float(), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)

        # Priors
        pl = self.get_prior_library_distribution(batch_ids)
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return {
            'p_gene': px,       # the distribution of the gene expression for each cell 
            'p_library': pl,    # the distribution of the library size for each cell
            'p_z': pz,          # the prior distribution of the latent variable z
        }

    def get_prior_library_distribution(self, batch_ids: torch.Tensor) -> Distribution:
        """Get the prior distribution of the library size."""
        if self.use_observed_lib_size:
            return None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_ids)

            return Normal(local_library_log_means, local_library_log_vars.sqrt())

    def loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[str, torch.Tensor | Distribution | None],
        generative_outputs: Dict[str, Distribution | None],
        kl_weight: float = 3.0,
    ) -> LossOutput:
        """Compute the loss."""
        from torch.distributions import kl_divergence

        x = tensors['gene_counts']
        kl_divergence_z = kl_divergence(
            inference_outputs['q_z'], generative_outputs['p_z']
        ).sum(dim=-1)
        if not self.use_observed_lib_size:
            kl_divergence_l = kl_divergence(
                inference_outputs['q_library'], generative_outputs['p_library']
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.tensor(0.0, device=x.device)

        reconst_loss = -generative_outputs['p_gene'].log_prob(x).sum(-1)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local={
                'kl_library': kl_divergence_l,
                'kl_z': kl_divergence_z,
            },
        )
