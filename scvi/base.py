import torch
from torch.nn import ModuleDict
from .nn import Embedding
import torch.nn as nn 
from typing import (
    Dict, 
    Tuple, 
    Union, 
)
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import field, dataclass

LossRecord = Union[Dict[str, torch.Tensor], torch.Tensor]

@dataclass
class LossOutput:
    """Loss signature for models.

    This class provides an organized way to record the model loss, as well as
    the components of the ELBO. This may also be used in MLE, MAP, EM methods.
    The loss is used for backpropagation during inference. The other parameters
    are used for logging/early stopping during inference.

    Parameters
    ----------
    loss
        Tensor with loss for minibatch. Should be one dimensional with one value.
        Note that loss should be in an array/tensor and not a float.
    reconstruction_loss
        Reconstruction loss for each observation in the minibatch. If a tensor, converted to
        a dictionary with key "reconstruction_loss" and value as tensor.
    kl_local
        KL divergence associated with each observation in the minibatch. If a tensor, converted to
        a dictionary with key "kl_local" and value as tensor.
    kl_global
        Global KL divergence term. Should be one dimensional with one value. If a tensor, converted
        to a dictionary with key "kl_global" and value as tensor.
    classification_loss
        Classification loss.
    logits
        Logits for classification.
    true_labels
        True labels for classification.
    extra_metrics
        Additional metrics can be passed as arrays/tensors or dictionaries of
        arrays/tensors.
    n_obs_minibatch
        Number of observations in the minibatch. If None, will be inferred from
        the shape of the reconstruction_loss tensor.


    Examples
    --------
    >>> loss_output = LossOutput(
    ...     loss=loss,
    ...     reconstruction_loss=reconstruction_loss,
    ...     kl_local=kl_local,
    ...     extra_metrics={"x": scalar_tensor_x, "y": scalar_tensor_y},
    ... )
    """

    loss: LossRecord
    reconstruction_loss: LossRecord | None = None
    kl_local: LossRecord | None = None
    kl_global: LossRecord | None = None
    classification_loss: LossRecord | None = None
    logits: torch.Tensor | None = None
    true_labels: torch.Tensor | None = None
    extra_metrics: Dict[str, torch.Tensor] | None = field(default_factory=dict)
    n_obs_minibatch: int | None = None
    reconstruction_loss_sum: torch.Tensor = field(default=None)
    kl_local_sum: torch.Tensor = field(default=None)
    kl_global_sum: torch.Tensor = field(default=None)

    def __post_init__(self):
        object.__setattr__(self, "loss", self.dict_sum(self.loss))

        if self.n_obs_minibatch is None and self.reconstruction_loss is None:
            raise ValueError("Must provide either n_obs_minibatch or reconstruction_loss")

        default = 0 * self.loss
        if self.reconstruction_loss is None:
            object.__setattr__(self, "reconstruction_loss", default)
        if self.kl_local is None:
            object.__setattr__(self, "kl_local", default)
        if self.kl_global is None:
            object.__setattr__(self, "kl_global", default)

        object.__setattr__(self, "reconstruction_loss", self._as_dict("reconstruction_loss"))
        object.__setattr__(self, "kl_local", self._as_dict("kl_local"))
        object.__setattr__(self, "kl_global", self._as_dict("kl_global"))
        object.__setattr__(
            self,
            "reconstruction_loss_sum",
            self.dict_sum(self.reconstruction_loss).sum(),
        )
        object.__setattr__(self, "kl_local_sum", self.dict_sum(self.kl_local).sum())
        object.__setattr__(self, "kl_global_sum", self.dict_sum(self.kl_global))

        if self.reconstruction_loss is not None and self.n_obs_minibatch is None:
            rec_loss = self.reconstruction_loss
            object.__setattr__(self, "n_obs_minibatch", list(rec_loss.values())[0].shape[0])

        if self.classification_loss is not None and (
            self.logits is None or self.true_labels is None
        ):
            raise ValueError(
                "Must provide `logits` and `true_labels` if `classification_loss` is " "provided."
            )

    @staticmethod
    def dict_sum(dictionary: Dict[str, torch.Tensor] | torch.Tensor):
        """Sum over elements of a dictionary."""
        if isinstance(dictionary, dict):
            return sum(dictionary.values())
        else:
            return dictionary

    @property
    def extra_metrics_keys(self) -> Iterable[str]:
        """Keys for extra metrics."""
        return self.extra_metrics.keys()

    def _as_dict(self, attr_name: str):
        attr = getattr(self, attr_name)
        if isinstance(attr, dict):
            return attr
        else:
            return {attr_name: attr}

def _get_dict_if_none(param):
    param = {} if not isinstance(param, dict) else param

    return param

def _generic_forward(
    module,
    tensors,
    inference_kwargs,
    generative_kwargs,
    loss_kwargs,
    get_inference_input_kwargs,
    get_generative_input_kwargs,
    compute_loss,
):
    """Core of the forward call shared by PyTorch- and Jax-based modules."""
    inference_kwargs = _get_dict_if_none(inference_kwargs)
    generative_kwargs = _get_dict_if_none(generative_kwargs)
    loss_kwargs = _get_dict_if_none(loss_kwargs)
    get_inference_input_kwargs = _get_dict_if_none(get_inference_input_kwargs)
    get_generative_input_kwargs = _get_dict_if_none(get_generative_input_kwargs)

    inference_inputs = module._get_inference_input(tensors, **get_inference_input_kwargs)
    inference_outputs = module.inference(**inference_inputs, **inference_kwargs)
    generative_inputs = module._get_generative_input(
        tensors, inference_outputs, **get_generative_input_kwargs
    )
    generative_outputs = module.generative(**generative_inputs, **generative_kwargs)
    if compute_loss:
        losses = module.loss(tensors, inference_outputs, generative_outputs, **loss_kwargs)
        return inference_outputs, generative_outputs, losses
    else:
        return inference_outputs, generative_outputs


class EmbeddingModuleMixin:
    """``EXPERIMENTAL`` Mixin class for initializing and using embeddings in a module.

    Notes
    -----
    Lifecycle: experimental in v1.2.
    """

    @property
    def embeddings_dict(self) -> ModuleDict:
        """Dictionary of embeddings."""
        if not hasattr(self, "_embeddings_dict"):
            self._embeddings_dict = ModuleDict()
        return self._embeddings_dict

    def add_embedding(self, key: str, embedding: Embedding, overwrite: bool = False) -> None:
        """Add an embedding to the module."""
        if key in self.embeddings_dict and not overwrite:
            raise KeyError(f"Embedding {key} already exists.")
        self.embeddings_dict[key] = embedding

    def remove_embedding(self, key: str) -> None:
        """Remove an embedding from the module."""
        if key not in self.embeddings_dict:
            raise KeyError(f"Embedding {key} not found.")
        del self.embeddings_dict[key]

    def get_embedding(self, key: str) -> Embedding:
        """Get an embedding from the module."""
        if key not in self.embeddings_dict:
            raise KeyError(f"Embedding {key} not found.")
        return self.embeddings_dict[key]

    def init_embedding(
        self,
        key: str,
        num_embeddings: int,
        embedding_dim: int = 5,
        **kwargs,
    ) -> None:
        """Initialize an embedding in the module."""
        self.add_embedding(key, Embedding(num_embeddings, embedding_dim, **kwargs))

    def compute_embedding(self, key: str, indices: torch.Tensor) -> torch.Tensor:
        """Forward pass for an embedding."""
        indices = indices.flatten() if indices.ndim > 1 else indices
        return self.get_embedding(key)(indices)


class BaseModuleClass(nn.Module):
    """Abstract class for scvi-tools modules.

    Notes
    -----
    See further usage examples in the following tutorials:

    1. :doc:`/tutorials/notebooks/dev/module_user_guide`
    """

    def __init__(
        self,
    ):
        super().__init__()

    @property
    def device(self):
        device = list({p.device for p in self.parameters()})
        if len(device) > 1:
            raise RuntimeError("Module tensors on multiple devices.")
        return device[0]

    def on_load(self, model):
        """Callback function run in :meth:`~scvi.model.base.BaseModelClass.load`."""

    def forward(
        self,
        tensors,
        get_inference_input_kwargs: dict | None = None,
        get_generative_input_kwargs: dict | None = None,
        inference_kwargs: dict | None = None,
        generative_kwargs: dict | None = None,
        loss_kwargs: dict | None = None,
        compute_loss=True,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, LossOutput]:
        """Forward pass through the network.

        Parameters
        ----------
        tensors
            tensors to pass through
        get_inference_input_kwargs
            Keyword args for ``_get_inference_input()``
        get_generative_input_kwargs
            Keyword args for ``_get_generative_input()``
        inference_kwargs
            Keyword args for ``inference()``
        generative_kwargs
            Keyword args for ``generative()``
        loss_kwargs
            Keyword args for ``loss()``
        compute_loss
            Whether to compute loss on forward pass. This adds
            another return value.
        """
        return _generic_forward(
            self,
            tensors,
            inference_kwargs,
            generative_kwargs,
            loss_kwargs,
            get_inference_input_kwargs,
            get_generative_input_kwargs,
            compute_loss,
        )

    @abstractmethod
    def _get_inference_input(self, tensors: Dict[str, torch.Tensor], **kwargs):
        """Parse tensors dictionary for inference related values."""

    @abstractmethod
    def _get_generative_input(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[str, torch.Tensor],
        **kwargs,
    ):
        """Parse tensors dictionary for generative related values."""

    @abstractmethod
    def inference(
        self,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor | torch.distributions.Distribution]:
        """Run the recognition model.

        In the case of variational inference, this function will perform steps related to
        computing variational distribution parameters. In a VAE, this will involve running
        data through encoder networks.

        This function should return a dictionary with str keys and :class:`~torch.Tensor` values.
        """

    @abstractmethod
    def generative(
        self, *args, **kwargs
    ) -> Dict[str, torch.Tensor | torch.distributions.Distribution]:
        """Run the generative model.

        This function should return the parameters associated with the likelihood of the data.
        This is typically written as :math:`p(x|z)`.

        This function should return a dictionary with str keys and :class:`~torch.Tensor` values.
        """

    @abstractmethod
    def loss(self, *args, **kwargs) -> LossOutput:
        """Compute the loss for a minibatch of data.

        This function uses the outputs of the inference and generative functions to compute
        a loss. This many optionally include other penalty terms, which should be computed here.

        This function should return an object of type :class:`~scvi.module.base.LossOutput`.
        """

    @abstractmethod
    def sample(self, *args, **kwargs):
        """Generate samples from the learned model."""