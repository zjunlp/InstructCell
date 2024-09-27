from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PretrainedConfig, 
) 
from typing import (
    Dict, 
    Tuple, 
    List, 
    Optional, 
)
import torch.nn as nn 
import warnings 

def _check_modality_name(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, 
    special_tokens_dict: Dict[str, str | List[str]]
) -> None:
    """Check if the special tokens are already in the vocabulary and if they are duplicated."""    
    vocab = tokenizer.get_vocab()
    visited = set()
    for tokens in special_tokens_dict.values():
        if isinstance(tokens, str):
            tokens = [tokens]
        for token in tokens:
            if token in vocab:
                raise ValueError(f"Token {token} is already in the vocabulary.")
            if token in visited:
                raise ValueError(f"Token {token} is duplicated.")
            visited.add(token)

def _validate_added_special_tokens(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, 
    added_tokens: List[str]
) -> None:
    """Check if the added tokens are correctly added to the tokenizer."""
    added_tokens = [token for token in set(added_tokens)]
    new_integers = [] 
    for token in added_tokens:
        int_list = tokenizer.encode(token, add_special_tokens=False)
        # we assume that the special token is not split into multiple tokens
        assert len(int_list) == 1, f"Token '{token}' is encoded into multiple integers: {int_list}."
        new_integers.append(int_list[0])
        if tokenizer.decode(int_list) != token:
            warnings.warn(
                f"For '{token}', the reconstructed token is "
                f"'{tokenizer.decode(int_list)}'.", 
                UserWarning 
            )

    added_tokens = [' '.join(added_tokens), "".join(added_tokens)]
    for token in added_tokens:
        int_list = tokenizer.encode(token, add_special_tokens=False) 
        # ensure the combination of the added tokens is split as expected
        assert int_list == new_integers, \
            f"It seems that the result ({int_list}) of encoding '{token}' is not correct. {new_integers} is expected."

def prepare_cell_text_llm(
    model: PreTrainedModel | nn.Module, 
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, 
    modality_tag: str = "CELL", 
    num_signal_tokens: int = 1, 
    ignore_index: Optional[int] = None, 
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None, 
) -> Tuple[PreTrainedModel | nn.Module, PreTrainedTokenizer | PreTrainedTokenizerFast]:
    """
    Prepare a model and tokenizer for handling cell-language tasks.

    This function helps in adapting a model and its tokenizer for a new modality, by adding
    special tokens that represent start, end, and signal tokens for that modality, which 
    can be useful in tasks such as processing cell-specific data in large language models (LLMs).

    Parameters
    ----------
    model: transformers.PreTrainedModel or torch.nn.Module
        The model to be adapted. This can be any pre-trained model from the transformers library 
        or a custom model that supports resizing its token embeddings.
    tokenizer: transformers.PreTrainedTokenizer or transformers.PreTrainedTokenizerFast
        The tokenizer to be adapted. This will have special tokens added for the new modality.
    modality_tag: str, default "CELL"
        The tag that represents the modality to be added to the tokenizer. It will be used to create
        start and end tags as well as signal tokens.
    num_signal_tokens: int, default 1
        The number of signal tokens to add. These tokens serve as signals enabling the model to generate
        data in the specified modality.
    ignore_index: int, optional, default None 
        Index to be ignored during loss computation in models like sequence classification or generation.
        If not provided, the model's ``ignore_index`` should be set.
    pad_token_id: int, optional, default None 
        Token ID to be used for padding. If not provided, the model's ``pad_token_id`` should be set.
    eos_token_id: int, optional, default None 
        Token ID to represent the end of the sequence. If not provided, the model's ``eos_token_id`` 
        should be set.
    pad_to_multiple_of: int, optional, default None 
        If set will pad the embedding matrix to a multiple of the provided value after resizing embedding 
        layer. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute 
        capability ``>= 7.5`` (Volta), or on TPUs which benefit from having sequence lengths be a multiple 
        of 128.

    Returns
    -------
    model: transformers.PreTrainedModel or torch.nn.Module
        The modified model with resized token embeddings to accommodate the new special tokens.
    tokenizer: transformers.PreTrainedTokenizer or transformers.PreTrainedTokenizerFast
        The modified tokenizer with the added special tokens for the new modality.

    Notes
    -----
    - This function adds special tokens like ``<MODALITY>``, ``</MODALITY>``, and signal tokens 
      ``<MODALITY1>``, ``<MODALITY2>``, etc., where ``MODALITY`` is derived from the ``modality_tag``.
      The placeholder token ``<MODALITY0>`` is also added to the tokenizer, which is used to identify the 
      position of the modality data within the text data.

    Examples
    --------
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    >>> MODEL_PATH = "t5-base" 
    >>> tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    >>> model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    >>> model, tokenizer = prepare_cell_text_llm(
    ...     model, 
    ...     tokenizer, 
    ...     modality_tag="CELL",
    ...     ignore_index=-100, 
    ...     pad_to_multiple_of=8, 
    ... )
    Follwing 4 tokens are added for modality CELL:
    ['<CELL>', '</CELL>', '<CELL0>', '<CELL1>']
    >>> tokenizer.get_added_vocab() 
    {'</CELL>': 32101, '<CELL0>': 32102, '<CELL1>': 32103, '<CELL>': 32100}
    >>> model.config.ignore_index
    -100
    """
    if not hasattr(model, "config"):
        model.config = PretrainedConfig()
    if not hasattr(model, "resize_token_embeddings"):
        raise ValueError("The model should support resizing its token embeddings.")
    special_tokens_dict = {} 

    # e.g. if there is a modality called "CELL"
    # start tag <CELL>, end tag </CELL>, placeholder <CELL0> 
    # and some signal tokens like <CELL1>, <CELL2>, <CELL3> etc are added to the tokenizer
    # note that we make sure that the order of adding the special tokens is fixed 
    special_tokens_dict["start_tag"] = f"<{modality_tag}>"
    special_tokens_dict["end_tag"] = f"</{modality_tag}>"
    special_tokens_dict["placeholder"] = f"<{modality_tag}0>"
    special_tokens_dict["signal_tokens"] = [
        f"<{modality_tag}{i}>" for i in range(1, num_signal_tokens + 1)
    ]
    _check_modality_name(tokenizer, special_tokens_dict)

    # we view these tokens as special tokens so they are not 
    # affected by the tokenizer's normalization or splitting
    added_tokens = [
        special_tokens_dict["start_tag"],
        special_tokens_dict["end_tag"],
        special_tokens_dict["placeholder"]
    ] + special_tokens_dict["signal_tokens"]
    # we don't want to replace the existing special tokens
    tokenizer.add_special_tokens(
        {"additional_special_tokens": added_tokens}, 
        replace_additional_special_tokens=False, 
    )
    _validate_added_special_tokens(tokenizer, added_tokens)
    print(f"Following {len(added_tokens)} tokens are added for modality {modality_tag}:\n{added_tokens}")
    
    # TO DO: placeholder tokens should be not added to the model's embedding layer 
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=pad_to_multiple_of)
    model.config.special_tokens_dict = special_tokens_dict
    added_vocab = tokenizer.get_added_vocab()
    special_tokens_index_dict = {}
    for token_type in special_tokens_dict.keys():
        tokens = special_tokens_dict[token_type]
        if isinstance(tokens, str):
            special_tokens_index_dict[token_type] = added_vocab[tokens]
        else:
            special_tokens_index_dict["first_signal_token"] = added_vocab[tokens[0]]
    model.config.special_tokens_index_dict = special_tokens_index_dict
    
    if pad_token_id is not None:
        model.config.pad_token_id = pad_token_id
    else:
        assert hasattr(model.config, "pad_token_id"), "Please provide pad_token_id argument or set model.config.pad_token_id"
    if eos_token_id is not None:
        model.config.eos_token_id = eos_token_id
    else:
        assert hasattr(model.config, "eos_token_id"), "Please provide eos_token_id argument or set model.config.eos_token_id"
    if ignore_index is not None:
        model.config.ignore_index = ignore_index
    else:
        assert hasattr(model.config, "ignore_index"), "Please provide ignore_index argument or set model.config.ignore_index"
    model.config.num_signal_tokens = num_signal_tokens

    return model, tokenizer 
