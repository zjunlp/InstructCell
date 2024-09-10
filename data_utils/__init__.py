from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    DataCollatorForSeq2Seq, 
    PreTrainedModel, 
) 
import anndata
import os   
import pickle 
import numpy as np 
from utils import unify_gene_features, read_templates
from collections import OrderedDict
from scipy.sparse import issparse, csr_matrix 
import torch 
from metadata import (
    DATASET_IDENTITY, 
    OPTION_DIR,
    OPTION_FILE_NAME, 
    COUNT_DATA_FILE_NAME, 
    CELL_LABEL, 
    RESPONSE_LABEL, 
) 
from typing import (
    Dict,  
    List, 
    Optional, 
    Iterable, 
    Literal, 
    Any, 
)

class TextCellDataset(Dataset):
    """
    A custom dataset class designed for various single-cell tasks such as cell type annotation, 
    conditional pseudo cell generation, and drug sensitivity prediction. It processes scRNA-seq 
    count data along with predefined templates and tokenizes the textual inputs and outputs for 
    model training or evaluation.

    For single cell understanding tasks, the input can be an interleaved sequence of cell data and
    text data while the output is text-only. For example, in cell type annotation tasks, the input 
    can be a cell data and a text instruction. The output is text-only, which conveys the corresponding
    cell type. 
    
    For single cell generation tasks, the input can be an interleaved sequence of cell data and text data 
    while the ouput is also an interleaved sequence of cell and texts with the cell at the end of 
    output. For example, in conditional pseudo cell generation tasks, the input is a textual 
    instruction conveying the cell type and the output is a mixture of single cell and text with
    single cell at the end of the output.

    The templates are stored in JSON files. Each JSON file contains a list of templates for a specific task.
    Each template contains two keys: "instruction" and "response". An example of JSON file which has only one 
    template is shown below:
    ```json 
        [
            {
                "instruction": "Classify the cell type of the following cell: {input}.",
                "response": "The cell type is {output}. Do you have any other questions?"
            }
        ]
    ```

    Currently Supported Tasks:
        * Cell Type Annotation (CTA)
        * Conditional Pseudo Cell Generation (CPCG)
        * Drug Sensitivity Prediction (DSP)

    Parameters
    ----------
    dir_name: str
        Path to the directory containing the datasets of a specific task. Each dataset 
        should be stored in a subdirectory with the corresponding dataset name.
    tokenizer: transformers.PreTrainedTokenizer or transformers.PreTrainedTokenizerFast
        The tokenizer used to tokenize the textual inputs and textual outputs.
    task_type: str, optional, default None
        The task type. If not provided, it is inferred from the dataset (looking up the metadata).
    template_dir_name: str, optional, default None
        The name of the directory containing task subdirectories with 'templates.json' files. If not provided,
        the current directory is used by default.
    dataset_name: str, optional, default None
        Name of the dataset within the ``dir_name`` directory. If not provided, all datasets within 
        the directory are loaded.
    split: {"train", "test", "valid"}, optional, default None
        The data split to load. Options are "train", "test", and "valid". If not provided, all splits
        are loaded. That means it will find "rna.h5ad" in each dataset directory and load the data.
        If the split is provided, it will load the corresponding split of the data. For example, if
        the split is "train", it will load "train_rna.h5ad" in each dataset directory.
    gene_vocab: an iterable object of str, optional, default None
        Vocabulary of gene features used to standardize different datasets. If not provided, the gene vocabulary 
        is derived from the intersection of gene features across all datasets. 
    modality: str, default "CELL"
        An identifier for the cell modality. The logice behind this is the same as ``mmllm.prepare_cell_text_llm``.
    num_signal_tokens: int, default 1
        Number of signal tokens used to condition the generative model in cell generation tasks. For more details, 
        please see ``mmllm.module.CellTextLLM``.
    force_gene_symbol_uppercase: bool, default True
        If True, when gene features are standardized, both the gene symbols in all datasets and gene vocabularies will be 
        converted to uppercase. If False, case sensitivity will be maintained. If ``gene_vocab`` is not provided, this is 
        ignored. 
    provide_choices: bool, optional, default True
        Whether to provide choices in the task prompts.
    no_extra_output_ratio: float, default 0.5
        Probability of generating an input-output pair whose output does not contain any extra textual sequences.
    is_encoder_decoder: bool, default True
        Whether the model is an encoder-decoder architecture. If False, the model is considered as a decoder-only model.
    random_state: int | None | np.random.RandomState | np.random.Generator, default None
        Random state or seed for reproducibility.
    """

    _SUPPORTED_CELL_TASKS = {
        "cell type annotation", 
        "conditional pseudo cell generation", 
        "drug sensitivity prediction", 
    }

    def __init__(
        self,
        dir_name: str, 
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, 
        task_type: Optional[str] = None, 
        template_dir_name: Optional[str] = None,
        dataset_name: Optional[str] = None, 
        split: Literal["train", "test", "valid"] | None = None, 
        gene_vocab: Optional[Iterable[str]] = None, 
        modality: str = "CELL", 
        num_signal_tokens: int = 1, 
        force_gene_symbol_uppercase: bool = True, 
        provide_choices: Optional[bool] = True,
        no_extra_output_ratio: float = 0.5, 
        is_encoder_decoder: bool = True,
        random_state: int | None | np.random.RandomState | np.random.Generator = None, 
    ) -> "TextCellDataset":
        ### TO DO: add max length of input
        # firstly, read the dataset and related metadata
        datasets = TextCellDataset.read_datasets(
            dir_name, 
            dataset_name=dataset_name, 
            split=split, 
            gene_vocab=gene_vocab,
            force_gene_symbol_uppercase=force_gene_symbol_uppercase, 
            provide_choices=provide_choices, 
        )
        if task_type is None:
            task_type = None
            for source in datasets:
                if task_type is None:
                    task_type = datasets[source]["metadata"]["tag"]
                else:
                    cur_task_type = datasets[source]["metadata"]["tag"]
                    assert task_type == cur_task_type, "There exist two datasets belonging to different tasks."

        # secondly, we get the prompt templates given the corresponding task type 
        self.templates = np.array(
            read_templates(template_dir_name=template_dir_name, task_type=task_type)[task_type]
        )
        if len(self.templates) == 0:
            raise ValueError(f"No templates are found for task {task_type}. Please check the template directory.")
        if provide_choices is not None:
            if not provide_choices:
                self.templates = self.templates[np.vectorize(lambda template: "{choices}" not in template.input)(self.templates)]
            else:
                self.templates = self.templates[np.vectorize(lambda template: "{choices}" in template.input)(self.templates)]

        # for each dataset, we add an attribute used to describe the source they are from 
        # and then we merge all of them
        self.count_data = [] 
        self.input_counts_indexer = [] 
        self.output_counts_indexer = [] 
        self.text_inputs = {}
        self.text_outputs = {}  
        for source in sorted(datasets.keys()):
            datasets[source]["count_data"].obs["_source"] = source
            self.count_data.append(datasets[source]["count_data"])
            outputs = TextCellDataset.preprocess(
                self.count_data[-1], 
                task_type, 
                modality=modality, 
                num_signal_tokens=num_signal_tokens,   
            )
            if outputs["input_counts_indexer"] is not None:
                self.input_counts_indexer.append(outputs["input_counts_indexer"])
            if outputs["output_counts_indexer"] is not None:
                self.output_counts_indexer.append(outputs["output_counts_indexer"])
            for key in outputs["text_inputs"]:
                if key not in self.text_inputs:
                    self.text_inputs[key] = []
                self.text_inputs[key].append(outputs["text_inputs"][key])
            for key in outputs["text_outputs"]:
                if key not in self.text_outputs:
                    self.text_outputs[key] = [] 
                self.text_outputs[key].append(outputs["text_outputs"][key]) 
        self._flatten()
        self.metadata = {
            source: datasets[source]["metadata"] for source in datasets
        } 
        self.choices = {
            source: datasets[source]["choices"] for source in datasets
        } 
        self.tokenizer = tokenizer 
        self.no_extra_output_ratio = no_extra_output_ratio 

        # process templates
        if random_state is None or isinstance(random_state, int):
            random_state = np.random.default_rng(random_state)

        if provide_choices is None or provide_choices:
            dataset_sources = self.count_data.obs["_source"].values
            self.predefined_choices = np.array(
                [
                    f"({', '.join(random_state.permutation(self.choices[dataset_source]))})" 
                    if len(self.choices[dataset_source]) > 0 else 
                    '' for dataset_source in dataset_sources
                ]
            )
        else:
            self.predefined_choices = None 

        self.template_indexer = random_state.choice(
            np.arange(len(self.templates)), 
            size=len(self)  
        )
        self.no_extra_output_instructions_indexer = random_state.random(size=len(self.template_indexer)) < self.no_extra_output_ratio

        self.modality = modality
        self.num_signal_tokens = num_signal_tokens
        self.is_encoder_decoder = is_encoder_decoder
    
    def _flatten(self) -> None:
        """Flatten input and output count indexers, adjust indices after merging datasets, 
        and concatenate count data and text information."""
        input_counts_indexer_ = []
        cum_length = [0]
        for i in range(len(self.count_data)):
            if self.input_counts_indexer:
                for j in range(len(self.input_counts_indexer[i])):
                    input_counts_indexer_.append(self.input_counts_indexer[i][j] + cum_length[-1])
            cum_length.append(cum_length[-1] + len(self.count_data[i]))
        self.input_counts_indexer = input_counts_indexer_ 
        
        if self.output_counts_indexer:
            for i in range(len(self.output_counts_indexer)):
                self.output_counts_indexer[i] = self.output_counts_indexer[i] + cum_length[i]
            self.output_counts_indexer = np.concatenate(self.output_counts_indexer, axis=0)
        else:
            self.output_counts_indexer = [] 

        # inner join
        self.count_data = anndata.concat(self.count_data)

        for text_info in [self.text_inputs, self.text_outputs]:
            for key in text_info:
                text_info[key] = np.concatenate(text_info[key], axis=0)

    @classmethod
    def preprocess(
        self, 
        adata: anndata.AnnData, 
        task_type: str,
        modality: str = "CELL", 
        num_signal_tokens: int = 1, 
    ) -> Dict[str, List[int] | List[List[int]] | Dict[str, Iterable[str]]]:
        """
        Preprocess the single-cell data to ensure it aligns with placeholders in corresponding task templates. 

        Parameters
        ----------
        adata: anndata.AnnData
            Annotated data object containing single-cell count data.
        task_type: str
            The task type to be performed. Must be one of the supported tasks: 'cell type annotation', 
            'conditional pseudo cell generation', or 'drug sensitivity prediction'.
        modality: str, optional, default "CELL"
            A string identifier representing the modality (e.g., 'CELL'). It is used to insert specific 
            modality markers in the text templates before tokenization.
        num_signal_tokens: int, optional, default 1
            Number of signal tokens used to extract conditional embeddings for cell generation.

        Returns
        -------
        outputs: dict
            A dict containing the following keys:
            - 'input_counts_indexer': List of indices for input cell counts.
            - 'output_counts_indexer': List of indices for output cell counts (used in conditional pseudo cell generation).
            - 'text_outputs': Dict of lists where keys are output types (e.g., 'output') and values are the corresponding texts.
            - 'text_inputs': Dict of lists where keys are input types (e.g., 'input', 'cell_type') and values are the corresponding texts.
        """
        assert task_type in TextCellDataset._SUPPORTED_CELL_TASKS, \
                f"Currently, we only support {TextCellDataset._SUPPORTED_CELL_TASKS}."
        
        # Notice: output_counts is a list of integers, which requires the output of task only contain one single cell 
        outputs = {
            "input_counts_indexer": None,   # instructions contain scRNA-seq modality (indexer)
            "output_counts_indexer": None,  # outputs contain scRNA-seq modality (indexer)
            "text_outputs": {},     # some output variables in the text form like the cell type in cell type annotation 
            "text_inputs": {},      # some input variables in the text form like the cell type in conditional pseudo cell generation 
        }
        modality = modality.upper()
        placeholder = f"<{modality}0>"
        start_tag = f"<{modality}>"
        end_tag = f"</{modality}>"
        signal = ''.join([f"<{modality}{i}>" for i in range(1, num_signal_tokens + 1)])
        
        if task_type in ["cell type annotation", "drug sensitivity prediction"]:
            # for these tasks, each input only contains one single cell 
            outputs["input_counts_indexer"] = [np.array([i]) for i in range(len(adata))] 
            outputs["text_inputs"]["input"] = np.array([f"{start_tag}{placeholder}{end_tag}"] * len(adata))
            if task_type == "cell type annotation":
                outputs["text_outputs"]["output"] = adata.obs[CELL_LABEL].values
            else:
                outputs["text_outputs"]["output"] = adata.obs[RESPONSE_LABEL].values
        else:
            outputs["output_counts_indexer"] = np.arange(len(adata))
            outputs["text_inputs"]["cell_type"] = adata.obs[CELL_LABEL].values
            outputs["text_outputs"]["output"] = np.array([signal] * len(adata))
        
        return outputs 
    
    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """
        Given an index, return the corresponding data item with following keys:
            - "input_counts": np.ndarray
            The input cell count. It is a dense matrix with shape (n_cells, n_genes). It represents this sample 
            has n_cells cells and each cell has n_genes genes. It may be an empty array.
            - "output_counts": np.ndarray
            The output cell count. It is a dense matrix with shape (1, n_genes). It represents the output cell data. 
            It may be an empty array. Note that the output only contains one cell.
            - "input_ids": np.ndarray
            The input text data. It is a 1D array containing the tokenized input text data.
            - "labels": np.ndarray
            The output text data. It is a 1D array containing the tokenized output text data.
        """
        item = {} 

        # get count data 
        input_counts = np.array([]) 
        if len(self.input_counts_indexer) > 0:
            input_counts_idx = self.input_counts_indexer[index]
            input_counts = self.count_data.X[input_counts_idx].toarray()
            input_counts = input_counts.astype(np.float32)
        output_counts = np.array([])
        if len(self.output_counts_indexer) > 0:
            output_counts_idx = self.output_counts_indexer[index]
            output_counts = self.count_data.X[output_counts_idx].toarray()
            output_counts = output_counts.astype(np.float32)
        item["input_counts"] = input_counts 
        item["output_counts"] = output_counts

        # get text data 
        source = self.count_data.obs["_source"].values[index]
        template = self.templates[self.template_indexer[index]]

        metadata = self.metadata[source]
        if self.predefined_choices is not None:
            choices = self.predefined_choices[index]
        else:
            choices = ''
        if len(choices) > 0:
            metadata = {
                **metadata, 
                "choices": choices
            }
        text_input_data = {key: self.text_inputs[key][index] for key in self.text_inputs}
        text_output_data = {key: self.text_outputs[key][index] for key in self.text_outputs}
        # TO DO: may cause some problems
        text_input_data = {
            **metadata, 
            **text_input_data
        }
        text_output_data = {
            **text_input_data, 
            **text_output_data
        }
        text_inputs = template.input.format(**text_input_data)
        text_outputs = template.output.format(**text_output_data)
        if self.no_extra_output_instructions_indexer[index]:
            text_outputs = "{output}".format(**text_output_data)
        # tokenize the text data to get input_ids, attention_mask, labels and so on
        # for signal tokens, the first signal token will be viewed as the ending token 
        # we will use -100 to mask the hidden states of signal tokens to avert computation of cross entropy
        # the hidden states of signal tokens will be used as conditional embeddings of downstream generative model
        text_inputs = f"User:\n{text_inputs}\n\nAssistant:\n"
        if not self.is_encoder_decoder: 
            encoding = self.tokenizer(text_inputs, add_special_tokens=False)
            input_ids = encoding.input_ids
            # do not compute cross entropy on user's instruction and system prompt
            labels = [-100] * len(input_ids)
            output_ids = self.tokenizer(text_outputs, add_special_tokens=False).input_ids
            input_ids += output_ids 
            labels += output_ids 
            if input_ids[-1] != self.tokenizer.eos_token_id:
                input_ids.append(self.tokenizer.eos_token_id)
                labels.append(self.tokenizer.eos_token_id)
        else:
            input_ids = self.tokenizer(text_inputs).input_ids
            labels = self.tokenizer(text_outputs).input_ids

        # mask the signal tokens 
        # the first sigal token is not masked so the model can learn to predict the first signal token
        added_vocab = self.tokenizer.get_added_vocab()
        signal_token_ids = set() 
        for i in range(1, self.num_signal_tokens + 1):
            signal_token = f"<{self.modality}{i}>"
            if signal_token not in added_vocab:
                raise ValueError(f"The added vocabulary of current tokenizer doesn't contain special token {signal_token}.")
            signal_token_ids.add(added_vocab[signal_token])
        loc_first_signal_token = None
        for i in range(len(labels)):
            if labels[i] in signal_token_ids:
                loc_first_signal_token = i 
                break
        if loc_first_signal_token is not None:
            labels[loc_first_signal_token + 1: ] = [-100] * (len(labels) - loc_first_signal_token - 1)
        
        # the ouput doesn't contain attention mask
        # it will be processed by collator
        item["input_ids"] = np.array(input_ids, dtype=np.int32)
        item["labels"] = np.array(labels, dtype=np.int32)

        return item 

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return max(len(self.output_counts_indexer), len(self.input_counts_indexer)) 
    
    @classmethod
    def read_datasets(
        self, 
        dir_name: str, 
        dataset_name: Optional[str] = None, 
        split: Literal["train", "test", "valid"] | None = None,
        gene_vocab: Optional[Iterable[str]] = None,
        force_gene_symbol_uppercase: bool = True, 
        provide_choices: bool = True, 
    ) -> Dict[str, anndata.AnnData | Dict[str, Any] | List[str]]:
        """
        Load single-cell datasets from specified directories. This function reads cell data in the 
        form of `.h5ad` files and unifies gene features across datasets if needed. Optionally,
        it loads predefined choices used for cell type annotation or other tasks.

        Parameters
        ----------
        dir_name: str
            Path to the directory containing the datasets of a specific task. Each dataset 
            should be stored in a subdirectory with the corresponding dataset name.
        dataset_name: str, optional, default None 
            Name of the dataset within the ``dir_name`` directory. If not provided, all datasets within 
            the directory are loaded.
        split: {"train", "test", "valid"}, optional, default None 
            The data split to load. Options are "train", "test", and "valid". If not provided, all splits
            are loaded. That means it will find "rna.h5ad" in each dataset directory and load the data.
            If the split is provided, it will load the corresponding split of the data. For example, if
            the split is "train", it will load "train_rna.h5ad" in each dataset directory.
        gene_vocab: an iterable object of str, optional, default None
            Vocabulary of gene features used to standardize different datasets. If not provided, the gene vocabulary 
            is derived from the intersection of gene features across all datasets. 
        force_gene_symbol_uppercase: bool, default True
            If True, when gene features are standardized, both the gene symbols in all datasets and gene vocabularies 
            will be converted to uppercase. If False, case sensitivity will be maintained. If ``gene_vocab`` is not provided, 
            this is ignored. 
        provide_choices: bool, optional, default True
            Whether to provide choices in the task prompts.
        
        Returns
        -------
        results: dict
            A dict containing the datasets and related metadata. The keys are the dataset names, and the values are 
            dictionaries containing the count data, metadata, and predefined choices. The predefined choices are only 
            provided if ``provide_choices`` is True.
        """
        if split is not None:
            assert split in {"train", "test", "valid"}, f"argument split must be one of 'train', 'test' and 'valid'"

        # remember the order of the datasets
        results = OrderedDict()        
        if dataset_name is not None:
            dataset_dir = os.path.join(dir_name, dataset_name)
            results[dataset_name] = {}
            results[dataset_name]["count_data"] = anndata.read_h5ad(
                os.path.join(
                    dataset_dir, 
                    COUNT_DATA_FILE_NAME if split is None else f"{split}_{COUNT_DATA_FILE_NAME}"
                )
            )
        else:
            for dataset_name in os.listdir(dir_name):
                path = os.path.join(dir_name, dataset_name)
                if os.path.isdir(path):
                    results[dataset_name] = {}
                    results[dataset_name]["count_data"] = anndata.read_h5ad(
                        os.path.join(
                            path, 
                            COUNT_DATA_FILE_NAME if split is None else f"{split}_{COUNT_DATA_FILE_NAME}"
                        )
                    )
        for dataset_name in results:
            if not issparse(results[dataset_name]["count_data"].X): 
                results[dataset_name]["count_data"].X = csr_matrix(results[dataset_name]["count_data"].X)
        
        # read related metadata 
        for source in results:
            metadata = DATASET_IDENTITY.get(source, {}) 
            results[source]["metadata"] = metadata 

        # read related choices
        for sources in results:
            results[sources]["choices"] = []
        if provide_choices is None or provide_choices: 
            with open(os.path.join(OPTION_DIR, OPTION_FILE_NAME), "rb") as f:
                choices_list = pickle.load(f)
            for source in results:
                results[source]["choices"] = choices_list.get(source, [])
        # we unify the gene features so we can merge all datasets 
        if gene_vocab is not None:
            for source in results:
                results[source]["count_data"] = unify_gene_features(
                    results[source]["count_data"], 
                    gene_vocab, 
                    force_gene_symbol_uppercase=force_gene_symbol_uppercase
                )

        return results 

class TextCellCollator(DataCollatorForSeq2Seq):
    """
    A custom data collator class for handling single-cell tasks with sequence-to-sequence (Seq2Seq) models.
    
    This collator processes both text and cell data for use in models that combine these modalities. 
    
    It inherits from ``transformers.DataCollatorForSeq2Seq`` and extends its functionality to collate input and 
    output cell counts along with the tokenized text inputs and outputs, ensuring the final batch is compatible 
    with the model requirements.

    Parameters
    ----------
    tokenizer: transformers.PreTrainedTokenizer or transformers.PreTrainedTokenizerFast
        The tokenizer used for tokenizing text data.
    model: transformers.PreTrainedModel, optional, default None
        The model that is being trained. If set and has the ``prepare_decoder_input_ids_from_labels``, 
        use it to prepare the ``decoder_input_ids``.
    **collator_kwargs: dict, optional
        Additional keyword arguments passed to ``transformers.DataCollatorForSeq2Seq`` constructor.  
    """
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, 
        model: Optional[PreTrainedModel] = None, 
        **collator_kwargs
    ) -> "TextCellCollator": 
        super().__init__(tokenizer, model=model, **collator_kwargs)
    
    def __call__(self, examples: List[Dict[str, np.ndarray | List[int]]]) -> Dict[str, torch.Tensor | None]:
        """Processes a batch of examples, collating both input/output counts and tokenized text data, 
        and returns a dictionary containing tensors for model inputs and outputs."""
        # we assume in each example, the number of placeholders is equal to the length of input_counts 
        # and the number of first signal tokens is equal to the length of output_counts
        # this assumption should be guaranteed by the dataset class
        # it will be checked in the model class
        input_counts, output_counts = [], [] 
        for example in examples:
            cur_input_counts = example.pop("input_counts")
            if len(cur_input_counts) > 0:
                input_counts.append(cur_input_counts)
            cur_output_counts = example.pop("output_counts")
            if len(cur_output_counts) > 0:
                output_counts.append(cur_output_counts)

        batch = super().__call__(examples)
        if len(input_counts) > 0:
            batch["input_counts"] = torch.from_numpy(np.concatenate(input_counts, axis=0))
        else:
            batch["input_counts"] = None 
        if len(output_counts) > 0:
            batch["output_counts"] = torch.from_numpy(np.concatenate(output_counts, axis=0))
        else:
            batch["output_counts"] = None

        return batch 
    