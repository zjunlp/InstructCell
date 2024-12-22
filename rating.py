import numpy as np
import torch  
import torch.nn as nn 
from metadata import (
    CPCG, 
    TASKS, 
    SEED, 
    GENE_VOCAB_DIR, 
    MODEL_PARAMETERS, 
)
from utils import parse_parameters, str2bool 
from torch.utils.data import ( 
    SequentialSampler, 
    DataLoader, 
) 
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
    Blip2QFormerConfig, 
) 
from data_utils import TextCellDataset, TextCellCollator
from mmllm import prepare_cell_text_llm
from mmllm.module import (
    Generator, 
    CellTextLLM, 
    SCQFormer, 
)
from scvi.utils import init_library_size
from metrics import compute_ratio_distinct_ngram  
import os
import re 
from copy import deepcopy 
from collections import Counter  
import requests
import json
import warnings 
import argparse
from typing import (
    List, 
    Dict, 
    Callable, 
    Any, 
    Optional, 
    Tuple, 
)  

PATTERN = r"Rating: \[\[(1|2|3|4|5)\]\]"

def get_rating_template(is_generation_task: bool) -> str:
    """Return a rating template based on the task type."""
    rating_template = (
        "[Instruction]\nPlease act as an impartial judge and evaluate the quality " 
        "of the response provided by an AI assistant to the user question displayed below. " 
        "Your evaluation should only consider fluency, grammar and whether " 
    ) 
    if is_generation_task:
        rating_template += "the response gives a generated cell. "
    else:
        rating_template += "the response gives a prediction. "
    rating_template += (
        "Begin your evaluation by providing a short explanation. " 
        "Be as objective as possible. After providing your explanation, you must " 
        "rate the response on a scale of 1 to 5 by strictly following this format: " 
        "\"[[rating]]\", for example: \"Rating: [[3]]\".\nNote that in our setting, " 
        "the response without any explanation is good as long as it is fluent and has no " 
        "grammatical error. "
    )
    if is_generation_task:
        rating_template += (
            "Since it is not possible to describe the generated cell in words, "
            "we simply use \"<output>\" to represent the generated cell. "
            "Don't care about the specific characteristics of this cell."
        )
    else:
        rating_template += "Don't care about the accuracy of prediction." 
    rating_template += (
        "\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n" 
        "[The End of Assistant's Answer]" 
    )
    return rating_template 

def postprocess_response(content: str) -> str: 
    """Extract the rating from the response."""
    match = re.search(PATTERN, content)
    if match:
        return match.group(1)
    else:
        return '' 

def postprocess_question(question: str) -> str:
    """Postprocess the question."""
    # "User: " is the prefix of the question in our case 
    # "Assistant: " is the suffix of the question in our case 
    # they are removed in the postprocessing step
    return question[6: -11]

class SimpleClaudeClient:
    """A simple client for Claude API."""
    def __init__(self, api_key: str, base_url: str) -> "SimpleClaudeClient":
        self.api_key = f"Bearer {api_key}"
        self.base_url = base_url
    
    def get_claude_model_response(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "claude-3-5-sonnet-20240620",
        post_processor: Optional[Callable[[str], Any]] = None,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, str]: 
        """Get the response from the Claude API."""
        query = { 
            **kwargs, 
            "model": model, 
            "messages": messages,
        } 
        headers = { 
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }
        response = requests.post(
            self.base_url, 
            headers=headers, 
            json=query,
        )
        try:
            response = response.json()
            content = response["choices"][0]["message"]["content"]  
        except:
            # JSONDecodeError may occur here  
            content = '' 
        processed_content = post_processor(content) if post_processor is not None else content 

        return {
            "content": content,
            "processed_content": processed_content
        }
    
def get_ratings(
    questions: List[str],
    candidates: List[Dict[str, str]],
    task_types: List[str],
    api_key: str, 
    base_url: str, 
    model: str = "claude-3-5-sonnet-20240620",
    num_successful_tries: int = 1000, 
    **kwargs: Dict[str, Any],
) -> Tuple[List[int], Dict[str, Dict[str, List[str]]]]:
    """Get ratings for all candidate answers to the questions."""
    num_questions = len(questions) 
    if num_questions != len(candidates):
        raise ValueError("The number of questions and candidates should be the same.")
    if num_questions == 0:
        raise ValueError("The number of questions should be greater than 0.")
    if num_questions < num_successful_tries:
        num_successful_tries = num_questions
        warnings.warn(
            f"The number of successful tries is set to {num_successful_tries} "
            "since the number of questions is less than the number of successful tries you specified.",
            UserWarning
        )
    
    client = SimpleClaudeClient(api_key, base_url)  
    outputs = {candidate: {"response": [], "rating": []} for candidate in candidates[0].keys()} 
    indices = [] 
    num_candidates = len(outputs)
    p = counter = 0 
    while p < num_questions and counter < num_successful_tries:
        is_generation_task = task_types[p] == CPCG
        rating_template = get_rating_template(is_generation_task)
        question = questions[p]
        candidate_answers = candidates[p]
        current_outputs = {} 
        for candidate, answer in candidate_answers.items():
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": rating_template.format(question=question, answer=answer)
                }
            ]
            response_dict = client.get_claude_model_response(
                messages=messages, 
                model=model, 
                post_processor=postprocess_response, 
                **kwargs,
            )
            if response_dict["processed_content"] == '':
                # unsuccessful try
                break  
            current_outputs[candidate] = {} 
            current_outputs[candidate]["rating"] = response_dict["processed_content"]
            current_outputs[candidate]["response"] = response_dict["content"]
        if len(current_outputs) == num_candidates:
            # successful try
            indices.append(p)
            for candidate in current_outputs:
                outputs[candidate]["rating"].append(current_outputs[candidate]["rating"])
                outputs[candidate]["response"].append(current_outputs[candidate]["response"])
            counter += 1
        p += 1 
        print(f"--Progress: {p} --Successful Tries: {counter}")
    
    return indices, outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_key", 
        required=True, 
        type=str,
        help="the API key for the Claude API"
    )
    parser.add_argument(
        "--base_url",
        required=True,
        type=str,
        help="the base URL of the Claude API"
    )
    parser.add_argument(
        "--trained_model_paths", 
        required=True,
        type=str,
        nargs="+",
        help="the list of paths storing parameters of trained models"
    )
    parser.add_argument(
        "--drop_ground_truth",
        default=False,
        type=str2bool,
        help="whether to evaluate the ground truth or not"
    )
    parser.add_argument(
        "--evaluator", 
        default="claude-3-5-sonnet-20240620", 
        type=str, 
        help="the model used for evaluation"
    )
    parser.add_argument(
        "--num_successful_tries", 
        default=1000, 
        type=int, 
        help="the number of samples to be evaluated"
    )
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
    parser.add_argument("--template_dir_name", default=None, type=str, help="the directory of evaluation templates")
    parser.add_argument("--batch_size", default=128, type=int, help="the batch size of the dataloader")
    args = parser.parse_args() 

    drop_ground_truth = args.drop_ground_truth
    modality_tag = args.modality_tag
    num_signal_tokens = args.num_signal_tokens
    force_gene_symbol_uppercase = False   
    random_state = np.random.default_rng(SEED) 
    gene_vocab_file_name = args.gene_vocab_file_name
    template_dir_name = args.template_dir_name
    batch_size = args.batch_size 
    device_id = args.device_id 
    trained_model_paths = args.trained_model_paths
    api_key = args.api_key 
    base_url = args.base_url
    evaluator = args.evaluator 
    num_successful_tries = args.num_successful_tries 

    # construct the model and dataset 
    model_parameters = parse_parameters(MODEL_PARAMETERS)
    model_path = model_parameters["language_model"]["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    is_encoder_decoder = model.config.is_encoder_decoder 
    gene_vocab = np.load(os.path.join(GENE_VOCAB_DIR, gene_vocab_file_name))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ignore_index = -100 if not hasattr(model.config, "ignore_index") else None
    pad_token_id = tokenizer.pad_token_id if not hasattr(model.config, "pad_token_id") else None

    datasets = []
    task_types = [] 
    for task_type in TASKS:
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
            no_extra_output_ratio=0.0, 
            is_encoder_decoder=is_encoder_decoder, 
            random_state=random_state, 
        )
        datasets.append((task_type, dataset))   
        task_types.extend([task_type] * len(dataset))    
    questions, candidates = [None] * len(task_types), [{} for _ in range(len(task_types))]   

    # CVAE
    count_dim = len(gene_vocab)
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
    library_log_means, library_log_vars = init_library_size(datasets[0][-1].count_data.X)

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
    mm_model = CellTextLLM(
        model, 
        tokenizer, 
        feature_encoder=feature_encoder, 
        feature_decoder=feature_decoder
    )
    mm_model = mm_model.to(f"cuda:{device_id}")
    mm_model = mm_model.eval()  

    dataloaders = [
        (
            task_type, 
            DataLoader(
                dataset, 
                batch_size=batch_size, 
                collate_fn=collator, 
                num_workers=8, 
                sampler=SequentialSampler(dataset), 
            )
        ) for task_type, dataset in datasets 
    ]

    # get all textual outputs from trained models
    num_invalid_cells = {"ground_truth": 0}  
    for trained_model_path in trained_model_paths:
        mm_model.load_state_dict(torch.load(trained_model_path,  map_location="cpu"))
        p = 0 
        num_invalid_cells[trained_model_path] = 0 
        for task_type, dataloader in dataloaders:
            for batch in dataloader: 
                num_samples = len(batch["input_ids"])
                if questions[-1] is None: 
                    instructions = tokenizer.batch_decode(
                        batch["input_ids"], 
                        skip_special_tokens=True
                    ) 
                    for shift in range(num_samples):
                        questions[p + shift] = postprocess_question(instructions[shift]) 
                    if not drop_ground_truth:
                        labels = batch["labels"].clone() 
                        labels[labels == ignore_index] = mm_model.base_model.config.pad_token_id 
                        responses = tokenizer.batch_decode(
                            labels, 
                            skip_special_tokens=True
                        )
                        # note that for conditional pseudo cell generation 
                        # special tokens are skipped during decoding
                        # so we need to add them back
                        if task_type == CPCG:
                            responses = [response + "\n\n<output>" for response in responses]
                        for shift in range(num_samples):
                            candidates[p + shift]["ground_truth"] = responses[shift]
                batch = {
                    key: value.to(
                        device=next(mm_model.parameters()).device
                    ) if value is not None else value for key, value in batch.items()
                }
                # in the setting of evaluating interactivity
                # decoding algorithm is not greedy 
                outputs = mm_model.generate(
                    batch["input_ids"],
                    batch["input_counts"], 
                    top_p=0.95,
                    top_k=50,
                    do_sample=True,
                    max_new_tokens=256,
                )
                if task_type == CPCG: 
                    answers = [] 
                    for answer, generated_cell in zip(outputs["texts"], outputs["cells"]):
                        if generated_cell is None:
                            num_invalid_cells[trained_model_path] += 1
                            answers.append(answer)
                        else:
                            answers.append(answer + "\n\n<output>")
                else:
                    answers = outputs["texts"]
                for shift in range(num_samples):
                    candidates[p + shift][trained_model_path] = answers[shift]
                p += num_samples

    # shuffle the questions and candidates
    candidates_copy = deepcopy(candidates) 
    for i in range(len(candidates_copy)):
        candidates_copy[i]["question"] = questions[i]
    with open("candidate_answers.json", 'w') as f:
        json.dump(candidates_copy, f, indent=4)
    indices = random_state.permutation(len(questions))
    questions, candidates, task_types = (
        [questions[i] for i in indices], 
        [candidates[i] for i in indices], 
        [task_types[i] for i in indices]
    ) 

    # get the ratings 
    indices, outputs = get_ratings(
        questions, 
        candidates, 
        task_types, 
        api_key, 
        base_url, 
        model=evaluator, 
        num_successful_tries=num_successful_tries,
        num_max_tokens=1024,
        temperature=0.7, 
    )

    # create a dictionary to store results 
    history = {} 
    print("Average Rating")
    print('-' * 50)
    for candidate in sorted(outputs.keys()):
        history[candidate] = {} 
        history[candidate]["evaluation"] = outputs[candidate]["response"]
        history[candidate]["score"] = outputs[candidate]["rating"]
        history[candidate]["num_invalid_cells"] = num_invalid_cells[candidate]
        history[candidate]["average_rating"] = sum(
            int(score) for score in outputs[candidate]["rating"]
        ) / len(outputs[candidate]["rating"])
        history[candidate]["stat"] = dict(Counter(outputs[candidate]["rating"]))
        print(f"{candidate}: {history[candidate]['average_rating']}")
    print('-' * 50)
    print() 

    # in the case that there are two candidates 
    # we display wins/ties/losses 
    if len(history) == 2:
        print("Wins/Ties/Losses")
        print('-' * 50)
        sorted_candidates = sorted(history.keys())
        ego = history[sorted_candidates[0]]["score"]
        other = history[sorted_candidates[1]]["score"]
        results = {
            "tie": 0, 
            "win": 0,
            "loss": 0,
        }
        for i in range(len(ego)):
            if ego[i] == other[i]:
                results["tie"] += 1
            elif ego[i] > other[i]:
                results["win"] += 1
            else:
                results["loss"] += 1
        print(
            f"{sorted_candidates[0]} vs {sorted_candidates[1]} (wins/ties/losses): "
            f"{results['win']}/{results['tie']}/{results['loss']}"
        )
        print('-' * 50)
        print()

    # compute lexical diversity 
    selected_responses = {
        candidate: np.array(
            [
                candidates[indice][candidate] for indice in indices
            ] 
        ) for candidate in history 
    }
    total = 0 
    for candidate in history: 
        candidate_responses = selected_responses[candidate] 
        history[candidate]["lexical_diversity"] = compute_ratio_distinct_ngram(
            candidate_responses, 
            n=1, 
        )
        total += history[candidate]["lexical_diversity"] 
        history[candidate]["output"] = list(candidate_responses)
    # since we conduct comparative analysis 
    # we normalize the lexical diversity scores 
    print("Lexical Diversity")
    print('-' * 50)
    for candidate in sorted(history.keys()):
        history[candidate]["lexical_diversity"] /= total 
        print(f"{candidate}: {history[candidate]['lexical_diversity']}")
    print('-' * 50) 
    with open("ratings.json", 'w') as f:
        json.dump(history, f, indent=4)
