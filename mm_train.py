import torch.nn as nn 
import os    
import numpy as np 
from metadata import ( 
    GENE_VOCAB_DIR, 
    SEED, 
    TASKS, 
    MODEL_PARAMETERS, 
    CPCG, 
) 
import torch 
from torch.utils.data import (
    ConcatDataset,
    DataLoader,   
    RandomSampler, 
    DistributedSampler, 
    SequentialSampler,
) 
import torch.distributed as dist
from metrics.logger import MetricLogger
from utils.ddp import (
    init_distributed_mode, 
    is_main_process, 
    get_world_size,
    get_rank,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
    AutoConfig, 
    Blip2QFormerConfig, 
    Adafactor,
) 
from data_utils import TextCellDataset, TextCellCollator
from mmllm import prepare_cell_text_llm
from mmllm.module import (
    Generator, 
    CellTextLLM, 
    SCQFormer, 
)
from scvi.utils import init_library_size
from scipy.sparse import vstack
from collections import defaultdict
from pathlib import Path 
from utils import (
    str2bool, 
    set_global_random_seed, 
    parse_parameters, 
)
import argparse
from typing import (
    Optional,
    Dict, 
    List, 
)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def train_step(
    model: nn.Module, 
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    print_freq: int = 10,
    grad_norm: Optional[float] = None,
    header: Optional[str] = None, 
    accumulation_steps: int = 1, 
) -> Dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    # create a logger for the training process
    metric_logger = MetricLogger(delimiter=" -")
    counter = 0 
    for samples in metric_logger.log_every(
        dataloader, 
        print_freq, 
        header=header,
        desc="Training batch:", 
    ):
        # samples is a dictionary
        for key in samples:
            if samples[key] is not None:
                samples[key] = samples[key].to(device=model.device, non_blocking=True)
        loss = model(**samples).loss
        # see https://github.com/pytorch/pytorch/issues/43259#issuecomment-964284292
        loss = loss + sum(0.0 * param.sum() for param in model.parameters())
        metric_logger.update(**{"train_loss": loss.item()})
        loss = loss / accumulation_steps
        loss.backward() 
        counter += 1
        if counter % accumulation_steps == 0:     
            # to avert the explosion of gradients 
            if grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            optimizer.step()
            optimizer.zero_grad()
    if counter % accumulation_steps != 0:
        # we just simply discard the remaining gradients
        optimizer.zero_grad()
    
    # to synchronize all processes
    # at this moment, we can get the global average of the loss computed across all batches in the dataloader
    metric_logger.synchronize_between_processes()
    # we print it in the master process
    print(f"Metrics (averaged) after training for one epoch: [{metric_logger.global_avg()}]")

    return {k: v.global_avg for k, v in metric_logger.items()}

def evaluate_step(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader,
    print_freq: int = 10,
    header: Optional[str] = None, 
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()

    # create a logger for the evaluation process
    metric_logger = MetricLogger(delimiter=" -")

    with torch.no_grad():
        for samples in metric_logger.log_every(
            dataloader, 
            print_freq, 
            header=header, 
            desc="Validating batch:", 
        ):
            for key in samples:
                if samples[key] is not None:
                    samples[key] = samples[key].to(device=device, non_blocking=True)
            outputs = model(**samples)
            loss = outputs.loss 
            metric_logger.update(**{"eval_loss": loss.item()})

    print(f"Metrics (averaged) during validation: [{metric_logger.global_avg()}]")

    return {k: v.global_avg for k, v in metric_logger.meters.items()}

def train(
    model: nn.Module, 
    args: argparse.Namespace, 
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader = None,  
    start_epoch: int = 1,
    best_val: Optional[float] = None, 
) -> Dict[str, List[float]]:
    """Train the model."""
    logger = defaultdict(list)
    if args.save_dir is not None:
        args.save_dir = Path(args.save_dir)
        args.save_dir.mkdir(exist_ok=True, parents=True)
    
    best_metric = float('inf') if best_val is None else best_val
    best_fn = lambda x: x < best_metric
    for i in range(start_epoch - 1, args.epochs):
        header = f"Training Epoch: [{i + 1}/{args.epochs}]"
        if args.distributed:
            train_dataloader.sampler.set_epoch(i)
        metric_logger = train_step(
            model, 
            train_dataloader,
            optimizer,
            print_freq=args.print_freq,
            grad_norm=args.grad_norm,
            header=header,
            accumulation_steps=args.accumulation_steps, 
        )
        for key, value in metric_logger.items():
            logger[key + " (training mode)"].append(value)
        if valid_dataloader is not None:
            logs = evaluate_step(model if not args.distributed else model.module, valid_dataloader)
            if args.best_model_name is not None:
                metric = logs["eval_loss"]
                if best_fn(metric):
                    best_metric = metric
                    save_path = f"{args.best_model_name}.pkl"
                    torch.save(model.state_dict() if not args.distributed else model.module.state_dict(), save_path)
            for key, value in logs.items():
                logger[key + " (validation mode)"].append(value)

        # if val_dataloader is not None, only the master process execute the evaluation step
        # other processes will wait until the master process finishes the evaluation step
        if args.distributed:
            dist.barrier()
        
        if args.save_dir is not None and is_main_process() and (i + 1) % args.save_freq == 0:
            # save the model.
            save_path = args.save_dir / f"model_checkpoint_{i + 1}.pth" 
            state_dict = {
                "model": model.state_dict() if not args.distributed else model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": i + 1
            }
            torch.save(state_dict, save_path)

    return logger, best_metric
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parameters for distributed data parallel training
    parser.add_argument("--device", default="cuda", type=str, help="the device to use")
    parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")
    # parameters for tokenizer 
    parser.add_argument("--modality_tag", default="CELL", type=str, help="the name of added modality")
    parser.add_argument("--num_signal_tokens", default=1, type=int, help="the number of signal tokens")
    parser.add_argument(
        "--force_gene_symbol_uppercase", 
        default=False, 
        type=str2bool, 
        help="whether to force gene symbols to be uppercase or not"
    )
    # parameters for datasets 
    parser.add_argument(
        "--train_no_extra_output_ratio", default=1.0, type=float,
        help="the ratio of training samples without extra text outputs"
    )
    parser.add_argument(
        "--eval_no_extra_output_ratio", default=1.0, type=float,
        help="the ratio of evaluation samples without extra text outputs"
    )
    parser.add_argument("--provide_choices", default=None, type=str2bool, help="whether to provide choices or not")
    parser.add_argument("--unify_gene", default=True, type=str2bool, help="whether to unify gene symbols or not")
    # for templates construction 
    parser.add_argument("--train_template_dir", default=None, type=str, help="the directory of training templates")
    parser.add_argument("--valid_template_dir", default=None, type=str, help="the directory of evaluation templates")
    # hyper-parameters of datalaoder 
    parser.add_argument("--batch_size", default=64, type=int, help="the batch size of the dataloader")
    # hyper-parameters of model
    parser.add_argument(
        "--from_pretrained", 
        default=True, 
        type=str2bool, 
        help="whether to load the model from the pretrained model or not"
    )
    
    # hyper-parameters of training
    parser.add_argument("--grad_norm", default=None, type=float, help="the maximum norm of gradients")
    parser.add_argument("--epochs", default=250, type=int, help="the number of training epochs")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="the learning rate of the optimizer")
    parser.add_argument("--print_freq", default=10, type=int, help="the frequency of printing training information")
    parser.add_argument("--save_dir", default="./checkpoints/", type=str, help="the directory of saving checkpoints")
    parser.add_argument("--save_freq", default=50, type=int, help="the frequency of saving checkpoints")
    parser.add_argument("--best_model_name", default="best_mm_model", type=str, help="the file name of the best model")
    parser.add_argument(
        "--accumulation_steps",
        default=1,
        type=int,
        help="the number of accumulation steps" 
    )
    # parameters of checkpoints 
    parser.add_argument(
        "--resume", 
        default=False, 
        type=str2bool, 
        help="whether to resume training from the latest checkpoint or not"
    )
    parser.add_argument(
        "--resume_path", 
        default="./checkpoints/model_checkpoint.pth", 
        type=str, 
        help="the file path of the latest checkpoint"
    )
    args = parser.parse_args()

    init_distributed_mode(args)
    set_global_random_seed(SEED, libraries=["torch"])
    model_parameters = parse_parameters(MODEL_PARAMETERS)

    dir_names = TASKS
    task_types = list(dir_names.keys())

    model_path = model_parameters["language_model"]["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if args.from_pretrained:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_config(config)
    if args.unify_gene:
        gene_vocab = np.load(os.path.join(GENE_VOCAB_DIR, "gene_vocab.npy"))
    else:
        gene_vocab = None 
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    is_encoder_decoder = model.config.is_encoder_decoder

    ignore_index = -100 if not hasattr(model.config, "ignore_index") else None
    pad_token_id = tokenizer.pad_token_id if not hasattr(model.config, "pad_token_id") else None
    random_state = np.random.default_rng(SEED)

    datasets = {}
    count_matrix = [] 
    pointer = None 
    for split in ["train", "valid"]:
        dataset_collection = [] 
        template_dir_name = None 
        if split == 'train' and args.train_template_dir is not None:
            template_dir_name = args.train_template_dir
        if split == 'valid' and args.valid_template_dir is not None:
            template_dir_name = args.valid_template_dir
        for task_type in task_types:
            dataset = TextCellDataset(
                dir_name=dir_names[task_type], 
                tokenizer=tokenizer, 
                task_type=task_type, 
                template_dir_name=template_dir_name, 
                split=split, 
                gene_vocab=gene_vocab, 
                modality=args.modality_tag, 
                num_signal_tokens=args.num_signal_tokens, 
                force_gene_symbol_uppercase=args.force_gene_symbol_uppercase, 
                provide_choices=args.provide_choices, 
                no_extra_output_ratio=args.train_no_extra_output_ratio if split == 'train' else args.eval_no_extra_output_ratio, 
                is_encoder_decoder=is_encoder_decoder,
                random_state=random_state, 
            )   
            dataset_collection.append(dataset)
            if split == 'train':
                count_matrix.append(dataset.count_data.X)
                if task_type == CPCG:
                    pointer = len(count_matrix) - 1
        dataset = ConcatDataset(dataset_collection)
        datasets[split] = dataset
    # concat the csr_matrix 
    if pointer is not None:
        count_matrix_simulation = count_matrix[pointer]
    else:
        count_matrix_simulation = None
    count_matrix = vstack(count_matrix)

    print(f"The shape of the count matrix: {count_matrix.shape}")
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
    library_log_means, library_log_vars = init_library_size(
        count_matrix if count_matrix_simulation is None else count_matrix_simulation,
    )

    # MMLLM
    if model_parameters["feature_encoder"]["is_q_former_encoder"]:
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
    feature_encoder.to(dtype=model.dtype)
    feature_decoder.to(dtype=model.dtype)

    model, tokenizer = prepare_cell_text_llm(
        model, 
        tokenizer, 
        modality_tag=args.modality_tag,
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
        feature_decoder=feature_decoder,  
    )

    train_dataset = datasets["train"]
    valid_dataset = datasets["valid"]
    if args.distributed:
        world_size = get_world_size() 
        local_rank = get_rank() 
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=local_rank,
            seed=SEED, 
        )
        # We only evaluate the model in the main process. 
        valid_sampler = SequentialSampler(valid_dataset) if is_main_process() else None
    else:
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)
    if args.distributed:
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            collate_fn=collator,
            num_workers=8, 
            sampler=train_sampler,
            pin_memory=True, 
        )
        valid_dataloader = DataLoader(
            valid_dataset, 
            batch_size=args.batch_size, 
            collate_fn=collator,
            sampler=valid_sampler, 
            num_workers=8
        ) if is_main_process() else None
    else:
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            collate_fn=collator,
            num_workers=8, 
            sampler=train_sampler,
        )
        valid_dataloader = DataLoader(
            valid_dataset, 
            batch_size=args.batch_size, 
            collate_fn=collator,
            sampler=valid_sampler, 
            num_workers=8
        ) 

    # begin training 
    epochs = args.epochs
    optimizer = Adafactor(mm_model.parameters(), lr=args.learning_rate, relative_step=False)
    device = args.device 
    # set model to the current device 
    mm_model.to(device=device)

    # Recover the training states.
    start_epoch = 1
    if args.resume_path is not None and args.resume:
        # Load the model.
        # If the states are on GPU, it will be converted to CPU first in order to avert 
        # the GPU memory surge when loading the model.
        state_dict = torch.load(args.resume_path, map_location='cpu')
        mm_model.load_state_dict(state_dict['model'])
        if args.resume:
            optimizer.load_state_dict(state_dict['optimizer'])
            start_epoch = state_dict['epoch'] + 1
            print(f"Resume training from epoch {start_epoch}...")
            r_epochs = epochs - start_epoch + 1
            print(f"Total epochs remaining: {r_epochs}")
    

    epochs = args.epochs
    # args.gpu has been set in init_distributed_mode() if args.distributed. 
    if args.distributed:
        model_ = torch.nn.parallel.DistributedDataParallel(mm_model, device_ids=[args.gpu])
    else:
        model_ = mm_model

    train(
        model_, 
        args, 
        optimizer, 
        train_dataloader, 
        valid_dataloader, 
        start_epoch=start_epoch,
    )
