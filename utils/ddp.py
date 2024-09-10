import torch.distributed as dist 
import torch 
import os   
from importlib.util import find_spec 
import argparse

MASTER_ID = 0 

def is_dist_avail_and_initialized() -> bool:
    """
    Check if PyTorch's distributed package is available and initialized.

    Returns
    -------
    is_avail_and_initialized: bool
        Return True if distributed training is available and the process group has 
        been initialized, otherwise return False.

    Notes
    -----
    - The distributed package can be initialized using ``dist.init_process_group``.
    """
    # check if distributed training is available
    # return True if the distributed package is available
    if not dist.is_available():
        return False
    
    # check whether the default process group has been initialized
    # to initialize the default process group, call init_process_group()
    if not dist.is_initialized():
        return False
    
    return True

def get_world_size() -> int:
    """
    Get the number of processes involved in the distributed computation (world size).

    Returns
    -------
    world_size: int
        The number of processes involved in the distributed computation. Return 1 if
        distributed training is uninitialized or only one process is involved.

    Notes
    -----
    - To use distributed computing, PyTorch needs to initialize a process group using 
      ``dist.init_process_group``. Once initialized, this function will reflect the 
      number of processes involved.
    """
    if not is_dist_avail_and_initialized():
        return 1
    
    return dist.get_world_size()


def get_rank() -> int:
    """
    Get the rank of the current process in a distributed training setup.

    This function returns the rank of the calling process in the current 
    distributed group. Rank is used to identify processes in distributed 
    training. The process with rank 0 is usually referred to as the master.

    Returns
    -------
    rank: int
        The rank of the current process. If the distributed training is not 
        initialized, the function returns 0. 

    Notes
    -----
    - To use this function effectively, ensure that the distributed process group 
      has been initialized using ``dist.init_process_group``.
    """
    if not is_dist_avail_and_initialized():
        return 0
    
    return dist.get_rank()
    
def setup_for_distributed(is_master: bool) -> None:
    """
    Adjust the behavior of the ``print`` function for distributed training.

    This function is useful when running training across multiple processes in a distributed setup, 
    where only the master process should print logs or messages to avoid cluttering the output.

    Parameters
    ----------
    is_master: bool
        A boolean flag indicating whether the current process is the master process. 
        If True, the current process will print information as usual.

    Notes
    -----
    - After init_distributed_mode is called, use ``print(..., force=True)`` to force printing in 
      non-master processes. 
    - This function will change the logging configuration in non-master processes after being called, 
      but you can restore the original logging configuration.
    """
    # all built-in functions are available in the builtins module
    import builtins as __builtin__
    builtin_print = __builtin__.print

    # if the current process is not the master process, only errors are printed 
    if not is_master:
        if find_spec("transformers") is not None:
            import transformers 
            transformers.logging.set_verbosity_error()
        import logging
        logging.basicConfig(level=logging.ERROR)
        import warnings
        warnings.filterwarnings("ignore")

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        # if is_master is True or force is set to True, print the information
        if is_master or force:
            builtin_print(*args, **kwargs)

    # __builtin__.print = print is to replace the built-in function print with the function print defined above
    __builtin__.print = print

def init_distributed_mode(args: argparse.Namespace) -> None:
    """
    Initialize the distributed mode for PyTorch training.

    Parameters
    ----------
    args : argparse.Namespace
        This will be modified in place to set ``rank``, ``world_size``, ``gpu``, ``dist_backend``, 
        and ``distributed`` fields.

    Notes
    -----
    - The function configures distributed training using the NCCL backend and initializes the process group.
    - The ``setup_for_distributed`` function is called to adjust the behavior of printing and logging 
      across different processes, where only the master process (rank 0) logs outputs by default.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # RANK and WORLD_SIZE are environment variables which have been set by torch.distributed.launch.
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    # set your device to local rank
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)
    dist.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=args.rank
    )
    # wait until all processes has been initialized
    dist.barrier()

    # here, we define 0 as the master process
    # we only print the information in the master process
    setup_for_distributed(args.rank == MASTER_ID)

def is_main_process() -> bool:
    """
    Check if the current process is the main process in a distributed training setup.

    In distributed training, multiple processes are involved, and this function 
    determines whether the current process is the main one, typically referred to 
    as the master process. The master process is identified by having a rank equal 
    to ``MASTER_ID``.

    Returns
    -------
    is_master: bool
        Return True if the current process has the rank equal to ``MASTER_ID``, indicating
        it is the main process. Otherwise, return False.
    """
    return get_rank() == MASTER_ID

