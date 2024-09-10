import time 
import datetime 
from collections import defaultdict, deque 
from typing import (
    Optional, 
    Iterable, 
    Tuple, 
    Any, 
    Dict, 
)
import torch
import torch.distributed as dist

import sys 
sys.path.append('..')
from utils.ddp import is_dist_avail_and_initialized

MB = 1024.0 * 1024.0

class SmoothedValue:
    """
    Track a series of values and computes statistics such as the median, average, and global average. 
    Useful in machine learning and distributed training to monitor performance metrics over time.

    Adapted from https://github.com/salesforce/ALBEF.

    Parameters
    ----------
    window_size : int, optional, default 20
        The size of the window over which the most recent values are tracked. Older values are 
        discarded once the deque exceeds this size.
    fmt : str, optional, default None
        The format string used to display the statistics when converting to a string. If None,
        the default format string "{median:.4f} ({global_avg:.4f})" is used.
    """

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None) -> "SmoothedValue":
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"

        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1) -> None:
        """"Add a value to the tracking deque and updates the cumulative total."""
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self) -> None:
        """Synchronize the total count and total sum across distributed processes (if in distributed mode)."""
        # skip it if not in the distributed training mode.
        if not is_dist_avail_and_initialized():
            return
        
        # if it is in the distributed mode, it assumes you are using GPU  
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        # to synchronizes all processes
        # we use barrier() to block the current processes until the whole group enters this function
        dist.barrier()
        # to synchronizes all processes and aggregate their data
        # we use all_reduce() to reduce the data across all processes
        # the default operation is sum
        # aggregation operations are performed until all processes enter this function
        # aggregated data are broadcasted to all processes
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    # property decorator is used to access the method as an attribute
    @property
    def median(self) -> float:
        """Compute the median of the tracked values."""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self) -> float:
        """Compute the average of the tracked values."""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        """Compute the global average of all values added, not just the ones currently tracked."""
        return self.total / self.count

    @property
    def max(self) -> float:
        """Return the maximum value in the tracked deque with size of ``window_size``."""
        return max(self.deque)

    @property
    def value(self) -> float:
        """Return the most recent value added to the deque."""
        # get the most recent value
        return self.deque[-1]

    def __str__(self) -> str:
        # for bulit-in method format, keyword arguments are used to specify the values to be formatted
        # for those that are not in the format string, they are ignored
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )
    
class MetricLogger:
    r"""
    A class to log and track performance metrics during model training, useful for keeping track 
    of statistics such as loss, training time, and memory usage in real time. The logger can also 
    synchronize metrics across distributed processes.

    Adapted from https://github.com/salesforce/ALBEF.

    Parameters
    ----------
    delimiter : str, optional, default "\t"
        A string used to separate different metrics when printing log messages.
    """

    def __init__(self, delimiter: str = "\t") -> "MetricLogger":

        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs) -> None:
        """Update the tracked values for the given metrics (provided as keyword arguments)."""
        # kwargs is a dictionary
        # for each key-value pair, the key is the name of the metric and the value is the value of the metric
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr: str) -> str | Dict[str, SmoothedValue] | SmoothedValue:
        # getattr() is used to access the method as an attribute
        # view the key of meters as an attribute
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self) -> str:
        loss_str = []
        for name, meter in self.meters.items():
            # call __str__ method of SmoothedValue
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )

        return self.delimiter.join(loss_str)

    def global_avg(self) -> str:
        """Return the global average of all tracked metrics."""
        loss_str = []
        for name, meter in self.meters.items():
            # call global_avg method of SmoothedValue
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )

        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self) -> None:
        """Synchronize the values of all metrics across distributed processes (if in distributed mode)."""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        """Manually add a new metric to be tracked."""
        self.meters[name] = meter
    
    def keys(self) -> Iterable[str]:
        """Return the names of all tracked metrics."""
        return self.meters.keys() 

    def items(self) -> Iterable[Tuple[str, SmoothedValue]]:
        """Return key-value pairs of all tracked metrics and their corresponding ``SmoothedValue`` instances."""
        return self.meters.items() 

    def values(self) -> Iterable[SmoothedValue]:
        """Return the ``SmoothedValue`` instances for all tracked metrics."""
        return self.meters.values() 

    def log_every(
        self, 
        iterable: Iterable[Any], 
        print_freq: int, 
        header: Optional[str] = None,
        desc: Optional[str] = None, 
    ) -> Iterable[Any]:
        """
        Log and track performance metrics at regular intervals during model training.

        This method iterates through the provided iterable object, and at every ``print_freq`` steps, 
        prints the current status of tracked metrics like time per iteration, data loading time, and 
        optionally memory usage (if a GPU is available). It also estimates the remaining time to complete the 
        training epoch.

        Parameters
        ----------
        iterable: a sequence of objects
            An iterable object, such as a ``torch.utils.data.DataLoader`` object, that will be iterated through during 
            training.
        print_freq: int
            The number of iterations between log prints. Metrics will be logged every ``print_freq`` iterations.
        header: str, optional, default None
            An optional string that will be prepended to the log message. This can be used to provide context, 
            such as the phase of training or validation being logged.
        desc: str, optional, default None
            An optional string used to describe the iterable object being logged.
        
        Examples
        --------
        >>> import time 
        >>> import numpy as np 
        >>> class DummyIterator:
        ...     
        ...     def __init__(self, num_samples: int, loading_time: float):
        ...         self.num_samples = num_samples
        ...         self.loading_time = loading_time
        ...         self.counter = 0 
        ...     
        ...     def __len__(self):
        ...         return self.num_samples
        ...     
        ...     def __iter__(self):
        ...         return self
        ...     
        ...     def __next__(self):
        ...         time.sleep(self.loading_time)
        ...         self.counter += 1 
        ...         if self.counter == self.num_samples:
        ...             raise StopIteration
        ...         return self.counter - 1
        ... 
        >>> iterator = DummyIterator(10, 1)
        >>> metric_logger = MetricLogger(delimiter=" -")
        >>> for _ in metric_logger.log_every(
        ...     iterator, 
        ...     print_freq=3, 
        ...     header="Epoch [1/1]",
        ...     desc="Training batch"
        ... ):
        ...     time.sleep(2)
        ...     x, y = np.random.randn(10), np.random.randn(10)
        ...     loss = np.abs(x - y).mean()
        ...     metric_logger.update(loss=loss)
        ... 
        Epoch [1/1] Training batch [3/10] -eta: 0:00:21 -loss: 1.1445 (1.2021) -time: 3.0148 -data: 1.0124 -max mem: 0 MB
        Epoch [1/1] Training batch [6/10] -eta: 0:00:12 -loss: 1.2941 (1.2923) -time: 3.0091 -data: 1.0068 -max mem: 0 MB
        Epoch [1/1] Training batch [9/10] -eta: 0:00:03 -loss: 1.1883 (1.2128) -time: 3.0072 -data: 1.0049 -max mem: 0 MB
        Epoch [1/1] Total time: 0:00:28 (2.8075s/it)
        >>> metric_logger.global_avg()
        'loss: 1.2128'
        """
        assert hasattr(iterable, '__len__'), "The iterable object must have a '__len__' attribute."

        counter = 0
        if header is None:
            header = ''
        start_time = time.time()
        end = time.time()

        # consider the most recent w values where w is the window size
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")

        # make enough space to print the longest training epochs
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            desc + ' ' + "[{0}/{1" + space_fmt + "}]" if desc is not None else "[{0}/{1" + space_fmt + "}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}"
        ]
        
        # record the memory usage
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f} MB")

        log_msg = header + ' ' + self.delimiter.join(log_msg)
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            # (iter_time - data_time) indicates the training time of one batch
            iter_time.update(time.time() - end)
            # after training the model for print_freq steps, we print the related information
            counter += 1 
            if counter % print_freq == 0 or counter == len(iterable):
                # to predict the time of end of training during one epoch
                eta_seconds = iter_time.global_avg * (len(iterable) - counter)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    # cuda.max_memory_allocated
                    # by default, this returns the peak allocated memory since the beginning of this program
                    print(
                        log_msg.format(
                            counter, 
                            len(iterable), 
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), 
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            counter, 
                            len(iterable), 
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), 
                            data=str(data_time)
                        )
                    )
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f}s/it)".format(header, total_time_str, total_time / len(iterable))
        )
