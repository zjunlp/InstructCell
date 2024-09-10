import numpy as np 
from typing import Tuple, Optional 
import warnings

def init_library_size(
    data: np.ndarray,
    batch_indices: Optional[np.ndarray] = None,
    n_batch: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute and return library size. It is used to compute the prior on the library size.
    Adapted from https://github.com/scverse/scvi-tools."""
    if batch_indices is None:
        n_batch = 1 
        batch_indices = np.zeros(data.shape[0], dtype=np.int32)

    library_log_means = np.zeros(n_batch)
    library_log_vars = np.ones(n_batch)

    for i_batch in np.unique(batch_indices):
        idx_batch = np.squeeze(batch_indices == i_batch)
        batch_data = data[idx_batch.nonzero()[0]]  # h5ad requires integer indexing arrays.
        sum_counts = batch_data.sum(axis=1)
        masked_log_sum = np.ma.log(sum_counts)
        if np.ma.is_masked(masked_log_sum):
            warnings.warn(
                "This dataset has some empty cells, this might fail inference."
                "Data should be filtered with `scanpy.pp.filter_cells()`",
                UserWarning,
            )

        log_counts = masked_log_sum.filled(0)
        library_log_means[i_batch] = np.mean(log_counts).astype(np.float32)
        library_log_vars[i_batch] = np.var(log_counts).astype(np.float32)

    return library_log_means.reshape(1, -1), library_log_vars.reshape(1, -1)