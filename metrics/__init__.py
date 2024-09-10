from typing import (  
    List, 
    Optional, 
    Tuple, 
)
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import accuracy_score, f1_score
import nltk
from scipy.stats import permutation_test 

def _average_precision(match: np.ndarray) -> float:
    """Compute the average precision."""
    if np.any(match):
        return np.sum(match) / match.shape[-1]
    
    return 0.0

def measure_simulation(
    predictions: np.ndarray,
    targets: np.ndarray,
    prediction_labels: np.ndarray, 
    target_labels: np.ndarray,
    k: int = 20, 
) -> float:
    """The implementation of RKNN."""
    knn = KNeighborsClassifier(n_neighbors=k, metric="minkowski")
    knn.fit(targets, target_labels)
    pred = knn.predict(predictions)
    accuracy = np.sum(pred == prediction_labels) / len(prediction_labels)

    return accuracy

def measure_bio_preservation(
    predictions: np.ndarray,
    targets: Optional[np.ndarray] = None,
    prediction_labels: Optional[np.ndarray] = None,
    target_labels: Optional[np.ndarray] = None,
    k: int = 20, 
) -> float:
    """The implementation of SKNN."""
    if prediction_labels is None:
        raise ValueError("The prediction labels must be provided")

    # k + 1: the first neighbor is the sample itself
    nn = NearestNeighbors(
        n_neighbors=min(prediction_labels.shape[0], k + 1), 
        metric="minkowski"
    ).fit(predictions)
    nni = nn.kneighbors(predictions, return_distance=False)
    # remove the first neighbor because it is the sample itself
    match = np.equal(prediction_labels[nni[:, 1:]], np.expand_dims(prediction_labels, 1))

    return np.apply_along_axis(_average_precision, 1, match).mean().item()

def estimate_bandwidth(
    samples: np.ndarray,  
    num_samples_for_bandwidth_estimation: Optional[int] = None,
    num_repeats: int = 20, 
    n_neighbours: Optional[int] = 25,
    random_state: int | np.random.RandomState | np.random.Generator | None = None,
) -> float:
    """
    Estimate the bandwidth for a Gaussian kernel using the median heuristic.

    This function estimates the bandwidth (gamma) for an RBF (Radial Basis Function) kernel using
    either the [median heuristic](https://arxiv.org/abs/1707.07269) [1] or the 
    [k-nearest neighbors (k-NN) method](https://academic.oup.com/bioinformatics/article/33/16/2539/3611270) [2].

    Parameters
    ----------
    samples: np.ndarray
        An array of shape (n_samples, n_features) representing the dataset from which the bandwidth
        is estimated.
    num_samples_for_bandwidth_estimation: int, optional, default None
        The number of samples to use for estimating the bandwidth. If None, all samples are used. 
        When there are many samples, using a subset can lower computational cost.
    num_repeats: int, default 20
        The number of sampling for the bandwidth estimation, to obtain a more robust estimate.
        This applies only when ``num_samples_for_bandwidth_estimation`` is not None.
    n_neighbours: int, optional, default 25
        The number of nearest neighbors to consider in the k-NN method for bandwidth estimation.
        If None, the median pairwise distances [1] between all samples are used instead. 
    random_state: int | np.random.RandomState | np.random.Generator | None, optional, default None
        A seed or random number generator for reproducibility. If None, the default random 
        generator is used. If ``num_samples_for_bandwidth_estimation`` is equal to the number of
        samples, the random state is ignored.

    Returns
    -------
    bandwidth: float
        The estimated bandwidth value (gamma) for the RBF kernel.
    
    References
    ----------
    [1] Garreau, D., Jitkrittum, W., & Kanagawa, M. (2017). Large sample analysis of the median heuristic. 
    *arXiv preprint arXiv:1707.07269*.

    [2] Shaham, U., Stanton, K. P., Zhao, J., Li, H., Raddassi, K., Montgomery, R., & Kluger, Y. (2017). 
    Removal of batch effects using distribution-matching residual networks. *Bioinformatics*, *33*(16), 2539-2546.
    """ 
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)

    if num_samples_for_bandwidth_estimation is None:
        num_repeats = 1
        num_samples_for_bandwidth_estimation = len(samples)

    # to lower the computational cost, we can use a subset of the samples to estimate the bandwidth
    gamma = np.zeros(num_repeats)
    for i in range(num_repeats):
        samples_ = random_state.choice(samples, num_samples_for_bandwidth_estimation, replace=False)
        if n_neighbours is None:
            # see Garreau et al (Large sample analysis of the median heuristic) 
            distances = np.sqrt(np.square(samples_[:, None, :] - samples_[None, :, :]).sum(axis=-1))
            distances = distances[np.triu_indices_from(distances, k=1)]
            gamma[i] = np.median(distances)
        else:
            # see Shaham et al (Removal of batch effects using distribution-matching residual networks)
            neigh = NearestNeighbors(n_neighbors=n_neighbours + 1).fit(samples_)
            neigh_dist, _ = neigh.kneighbors(samples_, return_distance=True)
            # remove self
            neigh_dist = neigh_dist[:, 1: ]
            gamma[i] = np.median(neigh_dist)
    gamma = np.median(gamma)

    return 1 / (gamma * gamma)

def compute_biased_mmd_rbf(
    predictions: np.ndarray, 
    targets: np.ndarray, 
    prediction_labels: Optional[np.ndarray] = None,
    target_labels: Optional[np.ndarray] = None,
    gamma: float | List[float] | None = None, 
    num_samples_for_bandwidth_estimation: Optional[int] = None, 
    num_repeats: int = 20, 
    n_neighbours: Optional[int] = 25, 
    random_state: int | np.random.RandomState | np.random.Generator | None = None,
    n_permutations: Optional[int] = 1000,
    return_pvalue: bool = False, 
) -> float | Tuple[float, float]:
    """
    Compute the biased Maximum Mean Discrepancy (MMD) using the Radial Basis Function (RBF) kernel.

    This function estimates the distribution difference between two datasets (predictions and targets) 
    using the biased MMD approach with an RBF kernel. MMD is widely used in hypothesis testing to 
    determine if two distributions are significantly different.

    Parameters
    ----------
    predictions: np.ndarray
        An array of shape (n_samples_pred, n_features) containing the predictions (or sample set 1).
    targets: np.ndarray
        An array of shape (n_samples_target, n_features) containing the target data (or sample set 2).
    prediction_labels: np.ndarray, optional, default None
        It is ignored.
    target_labels: np.ndarray, optional, default None 
        It is ignored.
    gamma: float | list of float | None, optional, default None
        The bandwidth (gamma) parameter for the RBF kernel. If None, the bandwidth is estimated using 
        the median heuristic. Multiple values of gamma can be provided, and the result will be the summation 
        of kernels with different gamma values.
    num_samples_for_bandwidth_estimation: int, optional, default None
        The number of samples to use for bandwidth estimation. If None, all samples are used. If ``gamma``
        is not None, this parameter is ignored.
    num_repeats: int, default 20
        The number of times to repeat the sampling process for bandwidth estimation (only applies if 
        ``num_samples_for_bandwidth_estimation`` is not None and ``gamma`` is None).
    n_neighbours: int, optional, default 25
        The number of nearest neighbors to use for bandwidth estimation using the k-nearest neighbors 
        (k-NN) approach [2]. If None, the median heuristic [1] is used instead.
    random_state: int | np.random.RandomState | np.random.Generator | None, optional
        A seed or random number generator for reproducibility. It is used for sampling the subset of
        samples for bandwidth estimation and permutation testing.
    n_permutations: int, optional, default 1000
        The number of permutations to perform when estimating the p-value. If None or ``return_pvalue`` is
        set to False, no permutation test is performed.
    return_pvalue: bool, default False
        If True, the p-value from the permutation test is returned alongside the MMD statistic.

    Returns
    -------
    stat: float
        The MMD statistic between predictions and targets.
    pvalue: float, optional
        The p-value from the permutation test, only returned if ``return_pvalue`` is set to True.

    References
    ----------
    [1] Garreau, D., Jitkrittum, W., & Kanagawa, M. (2017). Large sample analysis of the median heuristic. 
    *arXiv preprint arXiv:1707.07269*.

    [2] Shaham, U., Stanton, K. P., Zhao, J., Li, H., Raddassi, K., Montgomery, R., & Kluger, Y. (2017). 
    Removal of batch effects using distribution-matching residual networks. *Bioinformatics*, *33*(16), 2539-2546.

    Examples
    --------
    >>> a = np.random.normal(0, 1, (500, 10))
    >>> b = np.random.normal(0, 1, (500, 10))
    >>> compute_biased_mmd_rbf(predictions, targets, gamma=1.0)
    0.004009370623184339

    Measure the distribution difference between two datasets with permutation test.

    >>> compute_biased_mmd_rbf(
    ...     a, 
    ...     b, 
    ...     gamma=1.0, 
    ...     n_permutations=200, 
    ...     return_pvalue=True, 
    ... )
    (0.004009370623184339, 0.3781094527363184)

    A very small p value (< 0.05) means the two distributions are significantly different.

    >>> c = np.random.normal(0.5, 2, (500, 10)) 
    >>> compute_biased_mmd_rbf(a, c, gamma=1.0)
    0.004312268936380403
    >>> compute_biased_mmd_rbf(
    ...     a, 
    ...     c, 
    ...     gamma=1.0, 
    ...     n_permutations=200, 
    ...     return_pvalue=True, 
    ... )
    (0.004312268936380403, 0.004975124378109453)

    Multi-scale trick can be used to improve sensitivity of MMD. You can simply set ``gamma`` to None.

    >>> compute_biased_mmd_rbf(a, b)
    0.00948193802177779
    >>> compute_biased_mmd_rbf(a, c)
    0.5295954553952591
    """
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)

    if gamma is None:
        # we use median heuristic to estimate the bandwidth
        gamma = estimate_bandwidth(
            targets, 
            num_samples_for_bandwidth_estimation=num_samples_for_bandwidth_estimation,
            num_repeats=num_repeats,
            n_neighbours=n_neighbours,
            random_state=random_state,
        )
        gamma = [gamma / 2, gamma, gamma * 2]
    if isinstance(gamma, (float, int)):
        gamma = [gamma]

    m, n = predictions.shape[0], targets.shape[0]
    all_samples = np.concatenate((predictions, targets), axis=0)
    pred_indices, tgt_indices = np.arange(0, m), np.arange(m, m + n)   
    
    def biased_mmd_rbf(pred_indices: np.ndarray, tgt_indices: np.ndarray) -> float:
        
        predictions, targets = all_samples[pred_indices], all_samples[tgt_indices]

        res = 0
        for coeff in gamma:
            a = rbf_kernel(predictions, predictions, gamma=coeff)
            b = rbf_kernel(predictions, targets, gamma=coeff)
            c = rbf_kernel(targets, targets, gamma=coeff)
            res += a.mean() - 2 * b.mean() + c.mean()
        return res 
    
    if n_permutations is None or not return_pvalue:
        return biased_mmd_rbf(pred_indices, tgt_indices)

    stat = permutation_test(
        (pred_indices, tgt_indices), 
        biased_mmd_rbf,
        n_resamples=n_permutations,
        alternative='greater',
        vectorized=False,
        permutation_type='independent', 
        random_state=random_state,
    )

    return (stat.statistic, stat.pvalue)

def measure_classification_accuracy_text(
    predictions: np.ndarray,
    targets: np.ndarray,
    prediction_labels: Optional[np.ndarray] = None,
    target_labels: Optional[np.ndarray] = None,
) -> float:
    """Compute the classification accuracy."""
    return accuracy_score(targets, predictions)

def measure_classification_f1_score_text(
    predictions: np.ndarray,
    targets: np.ndarray,
    prediction_labels: Optional[np.ndarray] = None,
    target_labels: Optional[np.ndarray] = None,
    average: str = 'binary', 
) -> float:
    """Compute the F1 score for classification."""
    if target_labels is not None:
        target_labels = np.unique(target_labels)
    
    return f1_score(
        targets, 
        predictions, 
        labels=target_labels, 
        average=average
    )

def compute_ratio_distinct_ngram(
    predictions: np.ndarray,
    targets: Optional[np.ndarray] = None,
    prediction_labels: Optional[np.ndarray] = None,
    target_labels: Optional[np.ndarray] = None,
    n: int = 1, 
) -> float:
    """
    Compute the ratio of distinct n-grams for grouped sentences. This function calculates the 
    distinct n-grams within each group of sentences and returns the average ratio of unique 
    n-grams across all groups.

    The sentences are divided into groups based on the ``prediction_labels``. If no labels are 
    provided, all predictions are treated as part of the same group.

    Parameters
    ----------
    predictions: np.ndarray
        A numpy array of sentences.
    targets: np.ndarray, optional, default None
        It will be ignored. 
    prediction_labels: np.ndarray, optional, default None
        An array of labels corresponding to each sentence. If not provided, all sentences 
        are treated as a single group.
    target_labels: np.ndarray, optional, default None
        It will be ignored. 
    n: int, default 1
        The size of the n-grams to compute. For example, `n=1` computes unigrams, `n=2` computes 
        bigrams, and so on.

    Returns
    -------
    ratio: float
        The average ratio of distinct n-grams across all groups. The ratio is computed as the 
        number of unique n-grams divided by the total number of n-grams in each group, averaged 
        over all groups.

    Examples
    --------
    Compute the unigram diversity of a corpus.

    >>> corpus = np.array(["the cat sat", "how are you", "the dog ran"])
    >>> compute_ratio_distinct_ngram(corpus)
    0.8888888888888888

    Compute the bigram diversity of a set of sentences with labels.

    >>> sentences = np.array(
    ...     [
    ...         "Not bad",
    ...         "I'm doing well",
    ...         "He plans to go swimming", 
    ...         "He intends to go swimming"
    ...     ]
    ... )
    >>> prediction_labels = np.array([0, 0, 1, 1], dtype=int)
    >>> compute_ratio_distinct_ngram(sentences, prediction_labels=prediction_labels, n=2)
    0.875
    """
    if prediction_labels is None:
        prediction_labels = np.full_like(predictions, fill_value=0)
    group_num_distinct_ngram = {} 
    group_total_num_ngram = {} 

    for i, prediction in enumerate(predictions): 
        group = prediction_labels[i]
        if group not in group_num_distinct_ngram: 
            group_num_distinct_ngram[group] = set() 
            group_total_num_ngram[group] = 0 
        current_ngram = nltk.ngrams(prediction.split(), n=n)
        for ngram in current_ngram: 
            group_num_distinct_ngram[group].add(ngram)
            group_total_num_ngram[group] += 1 
    
    return sum(
        len(group_num_distinct_ngram[key]) / group_total_num_ngram[key] for key in group_num_distinct_ngram
    ) / len(group_total_num_ngram)