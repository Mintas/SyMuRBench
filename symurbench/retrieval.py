"""Functions for calculating retrieval metrics."""
import numpy as np
import numpy.linalg as la


def compute_sim_score(
    features_1: np.ndarray,
    features_2: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between two matrices of embeddings.

    Args:
        features_1 (np.ndarray, float or int):
            matrix 1 of shape (n_embeddings, embedding_size)
        features_2 (np.ndarray, float or int):
            matrix 2 of shape (n_embeddings, embedding_size)

    Returns:
        np.ndarray: cosine similarity matrix of shape (n_embeddings, n_embeddings)
    """
    inner_product = features_1 @ features_2.T
    normalization_term = la.norm(features_1, axis=1, keepdims=True)\
          @ la.norm(features_2, axis=1, keepdims=True).T
    logits_per_f1 = inner_product / normalization_term
    return logits_per_f1.T


def get_ranking(
    score_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rank similarity scores.

    Args:
        score_matrix (np.ndarray): matrix with similarity scores

    Raises:
        ValueError: if score_matrix.shape has number of dimensions other than 2

    Returns:
        tuple[np.ndarray, np.ndarray]:
            matrices with 1) indices of sorted objects;
                          2) indices with objects in the initial order
    """
    if len(score_matrix.shape) != 2:
        msg = "Incorrect score_matrix shape. It should be equal to 2."
        raise ValueError(msg)

    num_queries, num_items = score_matrix.shape

    retrieved_indices = np.argsort(-score_matrix, axis=1)
    gt_indices = np.tile(np.arange(num_queries)[:, np.newaxis], (1, num_items))

    return gt_indices, retrieved_indices


def compute_metrics(
    gt_indices: np.ndarray,
    retrieved_indices: np.ndarray,
    ranks: tuple[int]
) -> dict:
    """
    Compute retrieval metrics.

    Args:
        gt_indices (np.ndarray):
            a matrix with indices of objects in the initial order
        retrieved_indices (np.ndarray):
            a matrix with indices of sorted objects (by similarity)
        ranks (tuple[int]): a tuple of ranks to calculate

    Returns:
        dict: calculated metrics
    """
    num_items = gt_indices.shape[1]

    bool_matrix = retrieved_indices == gt_indices
    retrieval_metrics = {}
    for r in ranks:
        rank_value = 100 * bool_matrix[:, :r].sum() / num_items
        retrieval_metrics[f"R@{r}"] = rank_value

    median_rank = np.median(np.where(bool_matrix)[1] + 1)
    retrieval_metrics["Median_Rank"] = median_rank

    return retrieval_metrics

def run_retrieval_from_embeddings(
    features_1: np.ndarray,
    features_2: np.ndarray,
    ranks: tuple[int] = (1,5,10)
) -> dict:
    """
    Calculate retrieval metrics between two matrices of embeddings.

    Args:
        features_1 (np.ndarray, float or int):
            matrix 1 of shape (n_embeddings, embedding_size)
        features_2 (np.ndarray, float or int):
            matrix 2 of shape (n_embeddings, embedding_size)
        ranks (tuple[int], optional):
            tuple of ranks to calculate.
            Defaults to (1,5,10).

    Returns:
        dict: calculated metrics
    """
    score_matrix = compute_sim_score(features_1, features_2)
    retrieved_indices, gt_indices = get_ranking(score_matrix)
    return compute_metrics(gt_indices, retrieved_indices, ranks)
