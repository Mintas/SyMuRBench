"""Functions for calculating retrieval metrics."""
import numpy as np
import numpy.linalg as la


def compute_sim_score(
    features_1: np.ndarray,
    features_2: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity between two matrices of embeddings.

    Calculates the pairwise cosine similarity between all embeddings in `features_1`
    and `features_2`, resulting in a similarity matrix.

    Args:
        features_1 (np.ndarray):
            First embedding matrix of shape (n_embeddings_1, embedding_size).
            Must contain numeric values (float or int).
        features_2 (np.ndarray):
            Second embedding matrix of shape (n_embeddings_2, embedding_size).
            Must contain numeric values (float or int).

    Returns:
        np.ndarray:
            Cosine similarity matrix of shape (n_embeddings_1, n_embeddings_2).
    """
    inner_product = features_1 @ features_2.T
    normalization_term = la.norm(features_1, axis=1, keepdims=True)\
          @ la.norm(features_2, axis=1, keepdims=True).T
    logits_per_f1 = inner_product / normalization_term
    return logits_per_f1.T


def get_ranking(
    score_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ranking of items for each query based on similarity scores.

    Given a matrix of similarity scores between queries and items, this function returns
    the indices of items sorted in descending order of similarity for each query, along
    with a matrix of ground-truth query indices for alignment.

    Args:
        score_matrix (np.ndarray): 2D array of shape (num_queries, num_items) containing
            similarity scores (e.g., cosine similarities). Higher values indicate
            greater similarity between a query and an item.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - retrieved_indices (np.ndarray):
                Indices of items ranked by similarity for each query,
                sorted in descending order. Shape: (num_queries, num_items).
            - gt_indices (np.ndarray):
                Ground-truth query indices, where each row corresponds to
                a query and contains its index repeated across all items.
                Shape: (num_queries, num_items).

    Raises:
        ValueError: If `score_matrix` is not a 2D array.
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
    """Compute retrieval evaluation metrics.

    Calculates standard retrieval metrics such as Recall@R and Median Rank by comparing
    the ground-truth item positions with the retrieved (ranked) item positions.

    Args:
        gt_indices (np.ndarray): 2D array of shape (num_queries, num_items) containing
            the ground-truth indices.
        retrieved_indices (np.ndarray):
            2D array of shape (num_queries, num_items) containing
            the indices of items sorted by similarity score in descending order.
        ranks (tuple[int]):
            Tuple of integer cutoffs (e.g., (1, 5, 10)) at which to compute Recall@R.

    Returns:
        dict: Dictionary containing computed metrics:
            - 'R@r': Recall at rank r (percentage of queries where the correct item
                appears within top-r).
            - 'Median_Rank': Median position at which the correct item
                is retrieved across all queries.
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

    Computes standard retrieval performance metrics (e.g., Recall@R and Median Rank)
    by comparing the similarity between two sets of embeddings. The function calculates
    cosine similarity between the embeddings, ranks the results, and evaluates how well
    the most similar items match the ground truth (diagonal pairs).

    Args:
        features_1 (np.ndarray):
            First embedding matrix of shape (n_embeddings, embedding_size),
            containing numeric values (float or int). Each row represents an embedding.
        features_2 (np.ndarray):
            Second embedding matrix of shape (n_embeddings, embedding_size),
            with the same number of embeddings.
        ranks (tuple[int], optional): Tuple of rank cutoffs at which to compute Recall.
            Defaults to (1, 5, 10).

    Returns:
        dict: Dictionary containing retrieval metrics:
            - 'R@r': Recall at rank r (percentage of queries where the correct match
                is in the top-r retrieved results).
            - 'Median_Rank': Median rank of the correct match across all queries.
    """
    score_matrix = compute_sim_score(features_1, features_2)
    retrieved_indices, gt_indices = get_ranking(score_matrix)
    return compute_metrics(gt_indices, retrieved_indices, ranks)
