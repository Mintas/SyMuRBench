"""Base class for retrieval tasks."""
import numpy as np
import pandas as pd

from symurbench.metrics.metric_value import MetricValue
from symurbench.metrics.scorer import BaseScorer
from symurbench.retrieval import compute_sim_score, get_ranking

from .abstask import AbsTask


class RetrievalTask(AbsTask):
    """A class for evaluating FeatureExtractor on a retrieval task."""

    def __init__(
        self,
        metaloader_args_dict: dict,
        scorer: BaseScorer,
        postfixes: tuple[str, str] = ("sp", "ps")
    ) -> None:
        """
        Initialize the task and prepare the dataset for feature extraction.

        Args:
            metaloader_args_dict (dict):
                Dictionary of arguments passed to the metaloader constructor.
                Expected keys:
                - metadata_csv_path (str):
                Absolute path to the CSV file containing dataset metadata.
                - files_dir_path (str):
                Absolute path to the directory containing dataset files.
                - dataset_filter_list (list[str], optional):
                List of filenames to include (inclusion filter).
            scorer (BaseScorer):
                The scorer instance used to compute evaluation metrics.
            postfixes (tuple[str, str], optional): A pair of string postfixes used to
                distinguish metrics computed in different directions (e.g., "sp" for
                score-performance and "ps" for performance-score). Defaults to
                ("sp", "ps").
        """
        super().__init__(metaloader_args_dict)
        self.scorer = scorer
        self.postfixes = postfixes

    def calculate_metrics_one_direction(
        self,
        features_1: np.ndarray,
        features_2: np.ndarray,
        postfix: str = ""
    ) -> list[MetricValue]:
        """
        Calculate retrieval metrics for pair of embedding matrices.

        Args:
            features_1 (np.ndarray, float or int):
                matrix 1 of shape (n_embeddings, embedding_size)
            features_2 (np.ndarray, float or int):
                matrix 2 of shape (n_embeddings, embedding_size)
            postfix (str, optional): postfix to add to metric name.
                Defaults to "".

        Returns:
            list[MetricValue]: Calculated retrieval metrics.
        """
        score_matrix = compute_sim_score(features_1, features_2)
        indices = get_ranking(score_matrix)
        metrics = self.scorer.calculate_metrics(
            y_true=indices[0],
            preds=indices[1]
        )
        for i in range(len(metrics)):
            metrics[i].name = (metrics[i].name + f"_{postfix}")

        return metrics

    def calculate_metrics(
        self,
        data: pd.DataFrame
    ) -> list[MetricValue]:
        """
        Calculate metrics for features.

        Args:
            data (pd.DataFrame):
                dataframe with features

        Raises:
            ValueError: if data.shape[0] is not an even number

        Returns:
            list[MetricValue]:
                List of MetricValue objects (for each calculated metric).
                Each object has an unique name.
        """
        if data.shape[0] % 2 != 0:
            msg = "Number of midi scores and performances should be equal."
            raise ValueError(msg)

        features = data.values
        score_feats = features[:data.shape[0]//2]
        performance_feats = features[data.shape[0]//2:]

        metrics_sp = self.calculate_metrics_one_direction(
            features_1=score_feats,
            features_2=performance_feats,
            postfix=self.postfixes[0]
        )

        metrics_ps = self.calculate_metrics_one_direction(
            features_1=performance_feats,
            features_2=score_feats,
            postfix=self.postfixes[1]
        )

        return metrics_sp + metrics_ps
