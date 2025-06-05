"""Implementation of a Scorer for retrieval tasks."""
import numpy as np

from symurbench.constant import DEFAULT_RETRIEVAL_RANKS
from symurbench.retrieval import compute_metrics

from .metric_value import MetricValue
from .scorer import BaseScorer


class RetrievalScorer(BaseScorer):
    """
    Class for calculating retrieval metrics.

    User can configure tuple of ranks
    to define retrieval metrics to use.
    """

    def __init__(
        self,
        ranks: tuple[int] | None = DEFAULT_RETRIEVAL_RANKS
    ) -> None:
        """
        Initialize the Scorer class.

        Args:
            ranks (set[int], optional):
                a set of ranks to calculate for retrieval task.
                Defaults to DEFAULT_RETRIEVAL_RANKS.

        Raises:
            ValueError: if min(ranks) <= 0 or max(ranks) >= 100
        """
        self.ranks = ranks
        self.validate_ranks()

    def validate_ranks(
        self
    ) -> None:
        """
        Validate self.ranks.

        Raises:
            ValueError: if min(self.ranks) <= 0 or max(self.ranks) >= 100
        """
        if self.ranks is None:
            self.ranks = []
        if not isinstance(self.ranks, tuple)\
        or {isinstance(r, int) for r in self.ranks} != {True}:
            msg = "ranks argument should be a tuple of integers"
            raise ValueError(msg)
        if len(self.ranks) > 0 and (max(self.ranks) >= 100 or min(self.ranks) <= 0):
            msg = "Retrieval ranks should be integers between 1 and 100"
            raise ValueError(msg)

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        preds: np.ndarray,
    ) -> list[MetricValue]:
        """
        Calculate metrics for the task.

        Args:
            y_true (list | np.ndarray):
                np.array with indices of vectors in the initial order.
            preds (list | np.ndarray):
                np.array with indices of sorted objects.

        Returns:
            list[MetricValue]: List of MetricValue objects (for each calculated metric).
                Each object should have unique name.
        """
        metrics_dict = compute_metrics(
            gt_indices=y_true,
            retrieved_indices=preds,
            ranks=self.ranks)

        return [
            MetricValue(
                name=k,
                values=[v]
            ) for k,v in metrics_dict.items()
        ]
