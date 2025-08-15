"""Implementation of a Scorer for retrieval tasks."""
import numpy as np

from symurbench.constant import DEFAULT_RETRIEVAL_RANKS
from symurbench.retrieval import compute_metrics

from .metric_value import MetricValue
from .scorer import BaseScorer


class RetrievalScorer(BaseScorer):
    """Class for calculating retrieval metrics.

    This class allows users to configure a tuple of ranks
    to specify which retrieval metrics to compute, such as R@k.
    """
    def __init__(
        self,
        ranks: tuple[int] | None = DEFAULT_RETRIEVAL_RANKS
    ) -> None:
        """Initialize the Scorer class.

        Args:
            ranks (set[int], optional):
                Set of rank cutoffs at which to compute retrieval metrics.
                Each rank corresponds to a top-k cutoff.
                Defaults to DEFAULT_RETRIEVAL_RANKS.

        Raises:
            ValueError:
                if min(ranks) <= 0 or max(ranks) >= 100
        """
        self.ranks = ranks
        self.validate_ranks()

    def validate_ranks(
        self
    ) -> None:
        """Validate the ranks used for metric calculation.

        Raises:
            ValueError: If min(ranks) <= 0 or max(ranks) >= 100
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
        """Calculate retrieval metrics for the given predictions.

        Args:
            y_true (list | np.ndarray):
                Array of ground truth indices representing the correct order
                or relevant items.
            preds (list | np.ndarray):
                Array of predicted indices representing the ranked items.

        Returns:
            list[MetricValue]:
                List of MetricValue objects.
                Each object has a unique name corresponding to the metric
                and retrieval direction (e.g., 'R@5_sp').
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
