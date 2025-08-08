"""Base class for metrics calculation."""
from abc import ABC, abstractmethod

import numpy as np

from .metric_value import MetricValue


class BaseScorer(ABC):
    """
    Abstract base class for calculating classification metrics.

    Subclasses must implement the `calculate_metrics` method.
    """

    @abstractmethod
    def calculate_metrics(
        self,
        y_true: list | np.ndarray,
        preds: list | np.ndarray,
    ) -> list[MetricValue]:
        """
        Calculate metrics for the task.

        Args:
            y_true (list | np.ndarray):
                list or np.array with true values (e.g. class indices).
                Floats or integers.
            preds (list | np.ndarray):
                list or np.array with predicted values. (e.g. probabilities).
                Floats or integers.

        Raises:
            NotImplementedError

        Returns:
            list[MetricValue]: List of MetricValue objects (for each calculated metric).
                Each object should have unique name.
        """
        raise NotImplementedError
