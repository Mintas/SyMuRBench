"""Base class for metrics calculation."""
from abc import ABC, abstractmethod

import numpy as np

from .metric_value import MetricValue


class BaseScorer(ABC):
    """Abstract base class for calculating classification metrics.

    This class defines a common interface for classification metric calculators.
    Subclasses must implement the `calculate_metrics` method to provide specific
    metric computation logic.
    """

    @abstractmethod
    def calculate_metrics(
        self,
        y_true: list | np.ndarray,
        preds: list | np.ndarray,
    ) -> list[MetricValue]:
        """Calculate classification metrics for the given predictions.

        Args:
            y_true (list | np.ndarray): True labels or target values.
                Can be integers or floats.
            preds (list | np.ndarray):
                Predicted values, such as class probabilities, logits,
                or predicted class indices. Should be floats or integers.

        Returns:
            list[MetricValue]:
                List of MetricValue objects, each representing a computed metric
                (e.g., accuracy, F1-score). Each object has a unique name.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError
