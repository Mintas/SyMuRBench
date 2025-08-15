"""Implementation of a Scorer for classification tasks."""
import numpy as np
import sklearn.metrics as sm

from symurbench.constant import DEFAULT_SKLEARN_SCORER_CONFIG

from .metric_value import MetricValue
from .scorer import BaseScorer

# classification functions with probabilities as input
FUNCTIONS_WITH_PROB_INPUT = [
    "roc_auc_score",
    "brier_score_loss",
    "average_precision_score",
    "d2_log_loss_score",
    "det_curve",
    "log_loss",
    "top_k_accuracy_score"
]

CLASSIFICATION_TYPES = [
    "multiclass",
    "multilabel"
]

class SklearnClsScorer(BaseScorer):
    """Class for calculating classification metrics using scikit-learn.

    This class provides an implementation of common classification metrics
    (e.g., accuracy, F1-score) by leveraging the scikit-learn metrics module.
    It computes a standard set of performance measures based on true labels
    and predicted values.
    """
    def __init__(
        self,
        task_type: str,
        metrics_cfg: dict[str, dict] | None = None,
        threshold: float = 0.5
    ) -> None:
        """Initialize the MeanSklearnScorer class.

        Args:
            task_type (str): Type of classification task. Supported values are:
                - 'multiclass': for multi-class classification
                - 'multilabel': for multi-label classification
            metrics_cfg (dict[str, dict], optional): Configuration dictionary specifying
                which metrics to compute and their parameters. Defaults to None.
            threshold (float, optional):
                Threshold for converting predicted probabilities
                to multi-label predictions. Defaults to 0.5.

        Raises:
            ValueError: If `task_type` is not one of ['multiclass', 'multilabel'].
            ValueError: If `threshold` is not between 0 and 1.
        """
        if task_type not in CLASSIFICATION_TYPES:
            msg = f"{task_type} task type not implemented."
            raise ValueError(msg)

        if not 0 < threshold < 1:
            msg = "Treshold should be more than 0 and less than 1"
            raise ValueError(msg)

        if metrics_cfg is None:
            metrics_cfg = DEFAULT_SKLEARN_SCORER_CONFIG[task_type]

        self.task_type = task_type
        self.metrics_cfg = metrics_cfg
        self.threshold = threshold

    def calc_sklearn_score(
        self,
        metric_func_name: str,
        args: dict,
        y_true: list | np.ndarray,
        preds: list | np.ndarray
    ) -> float:
        """Calculate a scikit-learn classification metric.

        Args:
            task_name (str): Name of the classification task.
            metric_func_name (str): Name of the scikit-learn metrics function to use
                (e.g., 'f1_score').
            args (dict):
                Dictionary of additional arguments to pass to the metric function.
            y_true (list | np.ndarray): True labels.
            preds (list | np.ndarray): Predicted probabilities or class labels.

        Returns:
            float: The computed metric score.

        Raises:
            ValueError: If `self.task_type` is not supported
                (i.e., not 'multiclass' or 'multilabel').
        """
        metric_func = getattr(sm, metric_func_name)
        if metric_func_name in FUNCTIONS_WITH_PROB_INPUT:
            y_pred = preds
        elif self.task_type == "multiclass":
            y_pred = np.argmax(preds, axis=1)
        elif self.task_type == "multilabel":
            y_pred = np.where(preds > self.threshold, 1, 0)
        else:
            msg = f"Incorrect task type: {self.task_type}."
            raise TypeError(msg)

        if args is not None and len(args)>0:
            score = metric_func(y_true, y_pred, **args)
        else:
            score = metric_func(y_true, y_pred)

        return MetricValue(
            name=metric_func_name,
            values=[score]
        )

    def calculate_metrics(
        self,
        y_true: list | np.ndarray,
        preds: list | np.ndarray,
    ) -> list[MetricValue]:
        """Calculate classification metrics for the given task.

        Args:
            y_true (list | np.ndarray):
                True labels or target values. Can be integers or floats.
                Must have the same length as `preds`.
            preds (list | np.ndarray):
                Predicted values, such as class probabilities, logits,
                or predicted class indices. Should be floats or integers.
                Must have the same length as `y_true`.

        Returns:
            list[MetricValue]:
                List of MetricValue objects, each representing a computed metric
                (e.g., accuracy, F1-score). Each object has a unique name corresponding
                to the metric (e.g., 'accuracy', 'f1_score').

        Raises:
            ValueError: If `y_true` and `preds` have mismatched lengths.
        """
        if len(y_true) != len(preds):
            msg = "y_true and preds array lengths should be equal"
            raise ValueError(msg)

        metrics_list = []
        for name, args in self.metrics_cfg.items():
            metrics_list += [
                self.calc_sklearn_score(
                    metric_func_name=name,
                    args=args,
                    y_true=y_true,
                    preds=preds)
            ]

        return metrics_list





