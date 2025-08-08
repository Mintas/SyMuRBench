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
    """Class for calculating sklearn classificatoin metrics."""

    def __init__(
        self,
        task_type: str,
        metrics_cfg: dict[str, dict] | None = None,
        treshold: float = 0.5
    ) -> None:
        """
        Initialize the MeanSklearnScorer class.

        Args:
            task_type (str):
                classification task type.
                Variants: `multiclass`, `multilabel`
            metrics_cfg (dict[str, dict], optional):
                dict with metrics to use.
            treshold (float, optional):
                treshold for calculating labels. Defaults to 0.5.
        """
        if task_type not in CLASSIFICATION_TYPES:
            msg = f"{task_type} task type not implemented."
            raise TypeError(msg)

        if not 0 < treshold < 1:
            msg = "Treshold should be more than 0 and less than 1"
            raise ValueError(msg)

        if metrics_cfg is None:
            metrics_cfg = DEFAULT_SKLEARN_SCORER_CONFIG[task_type]

        self.task_type = task_type
        self.metrics_cfg = metrics_cfg
        self.treshold = treshold

    def calc_sklearn_score(
        self,
        metric_func_name: str,
        args: dict,
        y_true: list | np.ndarray,
        preds: list | np.ndarray
    ) -> float:
        """
        Method for calculating sklearn classification metrics.

        Args:
            task_name (str): name of the task
            metric_func_name (str): name of sklearn metrics funciton to use
            args (dict): dict with arguments for metric_func_name
            y_true (list | np.ndarray): list or np.array with true labels
            preds (list | np.ndarray): list or np.array with predicted probabilities

        Raises:
            ValueError: if self.task_type is incorrect.

        Returns:
            float: Calculated score
        """
        metric_func = getattr(sm, metric_func_name)
        if metric_func_name in FUNCTIONS_WITH_PROB_INPUT:
            y_pred = preds
        elif self.task_type == "multiclass":
            y_pred = np.argmax(preds, axis=1)
        elif self.task_type == "multilabel":
            y_pred = np.where(preds > self.treshold, 1, 0)
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
        """
        Calculate metrics for the task.

        Args:
            y_true (list | np.ndarray):
                list or np.array with true values (e.g. class indices).
                Floats or integers.
            preds (list | np.ndarray):
                list or np.array with predicted values. (e.g. probabilities).
                Floats or integers.

        Returns:
            list[MetricValue]: List of MetricValue objects (for each calculated metric).
                Each object should have unique name.
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





