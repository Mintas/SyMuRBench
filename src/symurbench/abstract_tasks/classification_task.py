"""Base class for classification tasks."""
import pandas as pd
from lightautoml.automl.base import AutoML
from lightautoml.ml_algo.linear_sklearn import LinearLBFGS
from lightautoml.pipelines.features.linear_pipeline import LinearFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task

from symurbench.constant import SEED
from symurbench.feature_extractor import FeatureExtractor, PersistentFeatureExtractor
from symurbench.metrics.metric_value import MetricValue
from symurbench.metrics.scorer import BaseScorer
from symurbench.utils import load_yaml

from .abstask import AbsTask


class ClassificationTask(AbsTask):
    """A class for evaluating FeatureExtractor on a classification task."""

    def __init__(
        self,
        automl_config_path: str, # relative path to YAML file with AutoML config
        scorer: BaseScorer
    ) -> None:
        """Task initialization. Prepare dataset for feature extraction.

        Args:
            automl_config_path (str): path to YAML file with AutoML config
            scorer (BaseScorer): scorer to use for metrics calculation.
        """
        super().__init__()
        self.automl_config_path = automl_config_path
        self.scorer = scorer

    def init_automl(
        self,
        preprocess_features: bool,
    ) -> tuple[AutoML, dict, int]:
        """
        Initialize automl model.

        Args:
            preprocess_features (bool): whether to preprocess features.
                If True, preprocess features according to automl config.
                If False, use features as they are.

        Raises:
            ValueError: if config at self.automl_config_path is None

        Returns:
            tuple[AutoML, dict, int]:
                AutoML object, dict with roles, verbose parameter
        """
        config = load_yaml(self.automl_config_path)

        if config is None:
            msg = f"You should provide automl config for {self.name} task."
            raise ValueError(msg)

        task = Task(**config["task"])
        reader = PandasToPandasReader(task, cv=config["n_folds"], random_state=SEED)

        if preprocess_features:
            pipe = LinearFeatures(**config["linearFeatures_params"])
        else:
            pipe = None

        model = LinearLBFGS(
            default_params=config["linearLBFGS_params"]
        )
        pipeline_lvl = MLPipeline([model], features_pipeline=pipe)
        roles = config["roles"]
        automl = AutoML(reader, [[pipeline_lvl]])

        return {
            "automl": automl,
            "roles": roles,
            "verbose": config["verbose"]
        }

    def calculate_metrics(
        self,
        data: pd.DataFrame,
        preprocess_features: bool,
    ) -> list[MetricValue]:
        """
        Run AutoML and calculate metrics for predictions.

        Args:
            data (pd.DataFrame):
                dataframe with features and labels (or only with features)
            preprocess_features (bool): whether to preprocess features.
                If True, preprocess features according to automl config.
                If False, use features as they are.

        Returns:
            list[MetricValue]: List of MetricValue objects (for each calculated metric).
                Each object has an unique name.
        """
        automl = self.init_automl(preprocess_features)
        predictions = automl["automl"].fit_predict(
            data,
            roles=automl["roles"],
            verbose=automl["verbose"]
        )
        folds_indexes = predictions.__dict__["folds"]
        n_folds = automl["automl"].reader.cv

        metrics_list = []
        for fold in range(n_folds):
            pred_probas = predictions.data[folds_indexes == fold]
            y_true = data[automl["roles"]["target"]].values[folds_indexes == fold]
            metrics = self.scorer.calculate_metrics(
                y_true=y_true,
                preds=pred_probas
            )
            if fold == 0:
                metrics_list = metrics
            else:
                for i in range(len(metrics_list)):
                    metrics_list[i] += metrics[i]

        return metrics_list

    def run(
        self,
        feature_extractor: FeatureExtractor | PersistentFeatureExtractor,
    ) -> list[MetricValue]:
        """
        Launch classification task for provided feature extractor.

        Args:
            feature_extractor (FeatureExtractor | PersistentFeatureExtractor):
                FeatureExtractor or PersistentFeatureExtractor object

        Returns:
            list[MetricValue]: List of MetricValue objects (for each calculated metric).
                Each object should have unique name.

        """
        df = feature_extractor.get_features_with_labels(
            task_name=self.name,
            meta_dataset=self.meta_dataset
        )
        return self.calculate_metrics(df, feature_extractor.preprocess_features)
