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
    """Evaluator class for assessing the performance on a classification task.

    This class facilitates training and evaluating a classifier using features
    extracted by a given `FeatureExtractor` or `PersistentFeatureExtractor`.
    It handles feature preprocessing, model training, and metric computation.
    """

    def __init__(
        self,
        metaloader_args_dict: dict,
        automl_config_path: str, # relative path to YAML file with AutoML config
        scorer: BaseScorer
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
            automl_config_path (str):
                Path to the YAML configuration file for AutoML.
            scorer (BaseScorer):
                The scorer instance used to compute evaluation metrics.
        """
        super().__init__(metaloader_args_dict)
        self.automl_config_path = automl_config_path
        self.scorer = scorer

    def init_automl(
        self,
        preprocess_features: bool,
    ) -> tuple[AutoML, dict, int]:
        """
        Initialize the AutoML model.

        Args:
            preprocess_features (bool): Whether to preprocess the input features.
                If True, features are preprocessed according to the AutoML
                configuration. If False, features are used as-is without preprocessing.

        Returns:
            tuple[AutoML, dict, int]: A tuple containing:
                - The initialized AutoML model object.
                - A dictionary specifying data roles (e.g., 'target', 'features').
                - The verbosity level (int) for logging control.

        Raises:
            ValueError: If the configuration at `self.automl_config_path` is None.
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
                Input dataframe containing features and labels.
            preprocess_features (bool): Whether to preprocess the features.
                If True, applies preprocessing as defined in the AutoML configuration.
                If False, uses the features in their raw form.

        Returns:
            list[MetricValue]:
                A list of MetricValue objects, one for each calculated metric.
                Each object has a unique name and the corresponding metric value(s).
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
        Run the task using the provided feature extractor.

        Args:
            feature_extractor (FeatureExtractor | PersistentFeatureExtractor):
                The feature extractor instance used to extract
                features from the dataset. Can be a regular `FeatureExtractor`
                or a `PersistentFeatureExtractor` that caches results to disk.

        Returns:
            list[MetricValue]:
                A list of `MetricValue` objects, one for each computed metric.
                Each object has a unique name and the corresponding metric value(s).
        """
        df = feature_extractor.get_features_with_labels(
            task_name=self.name,
            meta_dataset=self.meta_dataset
        )
        return self.calculate_metrics(df, feature_extractor.preprocess_features)
