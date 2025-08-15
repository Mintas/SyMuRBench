"""Implementation of an Abstract Base Class for tasks."""
import logging
from abc import ABC, abstractmethod

import pandas as pd

from symurbench.feature_extractor import FeatureExtractor, PersistentFeatureExtractor
from symurbench.metaloader import BaseMetaLoader
from symurbench.metrics.metric_value import MetricValue

logger = logging.getLogger(__name__)

class AbsTask(ABC):
    """
    Abstract base class for benchmark tasks.

    This class defines the interface for implementing evaluation tasks.
    Subclasses must implement the core methods to extract features and compute metrics.

    To create a custom task, users should:
        - Implement `calculate_metrics` to compute evaluation metrics
            from features and labels.
        - Implement `run` to execute feature extraction and metric computation.

    Args:
        name (str): Name of the task.
        description (str): Description of the task.
        metaloader (BaseMetaLoader):
            The metadata loader class (not instance) to be instantiated
            for loading dataset metadata and file paths.
    """
    name: str
    description: str = ""
    metaloader: BaseMetaLoader = BaseMetaLoader

    def __init__(
        self,
        metaloader_args_dict: dict
    ) -> None:
        """
        Initialize the task and prepare metadata for feature extraction.

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
        """
        self.metaloader = self.metaloader(**metaloader_args_dict)
        self.meta_dataset = self.metaloader.load_dataset()

    @abstractmethod
    def calculate_metrics(
        self,
        data: pd.DataFrame
    ) -> list[MetricValue]:
        """
        Calculate metrics for the extracted features.

        Args:
            data (pd.DataFrame):
                Input dataframe containing extracted features,
                and optionally labels. For classification tasks,
                the target (label) columns must be specified
                in the AutoML configuration.

        Returns:
            list[MetricValue]:
                A list of `MetricValue` objects, one for each computed metric.
                Each object contains a unique metric name
                and the corresponding value(s).
        """
        raise NotImplementedError

    def run(
        self,
        feature_extractor: FeatureExtractor | PersistentFeatureExtractor
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
        return self.calculate_metrics(df)

    @classmethod
    def pass_args(
        cls,
        metadata_csv_path: str,
        files_dir_path: str,
        dataset_filter_list: list[str] | None = None,
        **kwargs
    ) -> None:
        """
        Pass arguments for metadata loading or for subclass initialization.

        Args:
            metadata_csv_path (str): Absolute path to the CSV file containing metadata
                (e.g., filenames, class labels). Used by BaseMetaLoader.
            files_dir_path (str): Absolute path to the directory containing MIDI files.
                Used by BaseMetaLoader.
            dataset_filter_list (list[str] | None, optional):
                List of filenames to include (applies inclusion filtering).
                If None, no filtering is applied. Defaults to None.
            **kwargs (dict):
                Additional keyword arguments passed to the `__init__` method
                of the subclass. These may include parameters specific
                to the subclass's behavior.
        """
        metaloader_args_dict = {
            "metadata_csv_path": metadata_csv_path,
            "files_dir_path": files_dir_path,
            "dataset_filter_list": dataset_filter_list
        }
        return cls(metaloader_args_dict, **kwargs)

