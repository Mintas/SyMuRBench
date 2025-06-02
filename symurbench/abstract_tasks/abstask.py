"""Implementation of an Abstract Base Class for tasks."""
import logging
from abc import ABC, abstractmethod

import pandas as pd

from symurbench.feature_extractor import FeatureExtractor, PersistentFeatureExtractor
from symurbench.metaloaders.metaloader import BaseMetaLoader
from symurbench.metrics.metric_value import MetricValue

logger = logging.getLogger("feature extractor")

class AbsTask(ABC):
    """
    Abstract Base Class for tasks.

    This class provides an API interface for implementing tasks.

    Users can create their own subclass and implement
    the following methods:
    - `calculate_metrics` method to calculate metrics for features (and labels).
    - 'run` method to extract features and calculate metrics for them.

    Parameters:
        metadata (BaseMetaLoader): BaseMetaLoader instance with metadata
        name (str): name of the task
        description (str): description of the task
    """
    metadata: BaseMetaLoader
    name: str
    description: str = ""

    def __init__(
        self
    ) -> None:
        """Task initialization. Prepare metadata for feature extraction."""
        self.meta_dataset = self.metadata.load_dataset()

    @abstractmethod
    def calculate_metrics(
        self,
        data: pd.DataFrame
    ) -> list[MetricValue]:
        """
        Calculate metrics for extracted features.

        Args:
            data (pd.DataFrame):
                dataframe with features and labels (or only with features)
                For classification tasks, target columns
                should be predefined in AutoML config.

        Returns:
            list[MetricValue]: List of MetricValue objects (for each calculated metric).
                Each object has an unique name.
        """
        raise NotImplementedError

    def run(
        self,
        feature_extractor: FeatureExtractor | PersistentFeatureExtractor
    ) -> list[MetricValue]:
        """
        Launch task for provided feature extractor.

        Args:
            feature_extractor (FeatureExtractor | PersistentFeatureExtractor):
                FeatureExtractor or PersistentFeatureExtractor object

        Returns:
            list[MetricValue]: List of MetricValue objects (for each calculated metric).
                Each object should have an unique name.

        """
        df = feature_extractor.get_features_with_labels(
            task_name=self.name,
            meta_dataset=self.meta_dataset
        )
        return self.calculate_metrics(df)

    @classmethod
    def pass_args(
        cls,
        metadata_csv_path: str | None = None,
        files_dir_path: str | None = None,
        dataset_filter_list: list[str] | None = None,
        **kwargs
    ) -> None:
        """
        Pass arguments for metadata or for __init__.

        Args:
            metadata_csv_path (str, optional):
                BaseMetaLoader parameter,
                absolute path to CSV file with metadata (filenames, class labels).
                Defaults to None.
            files_dir_path (str, optional):
                BaseMetaLoader parameter,
                absolute path to folder with MIDI files.
                Defaults to None.
            dataset_filter_list (list[str] | None, optional):
                BaseMetaLoader parameter,
                list with files for inclusion filtering (optional).
                Defaults to None.
            **kwargs (dict): dict with __init__ method arguments
        """
        if metadata_csv_path is None:
            metadata_csv_path = cls.metadata.metadata_csv_path
        if files_dir_path is None:
            files_dir_path = cls.metadata.files_dir_path

        cls.metadata = cls.metadata.__class__(
            metadata_csv_path=metadata_csv_path,
            files_dir_path=files_dir_path,
            dataset_filter_list=dataset_filter_list
        )
        return cls(**kwargs)

