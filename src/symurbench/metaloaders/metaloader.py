"""Class for loading metadata for dataset."""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class MetaDataset:
    """Container for metadata."""
    def __init__(
        self,
        filenames: list[str],
        files_dir_path: str,
        labels: np.ndarray | None = None
    ) -> None:
        """
        Initialize dataset.

        Args:
            filenames (list[str]): list of filenames. (e.g. MIDI files)
            files_dir_path (str): absolute path to folder with MIDI files
            labels (np.ndarray | None, optional):
                np.array with target labels. Defaults to None.
        """
        self.filenames = filenames
        self.paths_to_files = [Path(files_dir_path, f) for f in filenames]
        if labels is not None:
            self.labels = labels

class BaseMetaLoader(ABC):
    """
    Abstract base class for metadata dataset loading.

    Parameters:
        metadata_csv_path (str):
            absolute path to CSV file with metadata (filenames, class labels)
        files_dir_path (str):
            absolute path to folder with MIDI files
        dataset_filter_list (list[str] | None, optional):
            list with files for inclusion filtering (optional).
            Defaults to None.
    """
    def __init__(
        self,
        metadata_csv_path: str,
        files_dir_path: str,
        dataset_filter_list: list[str] | None = None,
    ) -> None:
        """
        Metaloader initialization.

        Args:
            metadata_csv_path (str):
                absolute path to CSV file with metadata (filenames, class labels)
            files_dir_path (str):
                absolute path to folder with MIDI files
            dataset_filter_list (list[str] | None, optional):
                list with files for inclusion filtering (optional).
                Defaults to None.
        """
        self.metadata_csv_path = metadata_csv_path
        self.files_dir_path = files_dir_path
        self.dataset_filter_list = dataset_filter_list

    def validate(
        self
    ) -> None:
        """
        Validate paths: metadata_csv_path and files_dir_path.

        Raises:
            ValueError: if self.metadata_csv_path does not exists
            ValueError: if self.files_dir_path does not exists
        """
        if not Path.exists(Path(self.metadata_csv_path)):
            msg = "Provided path to CSV file does not exist"
            ValueError(msg)
        if not Path.exists(Path(self.files_dir_path)):
            msg = "Provided path to folder with MIDI files does not exist"
            ValueError(msg)

    def get_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Load dataframe from metadata_csv_path.

        Returns:
            pd.DataFrame: pd.DataFrame with metadata
        """
        return pd.read_csv(self.metadata_csv_path)

    @abstractmethod
    def get_filenames(
        self,
        df: pd.DataFrame
    ) -> list[str]:
        """
        Load filenames from pd.DataFrame with metadata.

        Can load files from multiple columns if needed.

        Args:
            df (pd.DataFrame): pd.DataFrame with metadata

        Returns:
            list[str]: filelist from file_columns.
                The order of the files is important:
                files from each column should not be shuffled
                either inside the column or between the columns.
        """
        raise NotImplementedError

    @abstractmethod
    def get_labels(
        self,
        df: pd.DataFrame
    ) -> np.ndarray | None:
        """
        Return np.array with target labels from DataFrame.

        If there are no labels in DataFrame, return None.

        Args:
            df (pd.DataFrame): DataFrame with metadata

        Returns:
            np.ndarray: target labels or None
        """
        raise NotImplementedError

    @abstractmethod
    def filter_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter dataframe using an inclusion list of filenames.

        Args:
            df (pd.DataFrame):
                dataframe with dataset metadata
            file_columns (list[str], optional):
                df column names with file names to use in the task.

        Returns:
            pd.DataFrame: filtered dataframe
        """
        raise NotImplementedError

    def load_dataset(
        self
    ) -> MetaDataset:
        """
        Load and filter dataset.

        Args:
            file_columns (list[str], optional):
                df column names with file names to use in the task.

        Returns:
            MetaDataset: MetaDataset object with loaded filtered dataset for the task.
        """
        self.validate()
        metadata_df = self.get_dataframe()
        if self.dataset_filter_list is not None:
            metadata_df = self.filter_dataframe(metadata_df)

        filenames = self.get_filenames(metadata_df)
        labels = self.get_labels(metadata_df)

        return MetaDataset(
            filenames=filenames,
            files_dir_path=self.files_dir_path,
            labels=labels)
