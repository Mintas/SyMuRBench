"""Class for loading metadata for dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

from .constant import MIDI_PATH_METADATA_COLUMNS, TARGET_COLUMN


class MetaDataset:
    """Container for metadata."""
    def __init__(
        self,
        filenames: list[str],
        files_dir_path: str,
        labels: np.ndarray | None = None
    ) -> None:
        """Initialize the dataset.

        Sets up the dataset with file paths and labels.

        Args:
            filenames (list[str]):
                List of filenames (e.g., MIDI files) included in the dataset.
            files_dir_path (str):
                Absolute path to the directory containing the data files.
            labels (np.ndarray | None, optional):
                Array of target labels corresponding to the files.
                Should be in the same order as `filenames`. Defaults to None.
        """
        self.filenames = filenames
        self.paths_to_files = [Path(files_dir_path, f) for f in filenames]
        if labels is not None:
            self.labels = labels

class BaseMetaLoader:
    """Base class for loading and managing metadata datasets."""
    def __init__(
        self,
        metadata_csv_path: str,
        files_dir_path: str,
        dataset_filter_list: list[str] | None = None,
    ) -> None:
        """Initialize the MetaLoader.

        Args:
            metadata_csv_path (str): Absolute path to the CSV file containing metadata,
                such as filenames and class labels.
            files_dir_path (str):
                Absolute path to the directory containing the data files
                (e.g., MIDI files).
            dataset_filter_list (list[str] | None, optional):
                List of filenames to include (inclusion filter). If provided, only files
                in this list will be used. If None, all files in the metadata
                are included. Defaults to None.
        """
        self.metadata_csv_path = metadata_csv_path
        self.files_dir_path = files_dir_path
        self.dataset_filter_list = dataset_filter_list

    def validate(
        self
    ) -> None:
        """Validate that the metadata and data directory paths exist.

        Raises:
            ValueError: If `self.metadata_csv_path` does not point to an existing file.
            ValueError:
                If `self.files_dir_path` does not point to an existing directory.
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
        """Load metadata from the CSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame:
                DataFrame containing the metadata, such as filenames and labels,
                as read from `metadata_csv_path`.
        """
        return pd.read_csv(self.metadata_csv_path)

    def get_filenames(
        self,
        df: pd.DataFrame
    ) -> list[str]:
        """Load filenames from the metadata DataFrame.

        Extracts filenames from one or more designated columns in the DataFrame.
        Preserves the original order of files both within and across columns to ensure
        consistent alignment with labels and features.

        Args:
            df (pd.DataFrame):
                DataFrame containing metadata, including filename columns.

        Returns:
            list[str]: List of filenames extracted from the specified file columns.
        """
        file_columns = [
            c for c in df.columns if c in MIDI_PATH_METADATA_COLUMNS
        ]
        res = []
        for col in file_columns:
            res += list(df[col].values)
        return res

    def get_labels(
        self,
        df: pd.DataFrame
    ) -> np.ndarray | None:
        """Extract target labels from the metadata DataFrame.

        Retrieves the label column(s) from the DataFrame and returns them
        as a numpy array. If no label columns are present, returns None.

        Args:
            df (pd.DataFrame):
                DataFrame containing metadata.

        Returns:
            np.ndarray | None:
                1D numpy array of labels if labels are present, otherwise None.
        """
        target_columns = [
            c for c in df.columns if c[:len(TARGET_COLUMN)] == TARGET_COLUMN
        ]
        if len(target_columns) > 1:
            return df[target_columns].values
        if len(target_columns) == 1:
            return df[target_columns[0]].values
        return None

    def filter_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter the metadata DataFrame using an inclusion list of filenames.

        Keeps only the rows where the filename is present in the `dataset_filter_list`.
        If no filter list is set, returns the original DataFrame.

        Args:
            df (pd.DataFrame): Metadata DataFrame to filter.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only rows with filenames
                present in the inclusion list.

        Raises:
            ValueError: If `len(self.dataset_filter_list)` is less than or equal to 1.
            ValueError:
                If the resulting DataFrame has one or zero rows (`df.shape[0] <= 1`).
        """
        if len(self.dataset_filter_list) <= 1:
            msg = "You should keep more that 1 file from dataset."
            raise ValueError(msg)

        file_columns = [
            c for c in df.columns if c in MIDI_PATH_METADATA_COLUMNS
        ]
        for col in file_columns:
            df = df[df[col].isin(self.dataset_filter_list)]
        df = df.reset_index(drop=True)

        if df.shape[0] <= 1:
            msg = "You should keep more that 1 file from dataset."
            raise ValueError(msg)

        return df

    def load_dataset(
        self
    ) -> MetaDataset:
        """Load and filter the dataset based on metadata and inclusion criteria.

        Reads the metadata CSV, extracts filenames and optional labels,
        applies filtering based on the provided `dataset_filter_list`,
        and returns a structured MetaDataset object.

        Returns:
            MetaDataset: An object containing the filtered list of filenames,
            absolute file paths, and corresponding labels (if available).
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
