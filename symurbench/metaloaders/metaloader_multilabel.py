"""Metaloader for multilabel classification task."""
import numpy as np
import pandas as pd

from symurbench.constant import MIDI_FILE_COLUMN, TARGET_COLUMN

from .metaloader import BaseMetaLoader


class MultilabelMetaLoader(BaseMetaLoader):
    """BaseMetaLoader subclass for multilabel tasks."""

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
                files from each column should not be
                shuffled either inside the column
                or between the columns.
        """
        return list(df[MIDI_FILE_COLUMN].values)

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
        return df[[c for c in df.columns if f"{TARGET_COLUMN}_" in c]].values

    def filter_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter dataframe using an inclusion list of filenames.

        Args:
            df (pd.DataFrame):
                dataframe with dataset metadata

        Raises:
            ValueError: if len(self.dataset_filter_list) <= 1
            ValueError: df.shape[0] <= 1

        Returns:
            pd.DataFrame: filtered dataframe
        """
        if len(self.dataset_filter_list) <= 1:
            msg = "You should keep more that 1 file from dataset."
            raise ValueError(msg)

        df = df[df[MIDI_FILE_COLUMN].isin(self.dataset_filter_list)]\
            .reset_index(drop=True)

        if df.shape[0] <= 1:
            msg = "You should keep more that 1 file from dataset."
            raise ValueError(msg)

        return df
