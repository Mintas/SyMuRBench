"""Abstract Base Class for feature extraction implemetation."""
import logging
from abc import ABC, abstractmethod
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .constant import MIDI_FILE_COLUMN, NUM_THREADS, TASK_NAME_COLUMN
from .metaloader import MetaDataset
from .utils import embs_and_labels_to_df, validate_file_paths

logger = logging.getLogger(__name__)

class FeatureExtractor(ABC):
    """Abstract base class for feature extraction from a list of files.

    This class defines a common interface for implementing feature extractors that
    convert raw files into vector representations (embeddings) for use in downstream
    benchmarking tasks.

    The primary method `extract_features_from_file` must be implemented by subclasses
    to define how features are extracted from a single file. Optionally, users can
    override `extract_features_from_files` for improved performance when processing
    multiple files in batch.

    Users should subclass this class and implement the required methods to integrate
    custom feature extraction logic into the benchmarking pipeline.
    """
    def __init__(
        self,
        extractor_name: str,
        fast: bool = False,
        preprocess_features: bool = True,
    ) -> None:
        """Initialize the FeatureExtractor class.

        Args:
            extractor_name (str):
                Name of the feature extractor. Must be unique to identify
                the extractor in benchmark results.
            fast (bool, optional):
                If True, enables multiprocessing to extract features
                from files in parallel for improved performance. Defaults to False.
            preprocess_features (bool, optional): If True, features will be preprocessed
                according to the AutoML configuration before being used in tasks.
                If False, raw extracted features are used without modification.
                Defaults to True.
        """
        self.name = extractor_name
        self.fast = fast
        self.preprocess_features = preprocess_features

    @abstractmethod
    def extract_features_from_file(
        self,
        file_path: str
    ) -> np.ndarray:
        """Extract features from a single file.

        Args:
            file_path (str): Absolute path to the input file
                (e.g., MIDI or other symbolic music format).

        Returns:
            np.ndarray: 1D numpy array of shape (n_features,)
                containing the extracted feature vector.
        """

    def _extract_features_from_files(
        self,
        file_paths: list[str]
    ) -> np.ndarray:
        """Extract features from a list of files.

        This method processes multiple files and returns a feature matrix
        where each row corresponds to a file and each column corresponds to a feature.
        By default, it processes files sequentially, but can be overridden
        to support batch processing or optimized extraction.

        Args:
            file_paths (list[str]):
                List of absolute paths to input files
                (e.g., MIDI or other symbolic music formats).
            fast (bool, optional):
                If True, uses multiprocessing to extract features in parallel.
                Defaults to False.

        Returns:
            np.ndarray: 2D numpy array of shape (n_files, n_features)
                containing the extracted feature matrix.
        """
        if self.fast:
            with Pool(processes=NUM_THREADS) as pool:
                vectors = list(tqdm(
                    pool.imap(
                    self.extract_features_from_file,
                    file_paths),
                    total=len(file_paths),
                    desc="Features extraction")
                )
        else:
            vectors = [
                self.extract_features_from_file(file)
                for file in tqdm(file_paths, desc="Features extraction")
            ]
        return np.vstack([v.reshape(1,-1) for v in vectors])

    def extract_features_from_files(
        self,
        file_paths: list[str]
    ) -> pd.DataFrame:
        """Extract features from a list of files.

        The order of the output features must match the order
        of the input file paths exactly.

        Args:
            file_paths (list[str]): List of absolute paths to input files
                (e.g., MIDI or other symbolic music formats).

        Returns:
            pd.DataFrame:
                DataFrame of shape (len(file_paths), n_features) containing
                the extracted features. The DataFrame does not include column headers
                or an index, with each row corresponding to a file.

        Raises:
            ValueError: If the extracted features matrix has fewer than 2 dimensions.
            ValueError: If the number of extracted feature rows does not match
            the number of input file paths.
        """
        validate_file_paths(file_paths)
        features = self._extract_features_from_files(file_paths)
        if len(features.shape) != 2:
            msg = "Features should contain at least 2 vectors."
            raise ValueError(msg)
        if features.shape[0] != len(file_paths):
            msg = "All files should be mapped to feature vectors"
            raise ValueError(msg)

        return pd.DataFrame(features)

    def get_features_with_labels(
        self,
        task_name: str,
        meta_dataset: MetaDataset
    ) -> pd.DataFrame:
        """Create a DataFrame containing features and labels for a specific task.

        This method extracts features for all files associated with the given task and
        combines them with their corresponding labels from the metadata.

        Args:
            task_name (str): Name of the task
                (must correspond to a task defined in the tasks folder).
            meta_dataset (MetaDataset): MetaDataset object containing:
                - Absolute paths to .mid files (or other supported formats)
                - Filenames
                - Optional labels (if available)

        Returns:
            pd.DataFrame: DataFrame containing the extracted features
                and corresponding labels.
                Shape: (n_files, n_features + n_label_columns).
                Does not include index or headers.
        """
        features = self.extract_features_from_files(meta_dataset.paths_to_files)
        if hasattr(meta_dataset, "labels"):
            df = embs_and_labels_to_df(features, meta_dataset.labels)
        else:
            df = pd.DataFrame(features)
        log_msg = f"Features shape for {task_name} task is {features.shape}"
        logger.info(log_msg)

        return df

class PersistentFeatureExtractor:
    """A class that extends FeatureExtractor to support feature persistence.

    This class provides functionality for saving and loading extracted features to and
    from the file system using Parquet files, enabling caching for faster
    subsequent runs. It falls back to regular in-memory extraction when persistence
    is not applicable.

    Fallback to FeatureExtractor behavior occurs in the following cases:
    - No `persistence_path` is provided
    - `use_cached` is set to False
    - `use_cached` is True, but no cached features are found
    at the specified `persistence_path`
    """
    def __init__(
        self,
        feature_extractor: FeatureExtractor = None,
        persistence_path: str = "",
        use_cached: bool = False,
        overwrite_features: bool = False,
        name: str = "feature extractor",
        preprocess_features: bool = True,
    ) -> None:
        """
        Initialize the PersistentFeatureExtractor class.

        Args:
            feature_extractor (FeatureExtractor, optional):
                FeatureExtractor to be used in fallback scenarios.
                Defaults to None.
            persistence_path (str, optional):
                path to parquet file where to write/load from features.
                Defaults to "".
            use_cached (bool, optional):
                flag responsible for loading/writing mode
                (if True, then load features from parquet,
                else write features to parquet).
                Defaults to False.
            overwrite_features (bool, optional):
                flag responsible for overwriting features in parquet file
                if file exists already. Defaults to False.
            name (str, optional):
                name that will be used as name of feature_extractor
                in benchmark if feature_extractor is None.
                Should be unique. Defaults to "feature extractor".
            preprocess_features (bool, optional):
                whether to preprocess features for tasks with AutoML.
                If feature_extractor is not None, preprocess_features
                is set to preprocess_features flag of feature_extractor.
                Ottherwise, defaults to True.

        Raises:
            ValueError: if neither feature_extractor nor persistence_path are passed
        """
        """Initialize the PersistentFeatureExtractor class.

        Args:
            feature_extractor (FeatureExtractor, optional):
                Underlying feature extractor to use in fallback scenarios.
                Defaults to None.
            persistence_path (str, optional): Path to the Parquet file for saving
                or loading cached features. If empty or not provided,
                caching is disabled. Defaults to "".
            use_cached (bool, optional):
                If True, attempt to load features from the persistence path.
                If False, extract features and save them to the persistence path
                (if specified). Defaults to False.
            overwrite_features (bool, optional):
                If True and `use_cached` is False, existing features in the Parquet file
                will be overwritten. If False, features will be merged or skipped.
                Defaults to False.
            name (str, optional):
                Name of the feature extractor, used in benchmark results.
                Must be unique. Used only if `feature_extractor` is None.
                Defaults to "feature extractor".
            preprocess_features (bool, optional):
                Whether to preprocess features for use in AutoML tasks.
                If `feature_extractor` is provided, this is set to the extractor's
                `preprocess_features` value. Otherwise, defaults to True.

        Raises:
            ValueError: If neither `feature_extractor` nor `persistence_path`
                are provided.
        """
        if feature_extractor is None and persistence_path == "":
            msg="You should provide feature_extractor or persistence_path."
            raise ValueError(msg)

        self.feature_extractor = feature_extractor
        self.name = feature_extractor.name if feature_extractor is not None else name

        log_msg = f"Name {self.name} is used for PersistentFeatureExtractor object"
        logger.info(log_msg)

        self.persistence_path = persistence_path
        self.use_cached = use_cached
        self.overwrite_features = overwrite_features

        if self.feature_extractor is not None:
            self.preprocess_features = self.feature_extractor.preprocess_features
        else:
            self.preprocess_features = preprocess_features

    def filter_features(
        self,
        df: pd.DataFrame,
        files_to_keep: list[str]
    ) -> pd.DataFrame:
        """Filter features and ensure their order matches the inclusion list.

        Filters the DataFrame to include only rows corresponding to the specified files,
        and sorts them in the same order as the `files_to_keep` list.

        Args:
            df (pd.DataFrame): DataFrame containing extracted features for the task.
            task_name (str): Name of the task associated with the features.
            files_to_keep (list[str]): List of filenames to retain in the DataFrame.
                The resulting DataFrame will be ordered according to this list.

        Returns:
            pd.DataFrame: Filtered and ordered DataFrame with shape
                (len(files_to_keep), n_features).

        Raises:
            ValueError: If `len(files_to_keep)` is less than or equal to 1.
            ValueError: If the number of rows in the filtered DataFrame is less than
                `len(files_to_keep)`, indicating some files were not found
                in the original DataFrame.
        """
        if len(files_to_keep) <= 1:
            msg = "You should keep more that 1 file from dataset."
            raise ValueError(msg)

        filtered_df = df[df[MIDI_FILE_COLUMN].isin(files_to_keep)]\
            .reset_index(drop=True)

        if filtered_df.shape[0] < len(files_to_keep):
            msg = "Not all files present in the dataframe."
            raise ValueError(msg)

        if list(filtered_df[MIDI_FILE_COLUMN].values) != files_to_keep:
            sorting_df = pd.DataFrame(files_to_keep)
            sorting_df.columns = [MIDI_FILE_COLUMN]
            filtered_df = sorting_df.merge(filtered_df, on=MIDI_FILE_COLUMN)

        return filtered_df

    def validate_writing(
        self,
        df: pd.DataFrame,
        existing_df: pd.DataFrame,
        task_name: str
    ) -> None:
        """Validate the existing DataFrame before writing new features.

        Performs checks to ensure compatibility and proper handling when saving features
        to an existing DataFrame.

        Args:
            df (pd.DataFrame): New DataFrame with features to be saved.
            existing_df (pd.DataFrame): Previously saved DataFrame with features.
            task_name (str):
                Name of the current task for which features are being written.

        Raises:
            ValueError: If `existing_df` does not contain required columns
                'task' or 'midi_file'.
            ValueError: If `existing_df` already contains features for the current
                `task_name` and `self.overwrite_features` is False.
            ValueError: If the number of feature columns in `existing_df` differs
                from `df` (i.e., `existing_df.shape[1] != df.shape[1]`),
                indicating a feature dimension mismatch.
        """
        if TASK_NAME_COLUMN not in existing_df.columns\
        and MIDI_FILE_COLUMN not in existing_df.columns:
            msg = "Features can be appended to existing parquet file only when"\
                f" it contains '{TASK_NAME_COLUMN}'"\
                f" and '{MIDI_FILE_COLUMN}' columns."  # noqa: ISC002
            raise ValueError(msg)

        if task_name in existing_df[TASK_NAME_COLUMN].values\
        and not self.overwrite_features:
            msg = f"Parquet file already contains features for {task_name} task."\
                "If you want to overwrite them, pass overwrite_features=True"\
                " in __init__ function or set attribute overwrite_features to True." # noqa: ISC002
            raise ValueError(msg)
        if existing_df.shape[1] != df.shape[1]:
            msg = "Existing parquet file should contain features"\
                " of same shape as new features."  # noqa: ISC002
            raise ValueError

    def write_features_to_pqt(
        self,
        pqt_path: str,
        task_name: str,
        filelist: list[str],
        features: pd.DataFrame
    ) -> None:
        """Write calculated features to a Parquet file.

        Saves the extracted features along with metadata (task name and filenames) into
        a Parquet file for persistent storage and later reuse. The resulting file
        includes columns for file names, task names, and feature values.

        Args:
            pqt_path (str): Path to the Parquet file to write. The file will contain:
                - `midi_file`: Filename corresponding to each feature vector (str)
                - `task`: Task name associated with the features (str)
                - Feature columns (e.g., `E_0`, `E_1`, ...):
                Numerical feature values (int or float)
            task_name (str): Name of the task for which features are extracted.
            filelist (list[str]):
                List of filenames corresponding to the rows in `features`.
                Must match the number of rows in `features`.
            features (pd.DataFrame): DataFrame of shape (n_files, n_features)
                containing the extracted features.
                Column names are automatically assigned as `E_0`, `E_1`, etc.

        Example:
            `| midi_file | task     | E_0 | E_1 | E_2 |`
            `|-----------|----------|-----|-----|-----|`
            `| file1.mid | genre    | 0   | 0.1 | 1.1 |`
            `| file2.mid | genre    | 1   | 3.8 | 4.0 |`
            `| file3.mid | composer | 0   | 1.1 | 4.8 |`
            `| file4.mid | composer | 0   | 4.5 | 7.6 |`
        """
        log_msg = f"Writing features to {pqt_path}."
        logger.info(log_msg)
        df = features.copy()
        columns = [f"E_{i}" for i in range(features.shape[1])]
        df.columns = columns
        df[TASK_NAME_COLUMN] = task_name
        df[MIDI_FILE_COLUMN] = pd.Series(filelist)
        df = df[[MIDI_FILE_COLUMN, TASK_NAME_COLUMN, *columns]] # reorder columns
        if Path.exists(Path(pqt_path)):
            log_msg = "Parquet file already exists. Appending features to previous data"
            logger.info(log_msg)
            df_prev = pd.read_parquet(pqt_path)

            self.validate_writing(
                df=df,
                existing_df=df_prev,
                task_name=task_name)

            if task_name in df_prev[TASK_NAME_COLUMN].values\
            and self.overwrite_features:
                log_warn = f"Overwriting features for {task_name} task."
                logging.warning(log_warn)
                df_prev = df_prev[df_prev[TASK_NAME_COLUMN]!=task_name]\
                    .reset_index(drop=True)

            df = pd.concat([df_prev, df], axis=0).reset_index(drop=True)
        df.to_parquet(pqt_path, index=False)

    def load_features_from_pqt(
        self,
        pqt_path: str,
        task_name: str,
        filelist: list[str]
    ) -> pd.DataFrame:
        """Load features from a Parquet file.

        Reads and filters features from a Parquet file based on the specified task
        and file list. The method ensures the returned DataFrame contains only
        the relevant feature vectors in the correct order.

        Args:
            pqt_path (str): Path to the Parquet file containing features.
                The file should contain:
                - A `midi_file` column with corresponding filenames (str)
                - A `task` column indicating the task name (str).
                If the file contains features
                for only one task, this column may be omitted.
                - Feature columns (e.g., `E_0`, `E_1`, ...) with numerical values
                (int or float)
            task_name (str): Name of the task to load features for.
            filelist (list[str]): List of filenames to include in the output.
                The resulting DataFrame will be ordered according to this list.

        Returns:
            pd.DataFrame: Filtered DataFrame of shape (len(filelist), n_features)
                containing the loaded features, ordered to match `filelist`.

        Raises:
            ValueError: If `pqt_path` does not exist.
        """
        if not Path.exists(Path(pqt_path)):
            msg = f"Parquet path '{pqt_path}' does not exist."\
                f" Cannot load features for {task_name} task."  # noqa: ISC002
            raise ValueError(msg)

        log_msg = f"Loading features from {pqt_path}"
        logger.info(log_msg)
        df = pd.read_parquet(pqt_path)
        columns2drop = [MIDI_FILE_COLUMN]

        # check if there are features in parquet for our task
        if TASK_NAME_COLUMN in df.columns\
        and task_name not in df[TASK_NAME_COLUMN].values:
            log_msg = f"No features for {task_name} task found in parquet file"\
                f" {pqt_path}. Extracting them." # noqa: ISC002
            logger.info(log_msg)
            return None

        if TASK_NAME_COLUMN in df.columns:
            df = df[df[TASK_NAME_COLUMN]==task_name].reset_index(drop=True)
            columns2drop.append(TASK_NAME_COLUMN)

        df = self.filter_features(df, filelist)

        return df.drop(columns=columns2drop)

    def get_features_with_labels(
        self,
        task_name: str,
        meta_dataset: MetaDataset
    ) -> pd.DataFrame:
        """Create a DataFrame with features and labels for the specified task.

        Loads or extracts features for the given task and combines them with
        corresponding labels  from the metadata. Uses cached features if available
        and configured; otherwise falls back to on-demand extraction.

        Args:
            task_name (str): Name of the task for which to create the DataFrame.
            meta_dataset (MetaDataset): MetaDataset object containing:
                - Absolute paths to .mid files (or other supported formats)
                - Filenames
                - Optional labels for each file

        Returns:
            pd.DataFrame: DataFrame containing the features and labels, with shape
                (n_files, n_features + n_label_columns).
        """
        if self.use_cached and self.persistence_path != "":
            features = self.load_features_from_pqt(
                pqt_path=self.persistence_path,
                task_name=task_name,
                filelist=meta_dataset.filenames
            )
            # if no features found in parquet,
            # None returned and feature extractor is called
            if features is None:
                features = self.feature_extractor\
                    .extract_features_from_files(meta_dataset.paths_to_files)
        else:
            features = self.feature_extractor\
                .extract_features_from_files(meta_dataset.paths_to_files)

        if not self.use_cached and self.persistence_path != "":
            self.write_features_to_pqt(
                pqt_path=self.persistence_path,
                task_name=task_name,
                filelist=meta_dataset.filenames,
                features=features
            )

        if hasattr(meta_dataset, "labels"):
            df = embs_and_labels_to_df(features, meta_dataset.labels)
        else:
            df = pd.DataFrame(features)

        log_msg = f"Features shape for {task_name} task is {features.shape}"
        logger.info(log_msg)
        return df
