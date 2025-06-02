"""Abstract Base Class for feature extraction implemetation."""
import logging
from abc import ABC, abstractmethod
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import utils
from .constant import MIDI_FILE_COLUMN, NUM_THREADS, TASK_NAME_COLUMN
from .metaloaders.metaloader import MetaDataset

logger = logging.getLogger("feature extractor")

class FeatureExtractor(ABC):
    """
    Abstract Base Class for feature extraction from a list of files.

    This class provides an API interface for implementing feature extraction functions.
    The result of this would be the desired vector representation or embedding
    to be tested in the underlying benchmark.

    Users can create their own subclass and implement
    the `extract_features_from_file` method to extract features from a file.

    Users can also override the `extract_features_from_files` method
    to extract features from files in batches rather than one at a time.
    """

    def __init__(
        self,
        extractor_name: str,
        fast: bool = False,
        preprocess_features: bool = True,
    ) -> None:
        """
        Initialize the FeatureExtractor class.

        Args:
            extractor_name (str): Name of the feature extractor. Should be unique.
            fast (bool, optional):
                if True, will use multiprocessing to extract features.
                Defaults to False.
            preprocess_features (bool):
                whether to preprocess features for tasks with AutoML.
                If True, preprocess features according to automl config.
                If False, use features as they are.
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
        """
        Extract features from a file.

        Args:
            file_path (str): absolute path to MIDI file.
                (or any other extension for symbolic music
                if datasets are converted to the desired format).

        Returns:
            np.ndarray:
                numpy array with shape (n_features,) with extracted features
        """

    def _extract_features_from_files(
        self,
        file_paths: list[str]
    ) -> np.ndarray:
        """
        Extract features from a list of files. Can be overridden.

        Args:
            file_paths (list[str]): list of absolute paths to MIDI files
            fast (bool, optional):
                if True, will use multiprocessing to extract features.
                Defaults to False.

        Returns:
            np.ndarray: numpy array with shape (n_files, n_features)
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
        """
        Extract features from a list of files.

        The order of the files is crucial.
        The extracted features should be placed in the same order as the input files.

        Args:
            file_paths (list[str]): list of absolute paths to MIDI files
            (or any other extension for symbolic music
            if datasets are converted to the desired format).

        Raises:
            ValueError: if the extracted features matrix has shape len less than 2.
            ValueError: if features.shape[0] != len(file_paths)

        Returns:
            pd.DataFrame:
                pandas.DataFrame with extracted features;
                Shape: (len(file_paths), n_features);
                DataFrame do not contain header or index column
        """
        utils.validate_file_paths(file_paths)
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
        """
        Create DataFrame with features and labels for the task with name task_name.

        Args:
            task_name (str): name of the task. (All tasks listed in tasks folder)
            meta_dataset (MetaDataset): MetaDataset object.
                Contains:
                1) absolute paths to .mid files;
                2) Filenames;
                3) labels (optional)

        Returns:
            pd.DataFrame: DataFrame with embeddings and labels
        """
        features = self.extract_features_from_files(meta_dataset.paths_to_files)
        if hasattr(meta_dataset, "labels"):
            df = utils.embs_and_labels_to_df(features, meta_dataset.labels)
        else:
            df = pd.DataFrame(features)
        log_msg = f"Features shape for {task_name} task is {features.shape}"
        logger.info(log_msg)

        return df

class PersistentFeatureExtractor:
    """
    A class that extends FeatureExtractor.

    Provides methods for writing and loading extracted features
    at the file system level (using parquet files).

    It also includes a fallback mechanism
    that delegates management to the FeatureExtractor class.

    Fallback to FeatureExtractor occurs in the following cases:
    - No "persistence_path" provided
    - "use_cached" is False
    - "use_cached" is True, but no features are found in the parquet file
    at the specified "persistence_path"
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
        """
        Filter features and check their order using an inclusion list of files.

        Args:
            df (pd.DataFrame): DataFrame with features for task with name task_name
            task_name (str): name of the task
            files_to_keep (list[str]):
                list of filenames to keep in the features DataFrame

        Raises:
            ValueError: if len(files_to_keep) <= 1.
            ValueError: if number of rows in filtered dataframe is less than\
                len(files_to_keep)

        Returns:
            pd.DataFrame: filtered df
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
        """
        Validate existing_df for writing features.

        Args:
            df (pd.DataFrame):
                DataFrame with features to save
            existing_df (pd.DataFrame):
                existing DataFrame with features
            task_name (str):
                name of the task

        Raises:
            ValueError: if there are no 'task' or 'midi_file' columns in existing_df
            ValueError: if existing_df contain features for current task
                and self.overwrite_features is False
            ValueError: if existing_df.shape[1] != df.shape[1]
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
        """
        Write calculated features to a parquet file.

        Args:
            pqt_path (str):
                path to parquet.
                Parquet will contain:
                1) column "midi_file" with corresponding file names - [str]
                2) column "task" with corresponding task name - [str]
                3) feature vectors in rows - [int or float]
                separator = ","
                Example:
                    `|midi_file|task    |E_0|E_1|E_2|`
                    `________________________________`
                    `|file1.mid|genre   | 0 |0.1|1.1|`
                    `|file2.mid|genre   | 1 |3.8|4.0|`
                    `|file3.mid|composer| 0 |1.1|4.8|`
                    `|file4.mid|composer| 0 |4.5|7.6|`

            task_name (str): name of the task
            filelist (list[str]): list of filenames from task dataset
            features (pd.DataFrame): DataFrame with extracted features
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
        """
        Load features from a parquet file.

        Args:
            pqt_path (str):
                path to parquet file.
                Parquet file should contain:
                1) feature vectors in rows - [int or float]
                2) column "midi_file" with corresponding file names - [str]
                3) column "task" with corresponding task name - [str]
                (if parquet contains features for one task only,
                column "task" is not neccesary)
            task_name (str): name of the task
            filelist (list[str]): list of filenames from task dataset

        Raises:
            ValueError: if pqt_path does not exist

        Returns:
            pd.DataFrame : DataFrame with loaded features
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
        """
        Create pd.DataFrame with features and labels for the task named task_name.

        Args:
            task_name (str): name of the task
            meta_dataset (MetaDataset): MetaDataset object.
                Contains:
                1) absolute paths to .mid files;
                2) Filenames;
                3) labels (optional)

        Returns:
            pd.DataFrame: DataFrame with embeddings (and labels)
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
            df = utils.embs_and_labels_to_df(features, meta_dataset.labels)
        else:
            df = pd.DataFrame(features)

        log_msg = f"Features shape for {task_name} task is {features.shape}"
        logger.info(log_msg)
        return df
