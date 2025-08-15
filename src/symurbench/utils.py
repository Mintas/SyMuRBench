"""Utilities."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.io.formats.style as pifs
import yaml
from huggingface_hub import hf_hub_download

from .constant import (
    DATASETS_CONFIG_PATH,
    DEFAULT_METRIC_NAMES_2MINIMIZE,
    HF_DATASET_REPO,
    TARGET_COLUMN,
)


def embs_and_labels_to_df(
    embeddings: pd.DataFrame,
    labels: np.ndarray
) -> pd.DataFrame:
    """
    Concatenate embeddings and labels into a single DataFrame.

    Combines a DataFrame of embeddings (features) with a numpy array of labels along
    the column axis, producing a unified DataFrame for use in downstream tasks.

    Args:
        embeddings (pd.DataFrame):
            DataFrame of shape (n_samples, n_features) containing feature vectors.
            Values should be numeric (int or float).
        labels (np.ndarray):
            Array of shape (n_samples,) for single-label (e.g., multiclass) tasks or
            (n_samples, n_targets) for multi-label tasks. Values should be integers.

    Returns:
        pd.DataFrame:
            Combined DataFrame of shape (n_samples, n_features + n_label_columns) with:
            - Feature columns: `E_0`, `E_1`, ..., `E_n`
            - Label columns: `target` (for multiclass) or
            `target_0`, `target_1`, ... (for multilabel)

    Example:
        For a multiclass task:
        ```
        | E_0 | E_1 | E_2 | target |
        |-----|-----|-----|--------|
        | 0.1 | 1.3 | 2.2 |   0    |
        ```

        For a multilabel task:
        ```
        | E_0 | E_1 | E_2 | target_0 | target_1 |
        |-----|-----|-----|----------|----------|
        | 0.1 | 1.3 | 2.2 |    1     |    0     |
        ```
    """
    embeddings.columns = [f"E_{i}" for i in range(embeddings.shape[1])]

    if labels is not None:
        n_label_cols = labels.shape[1] if len(labels.shape) == 2 else 1
        if n_label_cols > 1:
            columns = [f"{TARGET_COLUMN}_{i}" for i in range(labels.shape[1])]
        else:
            columns = [TARGET_COLUMN]
        labels = pd.DataFrame(labels, columns=columns)
        embeddings = pd.concat([embeddings, labels], axis=1)

    return embeddings

def validate_file_paths(
    file_paths: list[str]
) -> None:
    """
    Validate that all file paths in the given list exist.

    Args:
        file_paths (list[str]):
            List of absolute or relative file paths to validate.

    Raises:
        ValueError: If any of the file paths does not exist.
    """
    for file_path in file_paths:
        if not Path.exists(Path(file_path)):
            msg = f"File path '{file_path}' does not exist."
            raise ValueError(msg)


def load_yaml(
    yaml_path: str
) -> dict:
    """
    Load data from a YAML file.

    Args:
        yaml_path (str): Path to the YAML file to load. Can be absolute or relative.

    Returns:
        dict: Dictionary containing the parsed YAML data.
    """
    if not Path.exists(Path(yaml_path)):
        msg = "YAML file does not exists."
        raise FileNotFoundError(msg)

    with Path.open(Path(yaml_path)) as file:
        return yaml.safe_load(file)

def highlight_values(
    s: pd.Series,
    good: bool,
    ci: bool=False
) -> list[str]:
    """Apply background color highlighting to a pandas Series based on values.

    Highlights the best or worst values in a Series using green (good) or red (bad)
    background colors. Designed for use with DataFrame styling in Jupyter notebooks
    or HTML output.

    Args:
        s (pd.Series): Series representing a column of metric values to highlight.
        good (bool): If True, highlights the best-performing values in green.
            If False, highlights the worst-performing values in red.
        ci (bool, optional): If True, interprets values as strings in the format
            "value ± error" (e.g., "0.85 ± 0.02") and extracts the numeric value
            before comparison. If False, treats values as floats. Defaults to False.

    Returns:
        list[str]: List of CSS background color strings for each element in the Series:
            - "background: forestgreen" for top-performing values when `good=True`
            - "background: firebrick" for worst-performing values when `good=False`

    Note:
        The decision of which metrics are "better" when maximized or minimized
        is determined by `DEFAULT_METRIC_NAMES_2MINIMIZE`.
    """
    vals = np.array([float(v.split(" ")[0]) for v in s._values]) if ci else s._values

    if True in [v in s.name[1] for v in DEFAULT_METRIC_NAMES_2MINIMIZE]:
        bool_mask = vals == vals.min() if good else vals == vals.max()
    else:
        bool_mask = vals == vals.max() if good else vals == vals.min()

    if good:
        return ["background: forestgreen" if cell else "" for cell in bool_mask]
    return ["background: firebrick" if cell else "" for cell in bool_mask]

def display_styler(
    df: pd.DataFrame,
    round_num: int = 2,
    ci: bool=False,
    colored: bool=True
) -> pifs.Styler:
    """Create a styled DataFrame for displaying evaluation metrics.

    Args:
        df (pd.DataFrame): DataFrame containing evaluation metrics to display.
        round_num (int, optional):
            Number of decimal places to use when formatting numeric values.
            Defaults to 2.
        ci (bool, optional):
            If True, interprets cell values as strings in the format "value ± error"
            (e.g., "0.85 ± 0.02") and formats accordingly. If False, treats values as
            plain floats. Defaults to False.
        colored (bool, optional): If True, applies color styling:
            - Best values are highlighted in green (forestgreen)
            - Worst values are highlighted in red (firebrick)
            The direction (higher=better or lower=better) is determined by
            `DEFAULT_METRIC_NAMES_2MINIMIZE`.
            Defaults to True.

    Returns:
        pandas.io.formats.style.Styler: A Styler object with applied formatting
    """
    styler = df.style

    border_style = {
        "selector": "td, th",
        "props": [
            ("border", "1px solid black"),
            ("border-collapse", "collapse"),
            ("padding", "5px"),
            ("text-align", "center")
        ]
    }

    styler.set_table_styles([border_style])
    styler.format(precision=round_num)

    if colored:
        def highlight_poor(
            s: pd.Series
        ) -> list[str]:
            """Highlight poor metrics with red."""
            return highlight_values(s=s, good=False, ci=ci)

        def highlight_good(
            s: pd.Series
        ) -> list[str]:
            """Highlight good metrics with green."""
            return highlight_values(s=s, good=True, ci=ci)

        styler.apply(highlight_poor).apply(highlight_good)

        logging.info("Green - the best values; red - the worst values")

    return styler

def nested_dict_to_list(x: dict) -> list:
    """Convert a nested dictionary into a flat list of leaf nodes with their full paths.

    Recursively traverses a nested dictionary and returns a list of tuples,
    where each tuple contains the full path to a leaf node (including the final key)
    and its associated value.

    A leaf node is defined as a key-value pair where the value is not a dictionary.

    Args:
        x (dict): The nested dictionary to flatten. Must be a non-empty dictionary.

    Returns:
        list[tuple]: A list of tuples in the form (*path_keys, value), where:
            - `path_keys` are the nested keys leading to the leaf
            - `value` is the leaf node value
    """
    result = []
    def traverse(current_dict: dict, path: list) -> None:
        for key, value in current_dict.items():
            if isinstance(value, dict):
                traverse(value, [*path, key])
            else:
                result.append((*path, key, value))
    traverse(x, [])
    return result

def load_datasets(
    output_folder: str | Path = "./",
    load_features: bool = True,
    token: str | None = None
) -> None:
    """Download and extract datasets and precomputed features from Hugging Face Hub.

    This function automates the setup of the dataset environment by:
    1. Downloading a zip archive containing dataset metadata and structure from HF.
    2. Extracting it to the specified output directory.
    3. Optionally downloading precomputed feature files (music21 and jSymbolic).
    4. Updating the local configuration file with absolute paths to the extracted data.

    The resulting directory structure supports immediate use in benchmarking tasks.

    Args:
        output_folder (str | Path, optional): Root directory where datasets and features
            will be extracted. Created if it does not exist. Defaults to "./".
        load_features (bool, optional): If True, downloads and saves precomputed
            music21 and jSymbolic feature files. If False, only metadata is downloaded.
            Defaults to True.
        token (str | None, optional):
            Authentication token for accessing private or gated
            repositories on Hugging Face Hub. Defaults to None.
    """
    datasets = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        subfolder="data",
        filename="datasets.zip",
        repo_type="dataset",
        token=token
    )

    import zipfile
    extract_path = Path(output_folder)
    if not extract_path.exists():
        extract_path.mkdir()
    with zipfile.ZipFile(datasets, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    logging.info("Datasets extraction finished.")


    if load_features:
        music21_feats_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            subfolder="data/features",
            filename="music21_full_dataset.parquet",
            repo_type="dataset",
            token=token
        )

        jsymbolic_feats_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            subfolder="data/features",
            filename="jsymbolic_full_dataset.parquet",
            repo_type="dataset",
            token=token
        )

        music21_df = pd.read_parquet(music21_feats_path)
        jsymbolic_df = pd.read_parquet(jsymbolic_feats_path)

        features_path = extract_path / "features"
        if not features_path.exists():
            features_path.mkdir()
        music21_df\
            .to_parquet(features_path / "music21_full_dataset.parquet", index=False)
        jsymbolic_df\
            .to_parquet(features_path / "jsymbolic_full_dataset.parquet", index=False)

        logging.info("Finished loading features.")


    default_tasks_config = load_yaml(DATASETS_CONFIG_PATH)
    for task in default_tasks_config:
        for k,v in default_tasks_config[task].items():
            default_tasks_config[task][k] = str(
                extract_path.resolve() / Path("/".join(v.split("/")[-3:]))
            )
    with Path.open(DATASETS_CONFIG_PATH, "w") as file:
        yaml.dump(default_tasks_config, file)
