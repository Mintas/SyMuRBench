"""Utilities."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.io.formats.style as pifs
import yaml

from .constant import DEFAULT_METRIC_NAMES_2MINIMIZE, TARGET_COLUMN


def embs_and_labels_to_df(
    embeddings: pd.DataFrame,
    labels: np.ndarray
) -> pd.DataFrame:
    """
    Concatenate embeddings and labels. Return DataFrame.

    Args:
        embeddings (pd.DataFrame): DataFrame with features (int or float)
        labels (np.ndarray): numpy array with labels (int)

    Returns:
        pd.DataFrame: DataFrame with embeddings and labels.
        example of column names for multiclass task:
        `|E_0|E_1|E_2|...|target|`
        example of column names for multiclass task:
        `|E_0|E_1|E_2|...|target_0|target_1|...|`
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
    Validate if all file paths exist.

    Args:
        file_paths (list[str]): list of file paths.

    Raises:
        ValueError: If any file path does not exist.
    """
    for file_path in file_paths:
        if not Path.exists(Path(file_path)):
            msg = f"File path '{file_path}' does not exist."
            raise ValueError(msg)


def load_yaml(
    yaml_path: str
) -> dict:
    """
    Load data from YAML file.

    Args:
        yaml_path (str): path to file

    Returns:
        dict: loaded data
    """
    if not Path.exists(Path(yaml_path)):
        msg = "YAML file does not exists."
        raise ValueError(msg)

    with Path.open(Path(yaml_path)) as file:
        return yaml.load(file, Loader=yaml.SafeLoader)

def highlight_values(
    s: pd.Series,
    good: bool,
    ci: bool=False
) -> list[str]:
    """Highlight values in dataframe.

    Args:
        s (pd.Series): dataframe column
        good (bool):
            If True, highlight good values with green color,
            otherwise highlight poor values with red color
        ci (bool, optional):
            If True, cell values are considered as strings in format `0.05 ± 0.01`,
            else cell values are considered as floats.
            Defaults to False.

    Returns:
        list[str]: list with bg colors
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
    """
    Display Styler object.

    Args:
        df (pd.DataFrame): DataFrame with metrics to display
        round_num (int):
            The number of decimals to use when rounding the number.
            Defaults to 2.
        ci (bool, optional):
            If True, cell values are considered as strings in format `0.05 ± 0.01`,
            else cell values are considered as floats.
            Defaults to False.
        colored (bool):
            flag responsible for highlighting the Best/Worst metrics.
            Defaults to True.
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
    """
    Recursively traverse a dict to get list of leaf nodes.

    Args:
        x (dict): dict for traversal

    Returns:
        list: list containing tuples with leaf nodes and paths to them
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
