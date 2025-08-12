"""Constants and default values."""
from importlib import resources
from pathlib import Path

import yaml

# seed for numpy
SEED = 42
# num threads for torch threading and fast feature extraction
NUM_THREADS = 4

# csv column names
TASK_NAME_COLUMN = "task"
MIDI_FILE_COLUMN = "midi_file"
MIDI_SCORE_COLUMN = "midi_score"
MIDI_PERFORMANCE_COLUMN = "midi_performance"
TARGET_COLUMN = "target"

# paths to the LightAutoML YAML configuration files (for classification tasks only)
DEFAULT_LAML_CONFIG_PATHS = {
    "multiclass":resources\
        .files("symurbench.default_configs")\
        .joinpath("automl_multiclass_config.yaml"),
    "multilabel":resources\
        .files("symurbench.default_configs")\
        .joinpath("automl_multilabel_config.yaml")
}

# default config for MeanSklearnScorer with metrics to calculate
DEFAULT_SKLEARN_SCORER_CONFIG = {
    "multiclass": {
        "balanced_accuracy_score": None,
        "f1_score": {"average":"weighted"}},
    "multilabel": {"f1_score": {"average":"weighted"}}
}

# default ranks to calculate R@K metrics for ScorePerformanceRetrieval task
DEFAULT_RETRIEVAL_RANKS = (1,5,10)

# possible metric names to minimize in task evaluation
DEFAULT_METRIC_NAMES_2MINIMIZE = {
    "Median_Rank",
    "hinge_loss",
    "hamming_loss",
    "log_loss",
    "zero_one_loss"
}

# path to the config containing dataset paths
DATASETS_CONFIG_PATH = resources\
    .files("symurbench.default_configs")\
    .joinpath("tasks_config.yaml")

# huggingface repo ID
HF_DATASET_REPO = "ai-forever/symurbench_datasets"

# function for loading dataset paths
def get_default_metadata_path(
    task: str
) -> dict:
    """
    Return arguments for task initialization from config.

    Args:
        task (str): name of the task.

    Returns:
        dict: dictionary with arguments.
    """
    with Path.open(DATASETS_CONFIG_PATH) as file:
        data = yaml.safe_load(file)

    return data[task]
