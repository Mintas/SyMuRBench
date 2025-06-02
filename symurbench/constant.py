"""Constants and default values."""
# seed for numpy
SEED = 42
# num threads for torch threading and fast feature extraction
NUM_THREADS = 8

# csv column names
TASK_NAME_COLUMN = "task"
MIDI_FILE_COLUMN = "midi_file"
MIDI_SCORE_COLUMN = "midi_score"
MIDI_PERFORMANCE_COLUMN = "midi_performance"
TARGET_COLUMN = "target"

# paths to LightAutoML YAML configuration files (for classification tasks only)
DEFAULT_LAML_CONFIG_PATHS = {
    "multiclass":"configs/automl_multiclass_config.yaml",
    "multilabel":"configs/automl_multilabel_config.yaml"
}

# default config for MeanSklearnScorer with metrics to calculate
DEFAULT_SKLEARN_SCORER_CONFIG = {
    "multiclass": {
        "balanced_accuracy_score": None,
        "f1_score": {"average":"weighted"}},
    "multilabel": {"f1_score": {"average":"weighted"}}
}

#default ranks to calculate R@K metrics for ScorePerformanceRetrieval task
DEFAULT_RETRIEVAL_RANKS = (1,5,10)

# possible metric names to minimize in task evaluation
DEFAULT_METRIC_NAMES_2MINIMIZE = {
    "Median_Rank",
    "hinge_loss",
    "hamming_loss",
    "log_loss",
    "zero_one_loss"
}

# default paths to CSV files with metadata and paths to folders with MIDI files
DEFAULT_METADATA_PATHS = {
    "ComposerClassificationASAP": {
        "metadata_csv_path":"data/datasets/composer_and_retrieval_datasets/metadata_composer_dataset.csv",
        "files_dir_path":"data/datasets/composer_and_retrieval_datasets/"
    },
    "EmotionClassificationEMOPIA": {
        "metadata_csv_path":"data/datasets/emopia_dataset/metadata_emopia_dataset.csv",
        "files_dir_path":"data/datasets/emopia_dataset/midis/"
    },
    "EmotionClassificationMIREX": {
        "metadata_csv_path":"data/datasets/mirex_dataset/metadata_mirex_dataset.csv",
        "files_dir_path":"data/datasets/mirex_dataset/midis/"
    },
    "GenreClassificationMMD": {
        "metadata_csv_path":"data/datasets/genre_dataset/metadata_genre_dataset.csv",
        "files_dir_path":"data/datasets/genre_dataset/midis/"
    },
    "GenreClassificationWMTX": {
        "metadata_csv_path":"data/datasets/wikimtx_dataset/metadata_wikimtx_dataset.csv",
        "files_dir_path":"data/datasets/wikimtx_dataset/midis/"
    },
    "InstrumentDetectionMMD": {
        "metadata_csv_path":"data/datasets/instrument_dataset/metadata_instrument_dataset.csv",
        "files_dir_path":"data/datasets/instrument_dataset/midis/"
    },
    "ScorePerformanceRetrievalASAP": {
        "metadata_csv_path":"data/datasets/composer_and_retrieval_datasets/metadata_retrieval_dataset.csv",
        "files_dir_path":"data/datasets/composer_and_retrieval_datasets/"
    },
}
