"""Genre classification task."""  # noqa: N999
from symurbench.abstract_tasks.classification_task import ClassificationTask
from symurbench.constant import DEFAULT_LAML_CONFIG_PATHS, get_default_metadata_path
from symurbench.metrics.scorer import BaseScorer
from symurbench.metrics.sklearn_scorer import SklearnClsScorer


class GenreClassificationWMTX(ClassificationTask):
    """Multiclass classification task. Genre classification."""

    name = "GenreClassificationWMTX"
    description = "Genre classification. WikiMT-X Dataset."

    def __init__(
        self,
        metaloader_args_dict: dict | None = None,
        automl_config_path: str = DEFAULT_LAML_CONFIG_PATHS["multiclass"],
        scorer: BaseScorer | None = None,
    ) -> None:
        """
        Task initialization. Prepare dataset for feature extraction.

        Args:
            metaloader_args_dict (dict):
                Dictionary of arguments passed to the metaloader constructor.
                Expected keys:
                - metadata_csv_path (str):
                Absolute path to the CSV file containing dataset metadata.
                - files_dir_path (str):
                Absolute path to the directory containing dataset files.
                - dataset_filter_list (list[str], optional):
                List of filenames to include (inclusion filter).
                If not provided, all files are used.
            automl_config_path (str, optional):
                Path to the AutoML configuration file.
                Defaults to DEFAULT_LAML_CONFIG_PATHS["multiclass"].
            scorer (BaseScorer | None, optional):
                Scorer instance to use for metric calculation.
                If None, a default scorer may be used based on the task type.
                Defaults to None.
        """
        if scorer is None:
            scorer = SklearnClsScorer(task_type="multiclass")
        if metaloader_args_dict is None:
            metaloader_args_dict = get_default_metadata_path(self.name)
        super().__init__(metaloader_args_dict, automl_config_path, scorer)
