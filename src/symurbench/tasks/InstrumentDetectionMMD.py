"""Instrument detection task."""  # noqa: N999
from symurbench.abstract_tasks.classification_task import ClassificationTask
from symurbench.constant import DEFAULT_LAML_CONFIG_PATHS, DEFAULT_METADATA_PATHS
from symurbench.metaloaders.metaloader_multilabel import MultilabelMetaLoader
from symurbench.metrics.scorer import BaseScorer
from symurbench.metrics.sklearn_scorer import SklearnClsScorer


class InstrumentDetectionMMD(ClassificationTask):
    """Multilabel classification task. Instrument detection."""

    name = "InstrumentDetectionMMD"
    description = "Instrument detection. MetaMIDI Dataset."
    metadata = MultilabelMetaLoader(
        metadata_csv_path=DEFAULT_METADATA_PATHS[name]["metadata_csv_path"],
        files_dir_path=DEFAULT_METADATA_PATHS[name]["files_dir_path"]
    )

    def __init__(
        self,
        automl_config_path: str = DEFAULT_LAML_CONFIG_PATHS["multilabel"],
        scorer: BaseScorer | None = None
    ) -> None:
        """
        Task initialization. Prepare dataset for feature extraction.

        Args:
            automl_config_path (str, optional): path to config for AutoML.
                Defaults to DEFAULT_LAML_CONFIG_PATHS["multiclass"].
            scorer (BaseScorer | None, optional):
                scorer to use for metrics calculation. Defaults to None.
        """
        if scorer is None:
            scorer = SklearnClsScorer(task_type="multilabel")
        super().__init__(automl_config_path, scorer)
