"""Score-perfomance retrieval task."""  # noqa: N999
from symurbench.abstract_tasks.retrieval_task import RetrievalTask
from symurbench.constant import get_default_metadata_path
from symurbench.metrics.retrieval_scorer import RetrievalScorer
from symurbench.metrics.scorer import BaseScorer


class ScorePerformanceRetrievalASAP(RetrievalTask):
    """Class for Score-perfomance retrieval task."""

    name = "ScorePerformanceRetrievalASAP"
    description = "Score-performance retrieval. ASAP Dataset."

    def __init__(
        self,
        metaloader_args_dict: dict | None = None,
        scorer: BaseScorer | None = None,
        postfixes: tuple[str, str] = ("sp", "ps")
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
            scorer (BaseScorer | None, optional):
                scorer to use for metrics calculation. Defaults to None.
            postfixes (tuple[str, str], optional):
                Postfixes to use in the names of retrieval metrics,
                representing retrieval direction. Defaults to ("sp", "ps").
        """
        if scorer is None:
            scorer = RetrievalScorer()
        if metaloader_args_dict is None:
            metaloader_args_dict = get_default_metadata_path(self.name)
        super().__init__(metaloader_args_dict, scorer, postfixes)
