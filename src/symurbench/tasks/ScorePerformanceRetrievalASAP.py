"""Score-perfomance retrieval task."""  # noqa: N999
from symurbench.abstract_tasks.retrieval_task import RetrievalTask
from symurbench.constant import DEFAULT_METADATA_PATHS
from symurbench.metaloaders.metaloader_retrieval import RetrievalMetaLoader
from symurbench.metrics.retrieval_scorer import RetrievalScorer
from symurbench.metrics.scorer import BaseScorer


class ScorePerformanceRetrievalASAP(RetrievalTask):
    """Class for Score-perfomance retrieval task."""

    name = "ScorePerformanceRetrievalASAP"
    description = "Score-performance retrieval. ASAP Dataset."
    metadata = RetrievalMetaLoader(
        metadata_csv_path=DEFAULT_METADATA_PATHS[name]["metadata_csv_path"],
        files_dir_path=DEFAULT_METADATA_PATHS[name]["files_dir_path"]
    )

    def __init__(
        self,
        scorer: BaseScorer | None = None,
        postfixes: tuple[str, str] = ("sp", "ps")
    ) -> None:
        """
        Task initialization. Prepare dataset for feature extraction.

        Args:
            scorer (BaseScorer | None, optional):
                scorer to use for metrics calculation. Defaults to None.
            postfixes (tuple[str, str], optional):
                postfixes to use in names of retrieval metrics.
                Defaults to ("sp", "ps").
        """
        if scorer is None:
            scorer = RetrievalScorer()
        super().__init__(scorer, postfixes)
