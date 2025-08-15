"""Benchmark class implementtion."""
import logging
import sys

import numpy as np
import pandas as pd
import torch

from . import utils
from .abstract_tasks.abstask import AbsTask
from .abstract_tasks.classification_task import ClassificationTask
from .abstract_tasks.retrieval_task import RetrievalTask
from .constant import NUM_THREADS, SEED
from .feature_extractor import FeatureExtractor, PersistentFeatureExtractor
from .tasks import *  # noqa: F403


def setup_logging(
    level: int = logging.INFO
) -> None:
    """Configure logging with a standardized format and handler.

    Sets up the root logger with a stream handler that outputs to stdout.
    Applies a consistent format including time, log level, and message.
    Ensures clean setup by removing any existing handlers to prevent duplicates.

    Args:
        level (int, optional): The minimum logging level to display for the handler.
            Defaults to `logging.INFO`.

    Note:
        - The root logger is set to `DEBUG` level to capture all messages,
          while the handler filters based on the provided `level`.
        - Propagation is enabled for the 'lightautoml' logger to ensure
          logs are handled correctly.
    """
    # clean up any existing handlers
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.DEBUG)

    # avoid duplicate logs
    logging.getLogger("lightautoml").propagate = True

    # formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    # add to root
    logging.getLogger().addHandler(handler)

setup_logging()
logger = logging.getLogger(__name__)
np.random.seed(SEED)  # noqa: NPY002
torch.manual_seed(SEED)
torch.set_num_threads(NUM_THREADS)

class Benchmark:
    """Main class for benchmarking different feature extractors."""
    def __init__(
        self,
        feature_extractors_list: list[FeatureExtractor | PersistentFeatureExtractor],
        tasks: list[str] | list[AbsTask] | None = None,
    ) -> None:
        """Initialize the benchmark for comparing feature extractors.

        Args:
            feature_extractors_list (list[FeatureExtractor|PersistentFeatureExtractor]):
                List of feature extractor instances to benchmark.
                Can include both regular and persistent (cached) extractors.
            tasks (list[str] | list[AbsTask] | None, optional):
                List of task names (as strings) or pre-instantiated task objects
                to evaluate on. If None, a default set of tasks may be used.
                Defaults to None.
        """
        self.feature_extractors = feature_extractors_list
        self.validate_feature_extractors()
        self.tasks = self.validate_tasks_argument(tasks)

        log_msg = f"Metadata is loaded for {len(self.tasks)} task(s)."
        logger.info(log_msg)

        # dict where calculated metrics are saved
        self.metrics = {t.name:{} for t in self.tasks}

    def validate_tasks_argument(
        self,
        tasks: list[str] | list[AbsTask] | None,
    ) -> list[AbsTask]:
        """Validate the tasks argument.

        Args:
            tasks (list[str] | list[AbsTask] | None):
                List of task names (strings) or pre-instantiated task objects.

        Returns:
            list[AbsTask]: List of instantiated task objects.

        Raises:
            TypeError: If `tasks` is not a list, is empty, or contains elements
                that are neither strings nor instances of AbsTask.
            ValueError: If task names are not unique (case-sensitive).
        """
        if tasks is None:
            return self.__class__.get_tasks(
                task_names=tasks
            )

        if not isinstance(tasks, list) or len(tasks) == 0:
            msg = "Provide non-empty list with task objects: "\
                "list[AbsTask()] or list with names of the tasks: "\
                "list[str]."  # noqa: ISC002
            raise TypeError(msg)

        is_str = {isinstance(task, str) for task in tasks}
        is_abstask = {isinstance(task, AbsTask) for task in tasks}

        if len(is_str) > 1\
        or len(is_abstask) > 1\
        or is_str == is_abstask == {False}:
            msg = "Argument 'tasks' should be a list "\
                "of one of the following types: "\
                "list[AbsTask] or list[str]."  # noqa: ISC002
            raise TypeError(msg)

        msg_unique = "Task names should be unique."
        if is_str == {True}:
            if len(set(tasks)) == len(tasks):
                return self.__class__.get_tasks(
                    task_names=tasks
                )
            raise ValueError(msg_unique)

        if not len({t.name for t in tasks}) == len(tasks):
            raise ValueError(msg_unique)

        return tasks

    def validate_feature_extractors(
        self,
    ) -> None:
        """Validate the feature extractors list.

        Raises:
            TypeError: If `self.feature_extractors` is not a list, is empty,
                or contains elements    that are not instances of FeatureExtractor
                or PersistentFeatureExtractor.
            ValueError: If the names of the feature extractors are not unique.
        """
        if not isinstance(self.feature_extractors, list)\
        or len(self.feature_extractors) == 0\
        or False in {
            isinstance(f, (FeatureExtractor, PersistentFeatureExtractor))
            for f in self.feature_extractors
        }:
            msg = "Provide non-empty list with FeatureExtractor "\
                "and/or PersistentFeatureExtractor objects: "\
                "list[FeatureExtractor|PersistentFeatureExtractor]."  # noqa: ISC002
            raise TypeError(msg)

        if len({f.name for f in self.feature_extractors}) != len(self.feature_extractors):  # noqa: E501
            msg = "Feature extractors names should be unique."
            raise ValueError(msg)


    @property
    def task_names(
        self
    ) -> list[str]:
        """Get names of all loaded tasks.

        Returns:
            list[str]: List of task names.
        """
        return [t.name for t in self.tasks]

    @classmethod
    def get_tasks(
        cls,
        task_names: list[str] | None,
        init_tasks: bool = True
    ) -> list[ClassificationTask | RetrievalTask]:
        """Load and initialize tasks.

        Args:
            task_names (list[str] | None): List of task names to load.
                If None, all available tasks are loaded.
            init_tasks (bool, optional): If True, returns initialized task objects.
                If False, returns task classes without instantiation. Defaults to True.

        Returns:
            list[ClassificationTask | RetrievalTask]:
                List of task objects (if `init_tasks` is True)
                or task classes (if `init_tasks` is False).
        """
        tasks_lvl_1 = AbsTask.__subclasses__()
        tasks_lvl_2 = []
        for task in tasks_lvl_1:
            tasks_lvl_2 += task.__subclasses__()

        if task_names is not None:
            if init_tasks:
                return [t() for t in tasks_lvl_2 if t.name in task_names]
            return [t for t in tasks_lvl_2 if t.name in task_names]

        if init_tasks:
            return [t() for t in tasks_lvl_2]
        return tasks_lvl_2

    @classmethod
    def init_from_config(
        cls,
        feature_extractors_list: list[FeatureExtractor | PersistentFeatureExtractor],
        tasks_config: dict
    ) -> None:
        """Initialize the benchmark from a configuration dictionary.

        Args:
            feature_extractors_list (list[FeatureExtractor|PersistentFeatureExtractor]):
                List of feature extractor instances (regular or persistent) to compare.
            tasks_config (dict):
                Configuration dictionary specifying tasks and their parameters.
                Structure:
                - Key: task name (str)
                - Value: dict containing:
                1) Arguments for MetaLoader (e.g., `metadata_csv_path`)
                2) Additional arguments for the task's `__init__` method
                Example configuration can be found in
                symurbench.constant.DATASETS_CONFIG_PATH file.

        Returns:
            Benchmark: An initialized Benchmark instance configured with the provided
                feature extractors and tasks.

        Raises:
            ValueError: If `tasks_config` is empty or invalid.
        """
        if {isinstance(task, str) for task in tasks_config} != {True}:
            msg = "Argument 'tasks_config' should be a list "\
                             "of task names: list[str]."  # noqa: ISC002
            raise ValueError(msg)
        tasks_cls = cls.get_tasks(
            task_names=list(tasks_config.keys()),
            init_tasks=False
        )
        tasks = [task.pass_args(**tasks_config[task.name]) for task in tasks_cls]

        return cls(
            feature_extractors_list=feature_extractors_list,
            tasks=tasks
        )

    @classmethod
    def init_from_config_file(
        cls,
        feature_extractors_list: list[FeatureExtractor | PersistentFeatureExtractor],
        tasks_config_path: str
    ) -> None:
        """Initialize the benchmark from a YAML configuration file.

        Args:
            feature_extractors_list (list[FeatureExtractor|PersistentFeatureExtractor]):
                List of feature extractor instances (regular or persistent) to compare.
            tasks_config_path (str):
                Path to the YAML file containing the task configuration.
                Config has the following structure:
                - Key: task name (str)
                - Value: dict containing:
                1) Arguments for MetaLoader (e.g., `metadata_csv_path`)
                2) Additional arguments for the task's `__init__` method
                Example configuration can be found in
                symurbench.constant.DATASETS_CONFIG_PATH file.

        Returns:
            Benchmark: An initialized Benchmark instance.

        Raises:
            ValueError: If the config is empty.
        """
        config = utils.load_yaml(tasks_config_path)
        if len(config) == 0:
            msg = "You provided an empty config."
            raise ValueError(msg)

        return cls.init_from_config(
            feature_extractors_list=feature_extractors_list,
            tasks_config=config
        )

    def run_task(
        self,
        task: ClassificationTask | RetrievalTask,
        feature_extractor: FeatureExtractor | PersistentFeatureExtractor
    ) -> None:
        """Run a single task using the specified feature extractor.

        Executes the given task with the provided feature extractor
        and saves the calculated metrics.

        Args:
            task (ClassificationTask | RetrievalTask): The task to execute.
            feature_extractor (FeatureExtractor | PersistentFeatureExtractor):
                The feature extractor to evaluate on the task.
        """
        self.metrics[task.name][feature_extractor.name] = task.run(feature_extractor)

    def run_all_tasks(
        self
    ) -> None:
        """Run all configured tasks for each feature extractor.

        Executes every task in the benchmark on every feature extractor in the list,
        computing and storing performance metrics for each combination.
        """
        for feature_extractor in self.feature_extractors:
            msg_log = f"Running tasks for {feature_extractor.name} features."
            logger.info(msg_log)
            for task in self.tasks:
                msg_log = f"Running {task.name} task with "\
                    f"{feature_extractor.name} features."  # noqa: ISC002
                logger.info(msg_log)
                self.run_task(task, feature_extractor)

    def get_result_dict(
        self,
        round_num: int = 2,
        return_ci: bool = False,
        alpha: float = 0.05
    ) -> dict:
        """Aggregate metrics into a human-readable dictionary.

        Args:
            round_num (int, optional):
                Number of decimal places to use when rounding metric values.
                Defaults to 2.
            return_ci (bool, optional):
                If True, returns the confidence interval (e.g., "0.85 ± 0.02")
                as a string. If False, returns the aggregated numeric value.
                Defaults to False.
            alpha (float, optional):
                Significance level for calculating the margin of error
                in confidence intervals. Supported values are 0.05, 0.01, and 0.001.
                Defaults to 0.05.

        Returns:
            dict: Nested dictionary with the following structure:
                {
                    "task_name": {
                        "feature_extractor_name": {
                            "metric_name": metric_value
                        }
                    }
                }
                Metric values are either floats/ints (if `return_ci=False`)
                or strings (if `return_ci=True`).
        """
        traversal_list = utils.nested_dict_to_list(self.metrics)
        result_dict = {}
        for task, fe, metrics_list in traversal_list:
            current_metrics = {
                m.name: m.get_agg_value(
                    round_num=round_num,
                    return_ci=return_ci,
                    alpha=alpha
                )
                for m in metrics_list
            }
            if task not in result_dict: # add task name to dict
                result_dict[task] = {fe: current_metrics}
            else:
                result_dict[task][fe] = current_metrics

        return result_dict


    def get_result_df(
        self,
        round_num: int = 2,
        return_ci: bool = False,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """Aggregate metrics into a pandas DataFrame.

        Args:
            round_num (int, optional):
                Number of decimal places to use when rounding metric values.
                Defaults to 2.
            return_ci (bool, optional):
                If True, returns confidence intervals (e.g., "0.85 ± 0.02")
                as strings. If False, returns aggregated numeric values.
                Defaults to False.
            alpha (float, optional):
                Significance level for calculating the margin of error
                in confidence intervals. Supported values are 0.05, 0.01, and 0.001.
                Defaults to 0.05.

        Returns:
            pd.DataFrame: DataFrame with metrics structured such that:
                - Rows correspond to feature extractors
                - Columns correspond to tasks and metrics
                - Each cell contains the metric value
                (numeric or string based on `return_ci`)
        """
        traversal_list = utils.nested_dict_to_list(
            self.get_result_dict(
                round_num=round_num,
                return_ci=return_ci,
                alpha=alpha
            ))
        result_dict = {}

        for task, fe, name, value in traversal_list:
            if f"{task}||{name}" in result_dict:
                result_dict[f"{task}||{name}"][fe] = value
            else:
                result_dict[f"{task}||{name}"] = {fe: value}

        df = pd.DataFrame(result_dict)

        column_tuples = [(col.split("||")[0], col.split("||")[1]) for col in df.columns]
        df.columns = pd.MultiIndex.from_tuples(column_tuples, names=["Task", "Metric"])
        df.index = pd.MultiIndex.from_product([["Extractor"], df.index])
        return df

    def display_result(
        self,
        round_num: int = 2,
        return_ci: bool = False,
        alpha: float = 0.05,
        colored: bool = True
    ) -> None:
        """Display the calculated metrics DataFrame in HTML format.

        Renders the metrics table as HTML, optionally with color
        highlighting for best and worst values.

        Args:
            round_num (int, optional):
                Number of decimal places to use when rounding metric values.
                Defaults to 2.
            return_ci (bool, optional):
                If True, displays confidence intervals (e.g., "0.85 ± 0.02")
                as strings. If False, shows aggregated numeric values.
                Defaults to False.
            alpha (float, optional):
                Significance level for calculating the margin of error
                in confidence intervals. Supported values are 0.05, 0.01, and 0.001.
                Defaults to 0.05.
            colored (bool, optional):
                If True, applies color styling to highlight the best values
                in green and the worst values in red for each metric.
                Defaults to True.
        """
        df = self.get_result_df(
            round_num=round_num,
            return_ci=return_ci,
            alpha=alpha
        )

        return utils.display_styler(
            df=df,
            round_num=round_num,
            ci=return_ci,
            colored=colored
        )
