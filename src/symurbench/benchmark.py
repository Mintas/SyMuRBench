"""Benchmark class implementtion."""
import logging

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

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger("benchmark")
np.random.seed(SEED)  # noqa: NPY002
torch.manual_seed(SEED)
torch.set_num_threads(NUM_THREADS)

class Benchmark:
    """Class for benchmarking different feature extractors."""

    def __init__(
        self,
        feature_extractors_list: list[FeatureExtractor | PersistentFeatureExtractor],
        tasks: list[str] | list[AbsTask] | None = None,
    ) -> None:
        """
        Initialize benchmark.

        Args:
            feature_extractors_list (list[FeatureExtractor|PersistentFeatureExtractor]):
                list with FeatureExtractor or PersistentFeatureExtractor
                objects to compare.
            tasks (list[str] | list[AbsTask] | None):
                list of task names or list of task objects.
                Defaults to None.
        """
        self.feature_extractors = feature_extractors_list
        self.validate_feature_extractors()
        self.tasks = self.validate_tasks_argument(tasks)

        log_msg = f"Metadata is loaded for {len(self.tasks)} tasks."
        logger.info(log_msg)

        # dict where calculated metrics are saved
        self.metrics = {t.name:{} for t in self.tasks}

    def validate_tasks_argument(
        self,
        tasks: list[str] | list[AbsTask] | None,
    ) -> list[AbsTask]:
        """
        Check if tasks argument is valid.

        Args:
            tasks (list[str] | list[AbsTask] | None):
                list of task names or list of task instances
        Raises:
            TypeError: `tasks` is not a list or len(tasks) == 0
            TypeError: `tasks` elements have wrong types
            ValueError: names of the tasks are not unique

        Returns:
            list[AbsTask]: list of task instances
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
        """
        Check if feature_extractors_list argument is valid.

        Raises:
            TypeError: `self.feature_extractors` is not a list
                or len(self.feature_extractors) == 0
                or `self.feature_extractors` contains elements of wrong types.
            ValueError: names of feature extractors in `self.feature_extractors`
                are not unique.
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
            list[str]: list with task names
        """
        return [t.name for t in self.tasks]

    @classmethod
    def get_tasks(
        cls,
        task_names: list[str] | None,
        init_tasks: bool = True
    ) -> list[ClassificationTask | RetrievalTask]:
        """
        Load and initialize tasks.

        Args:
            task_names (list[str] | None): list of tasks names to load
            init_tasks (bool, optional):
                flag, if True, function returns list of objects,
                if False, function returns list of classes.
                Defaults to True.

        Returns:
            list[ClassificationTask | RetrievalTask]:
                list of task objects or list of task classes
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
        """
        Initialize benchmark from config.

        Args:
            feature_extractors_list (list[FeatureExtractor|PersistentFeatureExtractor]):
                list with FeatureExtractor and/or PersistentFeatureExtractor objects
                to compare
            tasks_config (dict):
                dict with config.
                Config has the following structure:
                1) key - task name;
                2) value - dict with arguments for task name.

                Config arguments include:
                1) arguments for MetaLoader;
                2) arguments for __init__() method of the class

                Example of config can be found in configs/tasks_config.yaml

        Returns:
            Benchmark class instance object
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
        """
        Initialize benchmark from YAML config file.

        Args:
            feature_extractors_list (list[FeatureExtractor|PersistentFeatureExtractor]):
                list with FeatureExtractor and/or PersistentFeatureExtractor
                objects to compare
            tasks_config_path (str): path to YAML file with config.
                Config has the following structure:
                1) key - task name;
                2) value - dict with arguments for task name.

                Config arguments include:
                1) arguments for MetaLoader;
                2) arguments for __init__() method of the class

                Example of config can be found in configs/tasks_config.yaml
        Raises:
            ValueError: if config is empty
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
        """
        Run single task for feature extractor.

        Save calculated metrics.

        Args:
            task (ClassificationTask  |  RetrievalTask]): task to run
            feature_extractor (FeatureExtractor  |  PersistentFeatureExtractor]):
                feature extractor to evaluate
        """
        self.metrics[task.name][feature_extractor.name] = task.run(feature_extractor)

    def run_all_tasks(
        self
    ) -> None:
        """Run all tasks for all feature extractor objects."""
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
        """
        Aggregate self.metrics into a human-readable dict.

        Args:
            round_num (int, optional):
                The number of decimals to use when rounding the number.
                Defaults to 2.
            return_ci (bool, optional):
                If True, the confidence interval is returned, otherwise
                the aggregated value is returned. Defaults to False.
            alpha (float, optional): the significance level
                for calculating margin of error. Supported values: 0.05, 0.01, 0.001.
                Defaults to 0.05.

        Returns:
            dict: dict with the following structure:
                {"task name":
                {"feature extractor name":
                {"metric name": metric value}
                }}
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
        """
        Aggregate self.metrics into a pandas.DataFrame.

        Args:
            round_num (int, optional):
                The number of decimals to use when rounding the number.
                Defaults to 2.
            return_ci (bool, optional):
                If True, the confidence interval is returned, otherwise
                the aggregated value is returned. Defaults to False.
            alpha (float, optional): the significance level
                for calculating margin of error. Supported values: 0.05, 0.01, 0.001.
                Defaults to 0.05.

        Returns:
            pd.DataFrame: pd.DataFrame with calculated metrics
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
        """
        Display dataframe with calculated metrics in HTML format.

        Args:
            round_num (int, optional):
                The number of decimals to use when rounding the number.
                Defaults to 2.
            return_ci (bool, optional):
                If True, the confidence interval is returned, otherwise
                the aggregated value is returned. Defaults to False.
            alpha (float, optional): the significance level
                for calculating margin of error. Supported values: 0.05, 0.01, 0.001.
                Defaults to 0.05.
            colored (bool, optional):
                If True, highlight the best values with green color
                and the worst values with red color.
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
