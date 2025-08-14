"""Implementation of a container class for calculated metrics."""
import numpy as np

AGGREGATION_FUNC = {
    "mean": np.mean,
    "median": np.median,
    "std": np.std
}

Z_ALPHA_OVER_TWO = {
    0.05: 1.96,
    0.01: 2.576,
    0.001: 3.291
}

class MetricValue:
    """
    Container for storing and managing calculated evaluation metrics.

    This class holds metric values resulting from the evaluation.
    It provides a structured way to access, manage, and export metrics
    for reporting or further analysis.
    """
    def __init__(
        self,
        name: str,
        values: list[float|int],
        aggregate: str = "mean"
    ) -> None:
        """Initialize metric values.

        Args:
            name (str): Name of the metric.
            values (list[float | int]): List of metric values. Must not be empty.
            aggregate (str, optional): Aggregation method for the list of values.
                Supported values are "mean" and "median". Defaults to "mean".
        """
        self._name = name
        self._values = values
        self.aggregation_type = aggregate
        self.validate()

    @property
    def values(
        self,
    ) -> list[float|int]:
        """Get the raw metric values.

        Returns:
            list[float | int]: The unaggregated list of metric values.
        """
        return self._values

    @property
    def name(
        self
    ) -> str:
        """Get the name of the metric.

        Returns:
            str: The name of the metric.
        """
        return self._name

    @name.setter
    def name(
        self,
        name: str
    ) -> None:
        """Set the name of the metric.

        Args:
            name (str): The new name for the metric.
        """
        self._name = name

    @property
    def is_single_value(
        self
    ) -> bool:
        """Check if the number of elements in `self.values` is exactly one.

        Returns:
            bool: True if the length of `self.values` is 1, False otherwise.
        """
        return len(self._values) == 1

    def validate(
        self
    ) -> None:
        """Validate the metric's name, values, and aggregation type.

        Raises:
            TypeError: If `self._name` is not a string.
            TypeError: If `self._values` is not a list or is empty.
            TypeError: If any element in `self._values` is not an int or float.
            ValueError: If `self.aggregation_type` is not one of {"mean", "median"}.
            ValueError: If `self.alpha` is not one of [0.05, 0.01, 0.001].
        """
        if not isinstance(self._name, str):
            msg = "Name should be a string."
            raise TypeError(msg)

        if not isinstance(self._values, list)\
        or len(self._values) == 0:
            msg = "`values` should be a non-empty list."
            raise TypeError(msg)

        values_float = {isinstance(val, float) for val in self._values}
        values_int = {isinstance(val, int) for val in self._values}

        if values_float | values_int == {False}:
            msg = "Values should contain only ints/floats."
            raise TypeError(msg)

        if self.aggregation_type not in ("mean", "median"):
            msg = "`aggregation_type` takes only 2 values (`mean` or `median`)"
            raise ValueError(msg)

    def _get_agg_value(
        self,
        round_num: int = 2
    ) -> float|int:
        """Return a rounded aggregated value for `self.values`.

        Args:
            round_num (int, optional):
                Number of decimal places to round to. Defaults to 2.

        Returns:
            float | int:
                The aggregated value (e.g., mean or median) of the metric values,
                rounded to the specified number of decimals.
        """
        if self.is_single_value:
            return round(self._values[0], round_num)

        return round(AGGREGATION_FUNC[self.aggregation_type](self._values), round_num)

    def _get_margin_of_error(
        self,
        round_num: int = 2,
        alpha: float = 0.05
    ) -> float:
        """Return a rounded margin of error for `self.values`.

        Args:
            round_num (int, optional):
                Number of decimal places to round the margin of error to.
                Defaults to 2.
            alpha (float, optional):
                Significance level for confidence interval calculation.
                Supported values are 0.05, 0.01, and 0.001. Defaults to 0.05.

        Returns:
            float:
                The rounded margin of error if the length of `self.values`
                is greater than 1;
                returns 0.0 if there is only one value (no variability).

        Raises:
            ValueError: If `alpha` is not one of [0.05, 0.01, 0.001].
        """
        if self.is_single_value:
            return 0.0

        if alpha not in Z_ALPHA_OVER_TWO:
            msg = "Please set alpha to one of these standard values: 0.05, 0.01, 0.001."
            raise ValueError(msg)

        std = AGGREGATION_FUNC["std"](self._values)
        n_folds = len(self._values)
        moe = Z_ALPHA_OVER_TWO[alpha]

        return round(moe * (std/np.sqrt(n_folds)), round_num)

    def _get_ci_as_str(
        self,
        round_num: int = 2,
        alpha: float = 0.05
    ) -> str:
        """Return a rounded confidence interval as a formatted string.

        Example:
            "0.522 ± 0.012"

        If there is only one value in `self.values`,
        only the aggregated value is returned (e.g., "0.522").

        Args:
            round_num (int, optional): Number of decimal places to use when rounding the
                aggregated value and margin of error. Defaults to 3.
            alpha (float, optional):
                Significance level for calculating the margin of error.
                Supported values are 0.05, 0.01, and 0.001. Defaults to 0.05.

        Returns:
            str: Formatted string representing the confidence interval or single value.

        Raises:
            ValueError: If `alpha` is not one of [0.05, 0.01, 0.001].
        """
        value_as_str = str(self._get_agg_value(round_num))

        if not self.is_single_value:
            value_as_str += " ± " + str(self._get_margin_of_error(round_num, alpha))

        return value_as_str

    def get_agg_value(
        self,
        round_num: int = 2,
        return_ci: bool = False,
        alpha: float = 0.05
    ) -> float|int|str:
        """
        Return a rounded aggregated value or confidence interval.

        Unites 2 methods: _get_agg_value and _get_agg_std_as_str.

        Args:
            round_num (int, optional):
                The number of decimals to use when rounding aggregated value
                and margin of error. Defaults to 3.
            return_ci (bool, optional):
                If True, the confidence interval is returned, otherwise
                the aggregated value is returned. If True, the function returns
                the str object. Float or int otherwise. Defaults to False.
            alpha (float, optional): the significance level
                for calculating margin of error. Supported values: 0.05, 0.01, 0.001.
                Defaults to 0.05.

        Returns:
            float|int|str: Aggregated values or confidence interval.
        """
        """Return a rounded aggregated value or confidence interval.

        This method combines the functionality of `_get_agg_value`
        and `_get_agg_std_as_str` to provide either a single aggregated value
        or a formatted confidence interval string.

        Args:
            round_num (int, optional):
                Number of decimal places to use when rounding the
                aggregated value and margin of error. Defaults to 3.
            return_ci (bool, optional):
                If True, returns the confidence interval as a string
                (e.g., "0.522 ± 0.012").
                If False, returns the aggregated value as a float or int.
                Defaults to False.
            alpha (float, optional):
                Significance level for calculating the margin of error.
                Supported values are 0.05, 0.01, and 0.001. Defaults to 0.05.

        Returns:
            float | int | str:
                The aggregated value (float or int) if `return_ci` is False,
                otherwise the confidence interval as a string.

        Raises:
            ValueError: If `alpha` is not one of [0.05, 0.01, 0.001].
        """
        if return_ci:
            return self._get_ci_as_str(round_num, alpha)
        return self._get_agg_value(round_num)

    def __add__(
        self,
        other: None
    ) -> None:
        """Overload the `+` operator to combine two MetricValue objects.

        The aggregation type of the resulting object is set to that
        of the first operand.
        Both objects must have the same metric name.

        Args:
            other (MetricValue): Another MetricValue object to add.
                Must have the same name.

        Returns:
            MetricValue: A new MetricValue object containing the combined values.

        Raises:
            ValueError: If the metric names of `self` and `other` do not match.
        """
        if self.name != other.name:
            msg = "Only identical metrics can be added together."
            raise ValueError(msg)

        return MetricValue(
            name = self.name,
            values = self._values+other._values,
            aggregate = self.aggregation_type
        )
