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
    """Container for calculated metrics."""
    def __init__(
        self,
        name: str,
        values: list[float|int],
        aggregate: str = "mean"
    ) -> None:
        """
        Initialize metric values.

        Args:
            name (str): metric name
            values (list[float | int]): list of values. Cannot be empty.
            aggregate (str, optional): aggregation type for list of values.
                Defaults to "mean". All variants: "median", "mean".
        """
        self._name = name
        self._values = values
        self.aggregation_type = aggregate
        self.validate()

    @property
    def values(
        self,
    ) -> list[float|int]:
        """Raw values."""
        return self._values

    @property
    def name(
        self
    ) -> str:
        """Name."""
        return self._name

    @name.setter
    def name(
        self,
        name: str
    ) -> None:
        """Change self.name to new name.

        Args:
            name (str): new name
        """
        self._name = name

    @property
    def is_single_value(
        self
    ) -> bool:
        """
        Check number of elements in `self.values`.

        Returns:
            bool: True if `self.values` length equals 1, False otherwise
        """
        return len(self._values) == 1

    def validate(
        self
    ) -> None:
        """
        Validate name, values, aggregation_type and alpha.

        Raises:
            TypeError: if self._name is not a string
            TypeError: if self._values is not a list / an empty list
            TypeError: if self._values contains data types other than int and float
            ValueError: if self.aggregation_type not in ("mean", "median")
            ValueError: if self.alpha not in [0.05, 0.01, 0.001]
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
        """
        Return a rounded aggregated value for `self.values`.

        Args:
            round_num (int, optional):
                The number of decimals to use when rounding the aggregated value.
                Defaults to 2.

        Returns:
            float|int: aggregated value
        """
        if self.is_single_value:
            return round(self._values[0], round_num)

        return round(AGGREGATION_FUNC[self.aggregation_type](self._values), round_num)

    def _get_margin_of_error(
        self,
        round_num: int = 2,
        alpha: float = 0.05
    ) -> float:
        """
        Return a rounded margin of error for `self.values`.

        Args:
            round_num (int, optional):
                The number of decimals to use when rounding the std value.
                Defaults to 2.
            alpha (float, optional): the significance level
                for calculating margin of error. Supported values: 0.05, 0.01, 0.001.
                Defaults to 0.05.

        Returns:
            float: rounded margin of error for values if `self.values` length > 1
                else 0.0
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
        """
        Return a rounded confidence interval as a string.

        Result example: `0.522 ± 0.012`.
        If there is only one element in `self.values`,
        only the aggregated value is returned.

        Args:
            round_num (int, optional):
                The number of decimals to use when rounding aggregated value
                and margin of error. Defaults to 3.
            alpha (float, optional): the significance level
                for calculating margin of error. Supported values: 0.05, 0.01, 0.001.
                Defaults to 0.05.

        Returns:
            str: confidence interval
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
        if return_ci:
            return self._get_ci_as_str(round_num, alpha)
        return self._get_agg_value(round_num)

    def __add__(
        self,
        other: None
    ) -> None:
        """
        Overload `+` operation.

        The aggregation type is set equal to the aggregation type of the first operand.

        Args:
            other (MetricValue): MetricValue object. Should have the same name.

        Raises:
            ValueError: if object names differ.

        Returns:
            MetricValue: Union of MetricValue objects
        """
        if self.name != other.name:
            msg = "Only identical metrics can be added together."
            raise ValueError(msg)

        return MetricValue(
            name = self.name,
            values = self._values+other._values,
            aggregate = self.aggregation_type
        )
