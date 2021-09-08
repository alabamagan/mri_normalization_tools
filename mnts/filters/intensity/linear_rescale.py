import SimpleITK as sitk
import numpy as np
from typing import Union, Tuple, List
from ..mnts_filters import MNTSFilter

__all__ = ['LinearRescale', 'ZScoreNorm', 'RangeRescale']

class LinearRescale(MNTSFilter):
    r"""
    Description:
        This class rescale the input's linearly to match the desired mean and standard deviation following:
        .. math::
            z = \left( \frac{x - \mu}{\sigma}\right) \times \sigma'+\mu'

    Attributes:
        mean (float):
            Target mean value.
        std (float):
            Target std value.

    """
    def __init__(self,
                 mean: float = None,
                 std: float = None):
        self._mean = mean
        self._std = std

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = float(mean)

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, std):
        self._std = float(std)

    def filter(self, input):
        super(LinearRescale, self).filter(input)
        f = sitk.StatisticsImageFilter()
        f.Execute(input)

        input_mean = f.GetMean()
        input_std = f.GetVariance()**.5

        # Use sitk function for overflow/underflow protection
        input = sitk.ShiftScale(input, -input_mean, 1./input_std)
        input = sitk.ShiftScale(input, 0, self._std)
        input = sitk.ShiftScale(input, self.mean, 1)
        return input


class ZScoreNorm(LinearRescale):
    r"""
    Z-score normalization, which is a special case of linear rescaling the intensity to a mean of 0 and standard
    deviation of 1.

    .. math::
        z = \frac{x-\mu}{\sigmal}
    """
    def __init__(self):
        super(ZScoreNorm, self).__init__(0, 1.)

    def filter(self, input):
        super(LinearRescale, self).filter(input)
        f = sitk.StatisticsImageFilter()
        f.Execute(input)

        input_mean = f.GetMean()
        input_std = f.GetVariance()**.5

        # Use sitk function for overflow/underflow protection
        input = sitk.ShiftScale(input, -input_mean, 1./input_std)
        return input


class RangeRescale(MNTSFilter):
    r"""
    Rescale the image to the given range.

    Attributes:
        min (float)
        max (float)
        quantiles (List[float, float])
    """
    def __init__(self,
                 min: float = None,
                 max: float = None,
                 quantiles: Union[None, Tuple[float, float], List[float, float]] = None):
        super(RangeRescale, self).__init__()
        self._min = min
        self._max = max
        self._quantiles = quantiles

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def quantiles(self):
        return self._quantiles

    @quantiles.setter
    def quantiles(self, lower, upper):
        if lower < upper:
            self._logger.warning("Lower quartile must be smaller than upper quartile, reversing the order to "
                                 f"[{upper} -> {lower}].")
        if not (0 <= lower < upper <= 1):
            self._logger.error("Quartile ranges must be within the range 0 to 1. Lower and upper quartile must "
                               "not be identical")
            return
        self._quantiles = (lower, upper)

    def filter(self, input):
        if not self._quantiles is None:
            l, u = np.quantile(dat.flatten(), self._quantiles)
            input = sitk.Clamp(input, l, u)
        input = sitk.RescaleIntensity(self.min, self.max)
        return input
