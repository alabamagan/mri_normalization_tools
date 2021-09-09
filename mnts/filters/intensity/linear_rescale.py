import SimpleITK as sitk
import numpy as np
from typing import Union, Tuple, List
from ..mnts_filters import MNTSFilter
from .intensity_base import MNTSIntensityBase

__all__ = ['LinearRescale']

class LinearRescale(MNTSIntensityBase, MNTSFilter):
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
        super(LinearRescale, self).__init__()
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

    def _filter(self, input, mask = None):
        if mask is None:
            np_im, np_mask = [sitk.GetArrayFromImage(x) for x in [input, mask]]

            input_mean = np_im[np_mask != 0].mean()
            input_std  = np_im[np_mask != 0].max()
        else:
            f = sitk.StatisticsImageFilter()
            f.Execute(input)
            input_mean = f.GetMean()
            input_std = f.GetVariance() ** .5

        # !!!TODO: Overflow protection
        # Cast to float if not float already.
        if input.GetPixelID() not in [8, 9]:
            self._logger.warning("Casting the")
            input = sitk.Cast(input, sitk.sitkFloat32)
        input = ((input - input_mean) / input_std + self._mean) * self._std
        return input

