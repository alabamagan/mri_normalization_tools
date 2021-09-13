import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List
from ..mnts_filters import MNTSFilter
from .intensity_base import MNTSIntensityBase

__all__ = ['RangeRescale']

class RangeRescale(MNTSIntensityBase, MNTSFilter):
    r"""
    Rescale the image to the given range.

    Attributes:
        min (float):
            Target min value.
        max (float)
            Target max value.
        quantiles (List[float, float], Optional)
            Whether to trim some percentage of values on the histogram before the rescaling.
    """
    def __init__(self,
                 min: float = None,
                 max: float = None,
                 quantiles: Union[None, Tuple[float], List[float]] = None):
        super(RangeRescale, self).__init__()
        self._min = min
        self._max = max
        self._quantiles = quantiles

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, val):
        self._min = float(val)

    @property
    def max(self):
        return self._max

    @property
    def max(self, val):
        self._max = float(val)

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

    def _filter(self,
                input: Union[str, Path, sitk.Image],
                mask: Union[str, Path, sitk.Image] = None):
        input = self.read_image(input)
        mask = self.read_image(input)
        if mask is not None:
            np_im, np_mask = [sitk.GetArrayFromImage(x) for x in [input, mask]]
            if not self._quantiles is None:
                l, u = np.quantile(np_im[np_mask != 0].flatten(), self._quantiles)
                input = sitk.Clamp(input, input.GetPixelID(), l, u)
            input = sitk.RescaleIntensity(input, self.min, self.max)
            return input
        else:
            dat = sitk.GetArrayFromImage(input)
            if not self._quantiles is None:
                l, u = np.quantile(dat.flatten(), self._quantiles)
                input = sitk.Clamp(input, input.GetPixelID(), l, u)
            input = sitk.RescaleIntensity(input, self.min, self.max)
            return input
