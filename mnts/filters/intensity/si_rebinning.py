import SimpleITK as sitk
from ..mnts_filters import MNTSFilter
from .intensity_base import MNTSIntensityBase
from . import RangeRescale

from pathlib import Path
from typing import Union, List

__all__ = ['SignalIntensityRebinning']

class SignalIntensityRebinning(MNTSIntensityBase, MNTSFilter):
    r"""
    Rebin the image into desired number of bins (2 to 65536) and than cast the image into sitkUInt16 type. This is done
    by first rescaling the image to the range 0 to (`num_of_bins` - 1) and than casting it into UInt16. Rescaling is
    done by :class:`RangeRescale` with no quantile setting, if you need to remove outliers, you can specify quantile
    like do in :class:`RangeRescale` filter.

    If you have already rescaled your image, you can perform rebin your image with sitk.Cast along and won't need this
    filter.

    Attributes:
         num_of_bins (int):
            Number of bins to rescale the image into.
        quantiles (list of loat):
            See :class:`RangeRescale`
    """
    def __init__(self,
                 num_of_bins: int = 64,
                 quantiles: List[float] = None):
        super(SignalIntensityRebinning, self).__init__()
        self._rrescale = RangeRescale()
        self.num_of_bins = num_of_bins
        self.quantiles = quantiles

    @property
    def num_of_bins(self):
        return self._num_of_bins

    @property
    def quantiles(self):
        return self._quantiles

    @quantiles.setter
    def quantiles(self, val):
        self._rrescale.quantiles = val

    @num_of_bins.setter
    def num_of_bins(self, val):
        assert 2 <= val <= 65536, f"Number of bins too small or too large: {val}, must be at least 2 and " \
                                  f"smaller than or equal to 65536."
        self._num_of_bins = int(val)
        self._rrescale.min = 0
        self._rrescale.max = self._num_of_bins - 1

    @property
    def skip_rescale(self):
        return self._skip_rescale

    def _filter(self,
                input: Union[str, Path, sitk.Image]) -> sitk.Image:
        input = self.read_image(input)
        input = self._rrescale(input)

        # images are always casted to UInt16 because UInt8 are considered segmentations in both sitk and this package.
        return sitk.Cast(input, sitk.sitkUInt16)


