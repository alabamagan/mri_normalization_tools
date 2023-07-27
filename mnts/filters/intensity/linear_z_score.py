import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List
from .linear_rescale import LinearRescale

__all__ = ['ZScoreNorm']

class ZScoreNorm(LinearRescale):
    r"""
    Z-score normalization, which is a special case of linear rescaling the intensity to a mean of 0 and standard
    deviation of 1.

    .. math::
        z = \frac{x-\mu}{\sigmal}

    .. note ::
        This filter changes the datatype of the image to float32 to retain the resolution of the
        input data.
    """
    def __init__(self):
        super(ZScoreNorm, self).__init__(0, 1.)

    def _filter(self,
               input: Union[str, Path, sitk.Image],
               mask: Union[str, Path, sitk.Image] = None):
        input = self.read_image(input)
        mask = self.read_image(input)

        if not mask is None:
            np_im, np_mask = [sitk.GetArrayFromImage(x) for x in [input, mask]]
            input_mean = np_im[np_mask != 0].mean()
            input_std  = np_im[np_mask != 0].std()
        else:
            f = sitk.StatisticsImageFilter()
            f.Execute(input)
            input_mean = f.GetMean()
            input_std = f.GetVariance() ** .5

        # Cast to float if not float already.
        if input.GetPixelID() not in [8, 9]:
            self._logger.warning("Casting the input to float32.")
            input = sitk.Cast(input, sitk.sitkFloat32)
        input = (input - input_mean) / input_std
        return input