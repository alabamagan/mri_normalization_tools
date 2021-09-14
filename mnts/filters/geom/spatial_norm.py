import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Union, Tuple
from ..mnts_filters import MNTSFilter


__all__ = ['SpatialNorm']

class SpatialNorm(MNTSFilter):
    r"""
    This class utilize the SimpleITK filter `ResampleImageFilter` to change the spacing. All other factors remains
    unchanged. However, note that the floating point rounding might results in slightly different image dimension, so
    this filter should be used with the cropping filter if you require uniform data size.

    Attributes:
        out_spacing (float, tuple of floats):
            Desired uniform spacing. Unit is mm. Use 0 or negative values if spacing is to be kept

    """
    def __init__(self,
                 out_spacing: Union[float, Tuple[float, float, float]] = None,
                 interpolation_method: str = 'linear'):
        super(SpatialNorm, self).__init__()
        self.out_spacing = out_spacing
        self._interpolation_names = {
            'linear': sitk.sitkLinear,
            'bspline': sitk.sitkBSpline,
            'nearest': sitk.sitkNearestNeighbor,
        }
        self._interpolation = self._interpolation_names[interpolation_method]


    @property
    def out_spacing(self):
        return self._out_spacing

    @out_spacing.setter
    def out_spacing(self, out_spacing: Union[float, Tuple[float, float, float]]):
        self._out_spacing = out_spacing if isinstance(out_spacing, (list, tuple)) else [out_spacing]*3

    @property
    def interpolation(self):
        return self._interpolation_names[self._interpolation]

    @interpolation.setter
    def interpolation(self, val):
        if isinstance(val, str):
            self._interpolation = self._interpolation_names.get(val, 'linear')
        else:
            self._interpolation = val

        if not self._interpolation in self._interpolation_names.values():
            raise IndexError(f"Incorrect interpolation scheme specified, possible options are:"
                             f"{list(self._interpolation_names.keys())}, , got '{val}' instead.")


    def filter(self,
               input: Union[str, Path, sitk.Image]
               ):
        input = self.read_image(input)

        original_size = np.asarray(input.GetSize())
        original_spacing = np.asarray(input.GetSpacing())

        # Keep spacing if there's a negative value in out_spacing
        new_spacing = np.asarray(self.out_spacing)
        new_spacing[new_spacing <= 0] = original_spacing[new_spacing <=0]

        new_size = np.round((original_size * original_spacing) / new_spacing).astype('int').tolist()
        self._logger.info(f"From {original_size} -> {new_size}")

        # For segmentation, force to use nearest neighbor to avoid funny results.
        if input.GetPixelID() == sitk.sitkUInt8:
            self._interpolation = sitk.sitkNearestNeighbor

        f = sitk.ResampleImageFilter()
        f.SetReferenceImage(input)
        f.SetOutputSpacing(new_spacing.tolist())
        f.SetSize(new_size)
        f.SetInterpolator(self._interpolation)
        out = f.Execute(input)
        return out

