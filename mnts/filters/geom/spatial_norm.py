import SimpleITK as sitk
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
            Desired uniform spacing. Unit is mm.

    """
    def __init__(self, out_spacing: Union[float, Tuple[float, float, float]] = None):
        self.out_spacing = out_spacing

    @property
    def out_spacing(self):
        return self._out_spacing

    @out_spacing.setter
    def out_spacing(self, out_spacing: Union[float, Tuple[float, float, float]]):
        self._out_spacing = out_spacing if isinstance(out_spacing, (list, tuple)) else [out_spacing]*3


    def filter(self, input):
        super(SpatialNorm, self).filter(input)
        f = sitk.ResampleImageFilter()
        f.SetReferenceImage(input)
        f.SetOutputSpacing(self._out_spacing)
        return f.Execute(input)

