import SimpleITK as sitk
import numpy as np
import re
from pathlib import Path
from typing import Union, Tuple, Optional
from ..mnts_filters import MNTSFilter


__all__ = ['ReorientFilter']

class ReorientFilter(MNTSFilter):
    r"""
    A wrapper for sitk.DICOMOrient

    Attributes:
        target_orientation (str):
            A three character string representing orientation code. E.g., 'LPS'.
    """
    def __init__(self, target_orientation: str = 'RAI'):
        super(ReorientFilter, self).__init__()
        self.target_orientation = target_orientation

    @property
    def target_orientation(self):
        return self._target_orientation

    @target_orientation.setter
    def target_orientation(self, val: str):
        assert isinstance(val, str), "Input must be a string of three characters"
        assert re.fullmatch(r"(?i)[railps]{3}", val) is not None, \
            f"Input must be a three-character orientation code using R/L, A/P, I/S, got '{val}'"
        self._target_orientation = val.upper()

    def filter(self,
               image: Union[str, sitk.Image],
               mask: Union[str, sitk.Image] = None):
        """
        Apply DICOMOrient filter to both inputs
        """
        image = self.read_image(image)
        mask = self.read_image(mask) if mask is not None else None

        ori_filter = sitk.DICOMOrientImageFilter()
        ori_filter.SetDesiredCoordinateOrientation(self._target_orientation)

        if mask is not None:
            return (ori_filter.Execute(image), ori_filter.Execute(mask))
        else:
            return ori_filter.Execute(image)
