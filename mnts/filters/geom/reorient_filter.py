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
        self.target_orientation = target_orientation

    @property
    def target_orientation(self):
        self._target_orientation

    @target_orientation.setter
    def target_orientation(self, val:str):
        assert type(val), "Input must be string of three characters"
        assert re.match(r"(?i)[railps]{3}", val) is not None, "Input must be string of three characters"
        self._target_orientation = val

    def filter(self,
               image: Union[str, sitk.Image],
               mask: Union[str, sitk.Image] = None):
        """
        Apply DICOMOirent filter to both inputs
        """
        image = self.read_image(image)
        mask = self.read_image(mask) if mask is not None else None

        ori_filter = sitk.DICOMOrientImageFilter()
        ori_filter.SetDesiredCoordinateOrientation(self._target_orientation)

        if mask is not None:
            return (ori_filter.Execute(image), ori_filter.Execute(mask))
        else:
            return ori_filter.Execute(image)
