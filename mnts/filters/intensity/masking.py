import SimpleITK as sitk
from pathlib import Path
from typing import Union, Tuple
from ..mnts_filters import MNTSFilter
from .intensity_base import MNTSIntensityBase

__all__ = ['OtsuTresholding']

class OtsuTresholding(MNTSIntensityBase, MNTSFilter):
    r"""
    This filter creates a foreground mask using the Otsu thresholding followed by binary closing and then hole filling.

    .. note::
        This class will return binary mask with datatype UInt8
    """
    def __init__(self):
        super(OtsuTresholding, self).__init__()
        self._closing_kernel_size = [2, 2, 2]

    @property
    def closing_kernel_size(self):
        return self._closing_kernel_size

    @closing_kernel_size.setter
    def closing_kernel_size(self, val):
        return self._closing_kernel_size

    def _filter(self,
                input: Union[str, Path, sitk.Image]) -> sitk.Image:
        input = self.read_image(input)
        outmask = sitk.OtsuThreshold(input, 0, 1) # For some reason the official implementation find background
                                                  # instead of foreground, but maybe I am wrong.
        outmask = sitk.Cast(outmask, sitk.sitkUInt8)
        outmask = sitk.BinaryMorphologicalClosing(outmask, [2, 2, 2]) #TODO: check kernel size
        outmask[:, :, 0] = sitk.BinaryFillhole(outmask[:,:,0])
        outmask[:, :, -1] = sitk.BinaryFillhole(outmask[:,:,-1])
        outmask = sitk.BinaryFillhole(outmask, fullyConnected=True)
        return outmask
