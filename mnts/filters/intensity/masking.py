import SimpleITK as sitk
from typing import Union, Tuple
from ..mnts_filters import MNTSFilter

__all__ = ['OtsuTresholding']

class OtsuTresholding(MNTSFilter):
    r"""
    This filter creates a foreground mask using the Otsu thresholding followed by binary closing and then hole filling.
    """
    def __init__(self):
        super(OtsuTresholding, self).__init__()

    def filter(self, input):
        outmask = sitk.OtsuThreshold(input, 0, 1) # For some reason the official implementation reverse this
        outmask = sitk.Cast(outmask, sitk.sitkUInt8)
        outmask = sitk.BinaryMorphologicalClosing(outmask, [2, 2, 2]) #TODO: check kernel size
        outmask[:, :, 0] = sitk.BinaryFillhole(outmask[:,:,0])
        outmask[:, :, -1] = sitk.BinaryFillhole(outmask[:,:,-1])
        outmask = sitk.BinaryFillhole(outmask, fullyConnected=True)
        return outmask
