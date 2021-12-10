r"""
This class is for identifying intensity filters such that segmentation labels won't be processed by
them.
"""
import SimpleITK as sitk
from pathlib import Path
from typing import Union

class MNTSIntensityBase(object):
    """
    This class marks intensity transform and any filters that inherit this class (before inheriting
    MNTSFitler will skip label images. Label images are identified as images with pixel type UInt8
    or LabelUInt8 or LabelUInt16.

    .. note::
        Don't inherit this class if you want the filter to operate on the specified pixel types


    """
    def __init__(self):
        super(MNTSIntensityBase, self).__init__()
        self._ignore_seg = True

    @property
    def ignore_seg(self):
        return self._ignore_seg

    def filter(self, *args):
        args = [self.try_read_image(a) for a in args]
        self._logger.debug(f"{args[0].GetPixelID()}")
        if (args[0].GetPixelID() in [sitk.sitkUInt8,
                                    sitk.sitkLabelUInt8,
                                    sitk.sitkLabelUInt16]) \
                & self._ignore_seg:
            self._logger.info("Skipping intensity transforms for "
                              "label images with UInt8, LabelUInt8 or LabelUInt16...")
            return args[0]
        else:
            return self._filter(*args)

    @staticmethod
    def try_read_image(input: Union[str, Path, sitk.Image]):
        try:
            if isinstance(input, (str, Path)):
                input = sitk.ReadImage(str(input))
        except:
            pass
        return input

    def __call__(self, *args, **kwargs):
        return self.filter(*args, **kwargs)