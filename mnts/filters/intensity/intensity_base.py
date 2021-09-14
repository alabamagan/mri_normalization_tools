r"""
This class is for identifying intensity filters such that segmentation labels won't be processed by
them.
"""
import SimpleITK as sitk
from pathlib import Path
from typing import Union

class MNTSIntensityBase(object):
    def __init__(self):
        super(MNTSIntensityBase, self).__init__()
        self._ignore_seg = True

    @property
    def ignore_seg(self):
        return True

    def filter(self, *args):
        args = [self.try_read_image(a) for a in args]
        if args[0].GetPixelID() == sitk.sitkUInt8:
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