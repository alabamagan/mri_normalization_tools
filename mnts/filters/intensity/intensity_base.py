r"""
This class is for identifying intensity filters such that segmentation labels won't be processed by
them.
"""
import SimpleITK as sitk

class MNTSIntensityBase(object):
    def __init__(self):
        super(MNTSIntensityBase, self).__init__()
        self._ignore_seg = True

    @property
    def ignore_seg(self):
        return True

    def filter(self, *args):
        if args[0].GetPixelID() == sitk.sitkUInt8:
            return args[0]
        else:
            return self._filter(*args)

    def __call__(self, *args, **kwargs):
        return self.filter(*args, **kwargs)