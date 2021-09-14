import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Optional, Callable
from ..mnts_filters import MNTSFilter
from .intensity_base import MNTSIntensityBase

__all__ = ['OtsuTresholding', 'ThresBinaryClosing', 'HuangThresholding']


class ThresBinaryClosing(MNTSIntensityBase, MNTSFilter):
    r"""
    You can use this method to throw in your own function for image fore-ground segmentation. This class first
    perform binary threshold and then binary morphological closing.

    The results based on my personal experience on head-and-neck images are as follow:

        | Method                      | T1w | T2w | T1w-ce-fs | T1w-ce | T2w-fs |
        |:---------------------------:|:---:|:---:|:---------:|:------:|:------:|
        | HuangThreshold              | ✔   |     | ✔         | ✔      | ✔    |
        | OtsuThreshold               | ✔   |     | ✔         | ✔      | ✘    |
        | IsoDataThreshold            | ✔   |     | ✔         | ✔      | ✘    |
        | MaximumEntropyThreshold     | ✘   |     | ✘         | ✘      | ✘    |
        | MomentsThreshold            | ✔   |     | ✔         | ✔      | ✘    |
        | KittlerIllingworthThreshold | ✘   |     | ✘         | ✘      | ✘    |
        | LiThreshold                 | ✔   |     | ✔         | ✔      | ✔    |
        | ShanbhagThreshold           | ✘   |     | ✘         | ✘      | ✘    |
        | RenyiEntropyThreshold       | ✘   |     | ✘         | ✔      | ✘    |
        | TriangleThreshold           | ✘   |     | ✔         | ✔      | ✔    |
        | YenThreshold                | ✘   |     | ✘         | ✔      | ✘    |

    Attributes:
        closing_kernel_size (list of floats):
            The closing kernel size in mm. The kernel size usded in the binary closing operation after the thresholding.
            It will be rounded to the nearest integer.

    """
    def __init__(self, core_func=None, closing_kernel_size=[5, 5, 5]):
        super(ThresBinaryClosing, self).__init__()
        self.closing_kernel_size = closing_kernel_size
        self.core_func = core_func

    @property
    def closing_kernel_size(self):
        return self._closing_kernel_size

    @closing_kernel_size.setter
    def closing_kernel_size(self, val):
        if not isinstance(val, (list, tuple)):
            val = [val, val, val]
        self._closing_kernel_size = np.asarray(val)

    @property
    def core_func(self):
        return self._core_func

    @core_func.setter
    def core_func(self, val):
        if val is None:
            return
        assert callable(val), "Specified function is not callable!"
        self._core_func = val

    def compute_closing_kernel_size(self, input):
        if len(self.closing_kernel_size) != input.GetDimension():
            msg = f"Specified kernel size is {len(self.closing_kernel_size)}-D but the image is " \
                  f"{input.GetDimension()}-D. Specified kernel: {'×'.join(self.closing_kernel_size)} (mm)."
            raise IndexError(msg)

        # compute closing kernel size in voxel
        im_spacing = np.asarray(input.GetSpacing())
        closing_kernel_size = self._closing_kernel_size / im_spacing
        closing_kernel_size = np.round(closing_kernel_size).astype('int')

        return closing_kernel_size

    def mask_closing(self,
                     closing_kernel_size: Union[np.ndarray, List[int]],
                     outmask: sitk.Image) -> sitk.Image:
        r"""
        Apply closing and binary fill hole to the input mask.

        .. warning::
            Outputs are converted to sitk.sitkUInt8 data type!

        Args:
            closing_kernel_size (list of int):
                Kernel_size in unit of voxels. Should be integers.
            outmask (sitk.Image):
                Mask output to be processed.

        Returns:
            sitk.Image
        """
        outmask = sitk.Cast(outmask, sitk.sitkUInt8)
        outmask = sitk.BinaryMorphologicalClosing(outmask, closing_kernel_size.tolist())  # TODO: check kernel size
        outmask[:, :, 0] = sitk.BinaryFillhole(outmask[:, :, 0])
        outmask[:, :, -1] = sitk.BinaryFillhole(outmask[:, :, -1])
        outmask = sitk.BinaryFillhole(outmask, fullyConnected=True)
        return outmask

    def _filter(self,
                input: Union[str, Path, sitk.Image]) -> sitk.Image:
        input = self.read_image(input)
        closing_kernel_size = self.compute_closing_kernel_size(input)

        mask = self.core_func(input)
        mask = self.mask_closing(closing_kernel_size, mask)
        return mask


class OtsuTresholding(ThresBinaryClosing):
    r"""
    See Also:
        :class:`sitk.OtsuThreshold`

    """
    def __init__(self, closing_kernel_size=[5, 5, 5]):
        super(OtsuTresholding, self).__init__(closing_kernel_size=closing_kernel_size)
        self.core_func = self._core_func

    def _core_func(self, input):
        return sitk.OtsuThreshold(input, 0, 1, 200)


class HuangThresholding(ThresBinaryClosing):
    r"""
    See Also:
        :class:`sitk.HuangThreshold`

    """
    def __init__(self, closing_kernel_size=[5, 5, 5]):
        super(HuangThresholding, self).__init__(closing_kernel_size=closing_kernel_size)
        self.core_func = self._core_func

    def _core_func(self, input):
        return sitk.HuangThreshold(input, 0, 1, 200)