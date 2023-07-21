from .mnts_filters import *
from typing import Any, Optional
import SimpleITK as sitk

__all__ = ["TypeCastNode", "DataNode"]

class DataNode(MNTSFilter):
    r"""
    Presents whatever data stored in this node. This is useful for storing intermediate results, or other data that
    are repeatedly accessed without the need of re-computing. This can be used as the entrance node in the directed
    graphs. See :class:`MNTSFilterGraph` for more.
    """
    def __init__(self, data=None):
        super(DataNode, self).__init__()
        self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, input: Any):
        self._logger.info(f"Setting data ({type(input)}) {input}")
        if isinstance(input, str):
            input = self.read_image(input)
        self._data = input

    def clear_data(self):
        self._data = None

    def filter(self,
               input: str or sitk.Image) -> Any:
        self.data = input
        return self.data

class TypeCastNode(MNTSFilter):
    r"""
    A class to cast the type to a specific datatype using `sitk.Cast`.
    It is recommended to keep a clear track of the type casting within a pipeline or a graph.

    Attributes:
        target_type (int): The target type to which the input will be casted.
        target_type_name (str): The name of the target type.
    """
    def __init__(self,
                 target_type: int = sitk.sitkInt16):
        self._target_type = target_type
        self._target_type_name = sitk.GetPixelIDValueAsString(self._target_type)
        self._overflow_protection = {
            sitk.sitkUInt8:     (0, int(2**8.) - 1),
            sitk.sitkUInt16:    (0, int(2**16.) - 1),
            sitk.sitkUInt32:    (0, int(2**32.) - 1),
            sitk.sitkUInt64:    (0, int(2**64.) - 1),
            sitk.sitkInt8:      (-int(2**7.), int(2**7.) -1),
            sitk.sitkInt16:     (-int(2**15.), int(2 ** 15.) - 1),
            sitk.sitkInt32:     (-int(2**31.), int(2 ** 31.) - 1),
            sitk.sitkInt64:     (-int(2**63.), int(2 ** 63.) - 1)
            # Ignore float numbers
        }

    @property
    def target_type(self) -> Any:
        r"""Return the target type as sitk code."""
        return self._target_type

    @target_type.setter
    def target_type(self, val) -> None:
        r"""Set the target type with sitk codes."""
        self._target_type = val

    @property
    def target_type_name(self) -> str:
        r"""Return the target type as string."""
        return self._target_type_name

    def filter(self, input: sitk.Image, ref_img: Optional[sitk.Image]=None) -> sitk.Image:
        """Filter method to cast the type of the input.

        If `ref_img` is not None, then the type cast references the type of it.
        Otherwise, the type is casted as the attribute `self._target_type`.

        Args:
            input (sitk.Image):
                The input to be casted.
            ref_img (sitk.Image, optional):
                The reference image to determine the type. Defaults to None.

        Returns:
            The input casted to the target type.

        Raises:
            ArithmeticError: If the type cast fails.
        """
        input = self.read_image(input)
        target_type = self._target_type if ref_img is None else ref_img.GetPixelID()
        try:
            # Overflow protection
            range = self._overflow_protection.get(self.target_type, None)
            if range is not None:
                f = sitk.MinimumMaximumImageFilter()
                f.Execute(input)

                max_val = f.GetMaximum()
                min_val = f.GetMinimum()

                if max_val > range[1] or min_val < range[0]:
                    input = sitk.Clamp(input, *range)
            return sitk.Cast(input, self._target_type)
        except Exception as e:
            raise ArithmeticError(f"Type cast failed in filter with parameters: {self.__str__()}") from e
