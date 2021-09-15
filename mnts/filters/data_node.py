from .mnts_filters import *
from typing import Any, Optional
import SimpleITK as sitk

__all__ = ["TypeCastNode"]

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

    @property
    def data(self, input):
        self._logger.info(f"Setting data ({type(input)}) {input}")
        self._data = input

    def filter(self,
               intput: Any) -> Any:
        if self.data is None:
            self.data = input
        return self.data

class TypeCastNode(MNTSFilter):
    r"""
    Cast the type to specific datatype using sitk.Cast. However, you are recommended to keep a clear track of the type
    casting within a pipeline or a graph.
    """
    def __init__(self,
                 target_type: int = sitk.sitkInt16):
        self._target_type = target_type
        self._target_type_name = sitk.GetPixelIDValueAsString(self._target_type)

    @property
    def target_type(self):
        return self._target_type

    @target_type.setter
    def target_type(self, val):
        self._target_type = val

    @property
    def target_type_name(self):
        return self._target_type_name

    def filter(self, input):
        input = self.read_image(input)
        try:
            return sitk.Cast(input, self._target_type)
        except Exception as e:
            raise ArithmeticError(f"Type cast failed in filter with parameters: {self.__str__()}") from e
