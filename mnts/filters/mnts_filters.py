from abc import ABCMeta, abstractmethod, abstractproperty
from ..mnts_logger import MNTSLogger
from typing import Union, Iterable

class MNTSFilter(object):
    def __init__(self):
        r"""
        Base class of filter
        """
        self._logger = MNTSLogger[self.get_name()]

    @property
    @abstractmethod
    def filter(self, *args, **kwargs):
        pass

    def get_name(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        self.filter(*args, **kwargs)

    def __str__(self):
        return f"{self.__class__.__name__}: \n\t" + '\n\t'.join(["{: >15} - %s"%item for item in vars(self).items()])


class MNTSFilterPipeline(list):
    def __init__(self,
                 *args: Iterable[MNTSFilter]
                 ):
        r"""
        A list of filter that will be called in sequential order
        Args:
            *args:
        """
        super(MNTSFilterPipeline, self).__init__(*args)

        self._logger = MNTSLogger[self.__class__.__name__]

        if len(args) > 1:
            assert all([isinstance(f, MNTSFilter) for f in args])

    def __str__(self):
        return '->'.join([f.name for f in self])

    def execute(self, *args):
        for f in iter(self):
            args = f(*args)
            if not isinstance(args, [tuple, list]):
                args = [args]
        return args

    def sort(self):
        pass