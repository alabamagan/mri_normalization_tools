from abc import abstractmethod
from pathlib import Path
from typing import Iterable, Union

import SimpleITK as sitk

from ..mnts_logger import MNTSLogger

__all__ = ['MNTSFilter', 'MNTSFilterPipeline', 'MNTSFilterRequireTraining']

class MNTSFilter(object):
    def __init__(self):
        r"""
        Base class of filter
        """
        self._logger = MNTSLogger[self.get_name()]

    @abstractmethod
    def filter(self, *args, **kwargs):
        raise NotImplemented("This is an abstract method.")

    @property
    def name(self):
        return self.get_name()

    def get_name(self):
        return self.__class__.__name__

    def get_all_properties(self):
        n = [(name, self.__getattribute__(name)) for name, value in vars(self.__class__).items() if isinstance(value, property)]
        return n

    @staticmethod
    def read_image(input: Union[str, Path, sitk.Image]):
        if isinstance(input, (str, Path)):
            MNTSLogger.global_logger.info(f"Reading image from: {str(input)}")
            input = sitk.ReadImage(str(input))

        if not isinstance(input, sitk.Image):
            msg = f"Cannot read file from: {str(input.absolute())}"
            raise IOError
        return input


    def __call__(self, *args, **kwargs):
        return self.filter(*args, **kwargs)

    def __str__(self):
        return f"{self.__class__.__name__}: \n" + "-" * (len(self.__class__.__name__)+1) + \
               '\n\t' + '\n\t'.join(["{: >20}: {}".format(item[0], item[1]) for item in self.get_all_properties()])

class MNTSFilterRequireTraining(MNTSFilter):
    def __init__(self):
        r"""
        Base class of filters that require training.
        """
        super(MNTSFilterRequireTraining, self).__init__()

    @abstractmethod
    def train(self, *args, **kwargs) -> object:
        raise NotImplementedError()

    @abstractmethod
    def save_state(self, path: Union[str, Path], with_suffix=None):
        location = Path(path)
        if location.exists():
            self._logger.warning(f"Found existing saved states at {location.resolve().__str__()}, tyring to cover it.")
            if location.is_dir():
                raise IOError(f"Recieved directory {path.__str__()} as argument.")

        # Create parent directory if not exist.
        if not location.parent.is_dir():
            self._logger.warning(f"Creating directory at {location.parent.resolve().__str__()}")
            location.parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_state(self, path):
        path = Path(path)
        if not path.exists():
            raise IOError(f"Cannot load state from {path.resolve().__str__()}")


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
        return ' -> '.join([f.name for f in self])

    def execute(self, *args):
        raise NotImplementedError
        for f in iter(self):
            args = f(*args)
            if not isinstance(args, [tuple, list]):
                args = [args]
        return args

    def sort(self):
        # Disable sort
        pass
