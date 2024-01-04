import SimpleITK as sitk
import numpy
from typing import Iterable, Union, Optional, List, Dict, Tuple
from pathlib import Path

try:
    import pydicom
    __all__ = ['DIXON_dcm_to_images']
except:
    __all__ = []

def DIXON_dcm_to_images(dcm_files: List[Path]) -> Dict[Tuple, sitk.Image]:
    r"""Converts a list of DIXON DICOM files into SimpleITK images.

    This function reads a list of DIXON DICOM files, identifies the unique
    image types present in the list, and converts each group of image types
    into SimpleITK Image objects. If multiple image types are detected, they are
    separated and processed independently.

    Args:
        dcm_files (List[Path]):
            A list of paths to DICOM files. Each path can be a `Path` object or a string.

    Returns:
        Dict[tuple, sitk.Image]:
            A dictionary mapping tuples of DICOM image types to their corresponding SimpleITK Image objects.
            Image types are determined by the DICOM tag (0x0008, 0x0008).

    Raises:
        TypeError: If `dcm_files` is not a list of paths or strings.
        FileNotFoundError: If any of the provided DICOM file paths does not exist.
        AssertionError: If any of the paths in `dcm_files` is not a file.
        InvalidDicomError: If an error occurs during reading non-DICOM files.

    Examples:
        >>> from pathlib import Path
        >>> dcm_paths = [Path('/path/to/dicom1.dcm'), Path('/path/to/dicom2.dcm')]
        >>> images = DIXON_dcm_to_images(dcm_paths)
        >>> type(images)
        <class 'dict'>

        # Accessing an image by its DICOM image type tuple
        >>> image_type = next(iter(images.keys()))
        >>> type(images[image_type])
        <class 'SimpleITK.SimpleITK.Image'>
    """
    # use first element to determine type
    if isinstance(dcm_files[0], (Path, str)):
        dcm_files = [Path(ff) for ff in dcm_files]
        # make sure all of them exist
        assert all([ff.is_file() for ff in dcm_files])
    else:
        msg = f"Input must be a list of path or strings, got {type(dcm_files[0])} instead."
        raise TypeError(msg)

    # read image types
    image_types = [(ff, pydicom.read_file(ff, specific_tags=[pydicom.tag.Tag(0x0008, 0x0008)]).ImageType) for ff in dcm_files]

    # check if there's multiple image types
    unique_image_types = [tuple(ds) for _, ds in image_types]
    unique_image_types = set(unique_image_types)

    # if more than one type, split the files based on image_types
    if len(unique_image_types) > 1:
        #
        images = {t: [] for t in unique_image_types}
        for ff, t in image_types:
            images[t].append(ff)

        sitk_images = {}
        for t in unique_image_types:
            # read images with sitk
            image_reader = sitk.ImageSeriesReader()
            image_reader.SetFileNames(images[t])
            sitk_images[t] = image_reader.Execute()
    else:
        t = list(unique_image_types)[0]
        sitk_images = {
            t: sitk.ReadImage(dcm_files)
        }
    return sitk_images





