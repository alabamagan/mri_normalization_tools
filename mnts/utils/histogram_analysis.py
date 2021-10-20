import os
import numpy as np
import SimpleITK as sitk
import multiprocessing as mpi
from functools import partial
from tqdm.auto import tqdm

from pathlib import Path
from typing import Union, Optional, Iterable, Any
from . import repeat_zip
import fnmatch

__all__ = ['batch_get_distribtuion', 'plot_hist']

def get_distribution(img_dir: Union[str, Path],
                     bins: Optional[int] = 200,
                     masking_method: Optional[str] = None,
                     remove_outliers: Optional[bool]  = True,
                     ):
    r"""Return the normalized histogram and the bin centers of the image specified.

    Args:
        img_dir:
            Path to input image. Must be nii.gz or in other format readable using function `sitk.ReadImage`.
        bins (Optional, int):
            Bin number of normalized histogram anaslysis. Default to 200.
        masking_method (Optional, str):
            Either `'corner'` or a lambda function that returns a numpy Indexing array (boolean array).
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Tuple(counts[np.ndarray], bin center [np.ndarray]). Size of array is the `bin_size`.
    """
    if not Path(img_dir).is_file():
        raise FileNotFoundError(f"Cannot open file {img_dir.resolve()}.")
    if bins <= 0:
        raise ArithmeticError(f"'bin_size' must be positive, got {bins} instead.")
    bins = int(bins)
    if masking_method not in ('corner', '3sigma', None) and not callable(masking_method):
        raise TypeError(f"Masking error was not specified correctly. Got {masking_method}.")

    # Read image
    im = sitk.GetArrayFromImage(sitk.ReadImage(img_dir)).flatten()

    # Mask image
    if not masking_method is None:
        if masking_method == 'corner':
            bg_value = im[0]
            mask = im > bg_value
        elif masking_method == '3sigma':
            _mean = im.mean()
            _std = im.std()
            range = [_mean - 3 * _std, _mean + 3 * _std]
            mask = (range[0] <= im) & (im <= range[1])
        elif callable(masking_method):
            mask = masking_method(im)
        im = im[mask]

    # Use mu ± 3 sigma to filter out outliers

    h, b = np.histogram(im, bins=bins, density=True)
    b_cent = (b[1:] + b[:-1]) / 2.
    return (h, b_cent)

def batch_get_distribtuion(imgs_dir: Union[str, Path, Iterable[Union[str, Path]]],
                           bins: Optional[int] = 200,
                           masking_method: Optional[str] = 'corner',
                           recursive_include: Optional[bool] = False,
                           numworkers: Optional[int] = 16) -> np.ndarray:
    r"""

    Args:
        imgs_dir (str or List[str]):
            Path to the directory that contains the target images.
        bins (int):
            Number of bins pass to the `get_distribution` function.
        masking_method (Optional, str):
            Either `'corner'` or a callable function that returns a numbpy Indexing array (boolean array). Default to
            `'corner'`
        recursive_include (Optional, bool):
            If true, include the '.nii.gz' files recursively. Ignored when imgs_dirs was a list.
        numworkers (Optional, int):
            Number of worker. If == 1, the code is run in linear mode, if <= 0, the code is run paralleled using
            `mpi.cpu_cout()` threads. Otherwise, the specified number of workers are used.

    Returns:
        np.ndarray: Shape is (I, 2, B). Where I is the number of images, B is the number of bins.
    """
    if isinstance(imgs_dir, (str, Path)):
        imgs_dir = Path(imgs_dir)
        if not imgs_dir.is_dir():
            raise FileNotFoundError(f"Cannot open files from: {imgs_dir.resolve()}")

        if recursive_include:
            tmp_dirs = []
            for r, d, f in os.walk(imgs_dir):
                if not len(f) == 0:
                    f = fnmatch.filter(f, '*.nii.gz')
                    f = [os.path.join(r, ff) for ff in f]
                    tmp_dirs.extend(f)
            imgs_dir = tmp_dirs
        else:
            imgs_dir = fnmatch.filter([str(s) for s in imgs_dir.iterdir()], "*.nii.gz")

    # Check if everything's founded
    paths = np.asarray([Path(f) for f in imgs_dir])
    found = np.asarray([f.is_file() for f in paths])
    if not all(found):
        print(f"These files are missing: {paths[~found]}")
        raise FileNotFoundError("Somes files are not found.")

    hists = np.zeros([len(imgs_dir), 2, bins])
    if numworkers != 1:
        if numworkers <= 0:
            numworkers = mpi.cpu_count()
        pool = mpi.Pool(numworkers)
        r = pool.map_async(partial(get_distribution,
                                   bins=bins,
                                   masking_method=masking_method),
                           imgs_dir)
        results = r.get()
        for i, (h, b_cent) in enumerate(results):
            hists[i, 0] = h
            hists[i, 1] = b_cent

    else:
        for i, f in enumerate(tqdm(imgs_dir)):
            print(f"Processing {f}")
            h, b_cent = get_distribution(f, bins, masking_method)
            hists[i, 0] = h
            hists[i, 1] = b_cent
    return hists


def plot_hist(hists: np.ndarray,
              ax: Optional[Any] = None,
              *args,
              **kwargs) -> None:
    r"""
    Plot the histogram obtained from `batch_get_histogram`
    """
    import matplotlib.pyplot as plt

    if ax is None:
        figsize = kwargs.pop('figsize') if 'figsize' in kwargs else None
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i in range(hists.shape[0]):
        ax.plot(hists[i, 1][5:], hists[i, 0][5:], *args, **kwargs)
    plt.show()
