import SimpleITK as sitk
import os
import numpy as np
import re
import multiprocessing as mpi
from tqdm import *
import random
import argparse
import pandas as pd

from pytorch_med_imaging.logger import Logger, LogExceptions
from typing import Optional, Union
import multiprocessing as mpi

__all__ = ['recursive_list_dir']

def recursive_list_dir(searchDepth, rootdir):
    r"""
    Recursively list directories only.
    """

    DD = []
    for r, d, f in os.walk(rootdir, followlinks=False):
        if len(f) != 0:
            # DD.extend([os.path.join(r, dd) for dd in d])
            DD.append(r)
    # DD.extend(nextlayer)
    return DD


def SmoothImages(root_dir, out_dir):
    import fnmatch

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    f = os.listdir(root_dir)
    fnmatch.filter(f, "*.nii.gz")

    for fs in f:
        print(fs)
        im = sitk.ReadImage(root_dir + "/" + fs)
        out = sitk.SmoothingRecursiveGaussian(im, 8, True)
        sitk.WriteImage(out, out_dir + "/" + fs)


def make_mask(inimage,
              outdir,
              threshold_lower,
              threshold_upper = None,
              inside_to_1 = True,
              pos=-1):
    r"""Create a mask of an input with specified threshold slice-by-slice.

    Args:
        inimage (str or sitk.Image):
            Input image.
        outdir (str):
            Ouptut directory.
        threshold_lower (float):
            Lower threshold.
        threshold_upper (float, Optional):
            Upper threshold. If none is provided, use the maximum value of the image as the threshold.
            Default to None.
        inside_to_1 (bool, Optional):
            If True, values inside the threholds becomes 1, else 0. Default to be True.
        pos (int):
            For MPI purpose, don't use.
    """
    workerid = mpi.current_process().name
    logger = Logger['utils.preprocessing-%s'%workerid]

    if isinstance(inimage, str):
        logger.info(f"Handling: {inimage}")
        inimage = sitk.ReadImage(inimage)

    if threshold_upper is None:
        threshold_upper = sitk.GetArrayFromImage(inimage).max()

    # setup variables
    inside_value = 1 if inside_to_1 else 0
    outside_value = 0 if inside_to_1 else 1

    # might need to cast type correctly in the future
    gttest = sitk.BinaryThreshold(inimage,
                                  upperThreshold=float(threshold_upper),
                                  lowerThreshold=float(threshold_lower),
                                  insideValue=bool(inside_value),
                                  outsideValue=bool(outside_value))

    gttest = sitk.BinaryDilate(gttest, [15, 15, 0], sitk.BinaryMorphologicalOpeningImageFilter.Ball)
    gttest = sitk.BinaryErode(gttest, [15, 15, 0], sitk.BinaryMorphologicalOpeningImageFilter.Ball)
    # gttest = sitk.BinaryMorphologicalClosing(gttest, [0, 25, 25], sitk.BinaryMorphologicalOpeningImageFilter.Ball)
    ss = []


    if pos == -1:
        try:
            pos = int(mpi.current_process().name.split('-')[-1])
        except Exception as e:
            logger.exception(e)

    try:
        for i in trange(gttest.GetSize()[-1], position=pos, desc=mpi.current_process().name):
            ss.append(sitk.GetArrayFromImage(sitk.BinaryFillhole(gttest[:,:,i])))
        gttest = sitk.GetImageFromArray(np.stack(ss))
        # gttest = sitk.BinaryDilate(gttest, [0, 3, 3], sitk.BinaryDilateImageFilter.Ball)
        gttest.CopyInformation(inimage)
        sitk.WriteImage(gttest, outdir)
        logger.info(f"Written to: {outdir}")
        return 0
    except Exception as e:
        logger.exception(e)

def make_mask_from_dir(indir, outdir, threshold_lower, threshold_upper, inside_to_1, num_worker=10):
    r"""Make mask from a directory"""
    import fnmatch
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    p = mpi.Pool(num_worker)
    processes = []
    filelist = os.listdir(indir)
    filelist = fnmatch.filter(filelist, '*.nii.gz')
    filelist = [indir + '/' + f for f in filelist]


    for i, f in enumerate(filelist):
        outname = f.replace(indir, outdir)
        # make_mask(f, outname, threshold_lower, threshold_upper, inside_to_1)
        subp = p.apply_async(LogExceptions(make_mask), (f, outname, threshold_lower, threshold_upper, inside_to_1))
        processes.append(subp)

    for pp in processes:
        pp.wait(50)
    p.close()
    p.join()


def mask_image_with_label(in_label: sitk.Image,
                          mask_label: sitk.Image,
                          mask_index: Optional[int] = None,
                          mask_dilation: Optional[int] = 0) -> sitk.Image:
    r"""Mask a label map with another label map

    Args:
        in_label (sitk.Image): Input
        mask_label (sitk.Image): Output
        mask_index (int, Optional): If an index is provided, it will be used for masking.
        mask_dilation (int, Optional): If a value is provided, binary map is dilated before masking.
    """
    # Error check
    assert mask_dilation >= 0, "Dilation radius cannot be negative."

    # If index is provided, make a binary mask out of it
    if mask_index is not None:
        mask_label = mask_label == mask_index
        pass

    # Otherwise, use all non-zero labels
    mask_label = sitk.LabelImageToLabelMap(mask_label)
    mask_label = sitk.LabelMapToBinary(mask_label)

    # Dilation if specified
    if mask_dilation > 0:
        mask_label = sitk.BinaryDilate(mask_label, mask_dilation, sitk.sitkBall)

    # Masking
    return sitk.Mask(in_label, mask_label, float(sitk.GetArrayFromImage(in_label).min()))

