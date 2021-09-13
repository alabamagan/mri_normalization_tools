r"""

"""
import sys
import os
import time

import SimpleITK as sitk
import numpy as np
import multiprocessing as mpi
from typing import Union, List, Tuple
from pathlib import Path
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
from ..mnts_filters import MNTSFilterRequireTraining
from .intensity_base import MNTSIntensityBase
import pprint

__all__ = ['NyulNormalizer']

# ********************************************************
# ************* Auxiliar functions ***********************
# ********************************************************

def getCdf(hist):
    """
        Given a histogram, it returns the cumulative distribution function.
    """
    aux = np.cumsum(hist)
    aux = aux / aux[-1] * 100
    return aux


def getPercentile(cdf, bins, perc):
    """
    Given a cumulative distribution function obtained from a histogram,
    (where 'bins' are the x values of the histogram and 'cdf' is the
    cumulative distribution function of the original histogram), it returns
    the x center value for the bin index corresponding to the given percentile,
    and the bin index itself.

    Example:

        >>> import numpy as np
        >>> hist = np.array([204., 1651., 2405., 1972., 872., 1455.])
        >>> bins = np.array([0., 1., 2., 3., 4., 5., 6.])
        >>>
        >>> cumHist = getCdf(hist)
        >>> self._logger.info cumHist
        >>> val, bin = getPercentile(cumHist, bins, 50)
        >>>
        >>> self._logger.info "Val = " + str(val)
        >>> self._logger.info "Bin = " + str(bin)

    """
    b = len(bins[1:][cdf <= perc])
    return bins[b] + ((bins[1] - bins[0]) / 2), b


def ensureDir(f):
    d = Path(f).absolute()
    if not d.exists():
        d.mkdir(parents=True)


def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")


# ********************************************************
# ************* NyulNormalizer class *********************
# ********************************************************

class NyulNormalizer(MNTSIntensityBase, MNTSFilterRequireTraining):
    """
    Nyul normalization, i.e., piecewise linear histogram matching

    Attributes:
        lower_origin:

        upper_origin:

        lower_target:

        upper_target:

        num_features_points:


    :param pLow, upper_origin  Percentiles corresponding to the positions where the first and last landmarks are going to be mapped.
    :param sMin, upper_target   Minimum and maximum value of the target intensity range. The value corresponding to the percentile pLow will be mapped to sMin, while
                        the value corresponding to upper_origin will be mapped to upper_target.
    :param numPoints    Number of landmarks that will be learned
    :param showLandmarks Show the landmarks for every image, overlapped in the corresponding histogram, using matplotlib.

    .. note::
        This is a fork of the repo from https://gitlab.com/eferrante/nyul.
        This is modified to align with the OOP structure of this package.
    """
    nbins = 1024

    def __init__(self,
                 lower_origin: float = 1,       # 0 - 9
                 upper_origin: float = 99,      # 91 - 100
                 lower_target: float = 1,       # 0 - 100
                 upper_target: float = 100,     # 0 - 100
                 num_feature_points: int = 10,  # Recomend 4 to 10
                 load_from_state: Union[str, Path] = None
                 ):
        super(NyulNormalizer, self).__init__()
        # Percentiles used to trunk the tails of the histogram
        if lower_origin > 10:
            raise ArithmeticError("NyulNormalizer Error: pLow may be bigger than the first lm_pXX landmark.")
        if upper_origin < 90:
            raise ArithmeticError("NyulNormalizer Error: upper_origin may be bigger than the first lm_pXX landmark.")

        self._lower_origin = lower_origin
        self._upper_origin = upper_origin
        self._lower_target = lower_target
        self._upper_target = upper_target
        self._num_features_points = num_feature_points
        self._mean_landmarks = None

        if not load_from_state is None:
            s = Path(load_from_state)
            if not s.exists():
                self._logger.warning("Cannot locate specified state.")
            else:
                self.load_state(load_from_state)


    @property
    def lower_origin(self):
        return self._lower_origin

    @lower_origin.setter
    def lower_origin(self, val):
        assert 0 <= val < 10, f"lower_origin must be 0 to 9, got {val} instead."
        self._lower_origin = val

    @property
    def upper_origin(self):
        return self._upper_origin

    @upper_origin.setter
    def upper_origin(self, val):
        assert 90 < val <= 100, f"upper_origin must be 91 to 100, got {val} instead."
        self._upper_origin = val

    @property
    def num_feature_points(self):
        return self._num_features_points

    @num_feature_points.setter
    def num_feature_points(self, val):
        assert val >= 4, "num_feature_points is meaningless when smaller than 4."
        self._num_features_points = int(val)

    @property
    def lower_target(self):
        return self._lower_target

    @lower_target.setter
    def lower_target(self, val):
        assert 0 <= val <= 100, f"lower_target must be 0 to 100, got {val} instead."
        self._lower_target = val

    @property
    def upper_target(self):
        return self._upper_target

    @upper_target.setter
    def upper_target(self, val):
        assert 0 <= val <= 100, f"upper_target must be 0 to 100, got {val} instead."
        self._upper_target = val

    def __getLandmarks(self, image, mask=None, showLandmarks=False):
        """
            This Private function obtain the landmarks for a given image and returns them
            in a list like:
                [lm_lower_origin, lm_perc1, lm_perc2, ... lm_perc_(numPoints-1), lm_upper_origin] (lm means landmark)

            :param image    SimpleITK image for which the landmarks are computed.
            :param mask     [OPTIONAL] SimpleITK image containing a mask. If provided, the histogram will be computed
                                    taking into account only the voxels where mask > 0.
            :param showLandmarks    Plot the landmarks using matplotlib on top of the histogram.

        """

        data = sitk.GetArrayFromImage(image)

        if mask is None:
            # Calculate useful statistics
            stats = sitk.StatisticsImageFilter()
            stats.Execute(image)
            mean = stats.GetMean()

            # Compute the image histogram
            histo, bins = np.histogram(data.flatten(), self.nbins, normed=True)

            # Calculate the cumulative distribution function of the original histogram
            cdfOriginal = getCdf(histo)

            # Truncate the histogram (put 0 to those values whose intensity is less than the mean)
            # so that only the foreground values are considered for the landmark learning process
            histo[bins[:-1] < mean] = 0.0
        else:
            # Calculate useful statistics
            dataMask = sitk.GetArrayFromImage(mask)

            # Compute the image histogram
            histo, bins = np.histogram(data[dataMask > 0].flatten(), self.nbins, density=True)

            # Calculate the cumulative distribution function of the original histogram
            cdfOriginal = getCdf(histo)

        # Calculate the cumulative distribution function of the truncated histogram, where outliers are removed
        cdfTruncated = getCdf(histo)

        # Generate the percentile landmarks for  m_i
        perc = [x for x in range(0, 100, 100 // self._num_features_points)]
        # Remove the first landmark that will always correspond to 0
        perc = perc[1:]

        # Generate the landmarks. Note that those corresponding to lower_origin and upper_origin (at the beginning and the
        # end of the list of landmarks) are generated from the cdfOriginal, while the ones
        # corresponding to the percentiles are generated from cdfTruncated, meaning that only foreground intensities
        # are considered.
        landmarks = [getPercentile(cdfOriginal, bins, self._lower_origin)[0]] + [getPercentile(cdfTruncated, bins, x)[0] for x in
                                                                                 perc] + [
                        getPercentile(cdfOriginal, bins, self._upper_origin)[0]]
        return landmarks

    def __landmarksSanityCheck(self, landmarks):
        if not (np.unique(landmarks).size == len(landmarks)):
            for i in range(1, len(landmarks) - 1):
                if landmarks[i] == landmarks[i + 1]:
                    landmarks[i] = (landmarks[i - 1] + landmarks[i + 1]) / 2.0

                self._logger.warning("Fixing duplicate landmark.")

            if not (np.unique(landmarks).size == len(landmarks)):
                msg = 'ERROR NyulNormalizer landmarks sanity check : One of the landmarks is ' \
                      'duplicate. You can try increasing the number of bins in the histogram ' \
                      '(NyulNormalizer.nbins) to avoid this behaviour. Landmarks are: ' + str(landmarks)
                raise Exception(msg)

        elif not (sorted(landmarks) == list(landmarks)):

            raise Exception(
                'ERROR NyulNormalizer landmarks sanity check: Landmarks in the list are not sorted, while they should be. Landmarks are: ' + str(
                    landmarks))

    def train(self,
              listOfImages: List[Union[str, Path]],
              listOfMasks: List[Union[str, Path]] =[],

              ) -> None:
        """
        Train a new model for the given list of images (listOfImages is a list of strings, where every element is the full path to an image)

        Optionally, a list containing full paths to the corresponding masks (listOfMasks) can be provided. If masks are provided, the image will

        Note that the actual number of points is numPoints that will be generated (including
        the landmarks corresponding to lower_origin and upper_origin) is numPoints + 1.

        Recommended values for numPoints are 10 and 4.

        Example 1: if lower_origin = 1, upper_origin = 99, numPoints = 10, the landmarks will be:

            [lm_p1, lm_p10, lm_p20, lm_p30, lm_p40, lm_p50, lm_p60, lm_p70, lm_p90, lm_p99 ]

        Example 2: if lower_origin = 1, upper_origin = 99, numPoints = 4, the landmarks will be:

            [lm_p1, lm_p25, lm_p50, lm_p75, lm_p99]

        Arguments:


        """
        allMappedLandmarks = []
        # For each image in the training set
        for fNameIndex in range(len(listOfImages)):
            fName = listOfImages[fNameIndex]
            self._logger.info("Processing: " + fName)
            # Read the image
            image = self.read_image(fName)

            if listOfMasks == []:
                # Generate the landmarks for the current image
                landmarks = self.__getLandmarks(image)
            else:
                mask = sitk.ReadImage(listOfMasks[fNameIndex])
                landmarks = self.__getLandmarks(image, mask)

            # Check the obtained landmarks ...
            self.__landmarksSanityCheck(landmarks)

            # Construct the linear mapping function
            mapping = interp1d([landmarks[0], landmarks[-1]], [self.lower_target, self.upper_target], fill_value="extrapolate")

            # Map the landmarks to the standard scale
            mappedLandmarks = mapping(landmarks)

            # Add the mapped landmarks to the working set
            allMappedLandmarks.append(mappedLandmarks)

        self._logger.info("ALL MAPPED LANDMARKS: ")
        self._logger.info(f"\n{pprint.pformat(allMappedLandmarks)}")

        self._mean_landmarks = np.array(allMappedLandmarks).mean(axis=0)

        # Check the obtained landmarks ...
        self.__landmarksSanityCheck(self._mean_landmarks)

        self._logger.info(f"MEAN LANDMARKS: ")
        self._logger.info(f"\n{pprint.pformat(self._mean_landmarks)}")

    def save_state(self,
                   location: Union[str, Path]):
        """
            Saves the trained model in the specified file location (it adds '.npz' to the filename, so
            do not specify the extension). To load it, you can use:

                nyulNormalizer.loadTrainedModel(outputModel)

            :param location Absolute path corresponding to the output file. It adds '.npz' to the filename, so
                            do not specify the extension (e.g. /path/to/file)
        """
        location = Path(location)
        super(NyulNormalizer, self).save_state(location)


        trainedModel = {
            'lower_origin': self._lower_origin,
            'upper_origin': self._upper_origin,
            'lower_target': self._lower_target,
            'upper_target': self._upper_target,
            'numPoints': self._num_features_points,
            'meanLandmarks': self._mean_landmarks
        }
        np.savez(location.with_suffix('.npz'), trainedModel=[trainedModel])
        self._logger.info(f"Model saved at: {location.__str__()}")

    def load_state(self, location):
        """
            Loads a trained model previously saved using:

                nyulNormalizer.saveTrainedModel(outputModel)

            :param location Absolute path (including extension) to the "npz" file with the corresponding learned
            landmarks.
        """
        location = location.with_suffix('.npz')
        super(NyulNormalizer, self).load_state(location)

        f = np.load(location, allow_pickle=True)
        tModel = f['trainedModel'].all()

        self._lower_origin = tModel['lower_origin']
        self._upper_origin = tModel['upper_origin']
        self._num_features_points = tModel['numPoints']
        self._lower_target = tModel['lower_target']
        self._upper_target = tModel['upper_target']
        self._mean_landmarks = tModel['meanLandmarks']

        # Check the obtained landmarks ...
        self.__landmarksSanityCheck(self._mean_landmarks)

    def transform(self, image, mask=None):
        """
            It transforms the image to the learned standard scale
            and returns it as a SimpleITK image.

            If mask is provided, the histogram of the image to be normalized will be only calculated based on the pixels where mask>0.
            All pixels (independently of their mask value) will be transformed.

            The intensities between [minIntensity, intensity(lower_origin)) are linearly mapped using
            the same function that the first interval.

            The intensities between [intensity(upper_origin), maxIntensity) are linearly mapped using
            the same function that the last interval.

        Attributes:
            image (str, path or sitk.Image):
                Image that will be transformed.
            mask (str, path or sitk.Image, Optional):
                If provided, only voxels where mask > 0 will be considered to compute the histogram.

        Returns:
            output image
        """

        # Get the raw data of the image
        image = self.read_image(image)
        data = sitk.GetArrayFromImage(image)

        # Calculate useful statistics
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)

        # Obtain the minimum
        origMin = stats.GetMinimum()
        origMax = stats.GetMaximum()
        origMean = stats.GetMean()
        origVariance = stats.GetVariance()

        self._logger.info("Input stats:")
        self._logger.info("Min = " + str(origMin))
        self._logger.info("Max = " + str(origMax))
        self._logger.info("Mean = " + str(origMean))
        self._logger.info("Variance = " + str(origVariance))

        # Get the landmarks for the current image
        landmarks = self.__getLandmarks(image, mask)
        landmarks = np.array(landmarks)

        # Check the obtained landmarks ...
        self.__landmarksSanityCheck(landmarks)

        # Recover the standard scale landmarks
        standardScale = self._mean_landmarks

        self._logger.info("Image landmarks: " + str(landmarks))
        self._logger.info("Standard scale : " + str(standardScale))

        # Construct the piecewise linear interpolator to map the landmarks to the standard scale
        mapping = interp1d(landmarks, standardScale, fill_value="extrapolate")

        # Map the input image to the standard space using the piecewise linear function

        flatData = data.ravel()
        self._logger.info("Mapping data...")
        mappedData = mapping(flatData)
        self._logger.info("Mapping done.")
        mappedData = mappedData.reshape(data.shape)

        # Save edited data
        output = sitk.GetImageFromArray(mappedData)
        output.CopyInformation(image)

        # Calculate useful statistics
        stats = sitk.StatisticsImageFilter()
        stats.Execute(output)

        self._logger.info("Output stats")

        # Obtain the minimum
        origMin = stats.GetMinimum()
        origMax = stats.GetMaximum()
        origMean = stats.GetMean()
        origVariance = stats.GetVariance()

        self._logger.info("Min = " + str(origMin))
        self._logger.info("Max = " + str(origMax))
        self._logger.info("Mean = " + str(origMean))
        self._logger.info("Variance = " + str(origVariance))

        return output

    def _filter(self,
                input: sitk.Image,
                mask: sitk.Image = None):
        input = self.read_image(input)
        mask = self.read_image(mask)

        if self._mean_landmarks is None:
            self.error("This node was no trained.")
            return

        # Cast to float if it wasn't already float
        if not input.GetPixelID() in [sitk.sitkFloat32, sitk.sitkFloat64]:
            self._logger.warning("Casting image to float!")
            sitk.Cast(input, sitk.sitkFloat32)
        return self.transform(input, mask)


