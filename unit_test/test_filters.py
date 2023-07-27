import unittest
from utils import create_dummy_image, create_dummy_segmentation
import mnts

from mnts.filters.geom import *
from mnts.filters.intensity import *
from mnts.filters.data_node import DataNode, TypeCastNode
from mnts.mnts_logger import MNTSLogger
from typing import Callable, Union, Iterable
import SimpleITK as sitk

class Test_Filters(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.logger = MNTSLogger['Test Dicom2nii']
        cls.logger.set_verbose(True)
        MNTSLogger.set_global_log_level('debug')

        # Load default input
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesIDs("./sample_data/sample1")[0]
        reader.SetFileNames(reader.GetGDCMSeriesFileNames("./sample_data/sample1", series))
        cls.test_input = reader.Execute()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.logger.cleanup()

    def test_DataNode(self):
        filter = DataNode()
        output = filter.filter(self.test_input)
        self.assertEqual(output, self.test_input)

    def test_TypeCastNode_normal(self):
        filter1 = TypeCastNode(sitk.sitkInt16)
        output1 = filter1.filter(self.test_input)
        self.assertEqual(output1.GetPixelID(), sitk.sitkInt16)

    def test_TypeCastNode_specify(self):
        filter = TypeCastNode(sitk.sitkUInt8)
        output = filter.filter(self.test_input)
        self.assertEqual(output.GetPixelID(), sitk.sitkUInt8)

    def test_TypeCastNode_reference(self):
        dummy = sitk.Image()
        filter = TypeCastNode(sitk.sitkUInt8)
        output = filter.filter(self.test_input, dummy)
        self.assertEqual(output.GetPixelID(), dummy.GetPixelID())


class Test_IntensityFilters(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.logger = MNTSLogger['Test Dicom2nii']
        cls.logger.set_verbose(True)
        MNTSLogger.set_global_log_level('debug')

        # Load default input
        cls.test_input = create_dummy_image([100, 100, 100])
        cls.test_segment = create_dummy_segmentation(size=[100, 100, 100],
                                                     center=[50, 50, 50],
                                                     cube_size=25)

    def test_ZscoreNorm(self):
        filter = ZScoreNorm()
        output = filter(self.test_input, self.test_segment)

        npout  = sitk.GetArrayFromImage(output)
        npmask = sitk.GetArrayFromImage(self.test_segment)
        mu    = npout[npmask != 0].mean()
        sigma = npout[npmask != 0].std()
        self.assertAlmostEqual(mu   , 0., delta = 1E-10)
        self.assertAlmostEqual(sigma, 1., delta = 1E-10)

    def test_LinearRescale(self):
        target_mu, target_sigma = 10, 20
        filter = LinearRescale(mean = target_mu, std = target_sigma)
        output = filter(self.test_input, self.test_segment)

        npout  = sitk.GetArrayFromImage(output)
        npmask = sitk.GetArrayFromImage(self.test_segment)
        mu    = npout[npmask != 0].mean()
        sigma = npout[npmask != 0].std()
        self.assertAlmostEqual(mu   , target_mu   , delta = 1E-10)
        self.assertAlmostEqual(sigma, target_sigma, delta = 1E-10)