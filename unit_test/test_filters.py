import unittest
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
