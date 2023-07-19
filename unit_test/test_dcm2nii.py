from mnts.io.data_formatting import DicomConverter
from pathlib import Path
import tempfile
import unittest
from mnts.mnts_logger import MNTSLogger

class Test_Dicom2nii(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.logger = MNTSLogger['Test Dicom2nii']
        cls.logger.set_verbose(True)
        MNTSLogger.set_global_log_level('debug')


    @classmethod
    def tearDownClass(cls) -> None:
        cls.logger.cleanup()

    def setUp(self) -> None:
        self.dcm_dir = Path("./sample_data/sample1/")
        self.output = tempfile.TemporaryDirectory()
        self.output_dir = self.output.name

    def tearDown(self) -> None:
        self.output.cleanup()

    def test_dicom2nii(self):
        converter = DicomConverter(str(self.dcm_dir),
                                   out_dir=self.output_dir,
                                   idglobber="sample[0-9]+",
                                   dump_meta_data=True)
        converter.Execute()
        pass
