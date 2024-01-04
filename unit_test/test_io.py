from mnts.io import pydicom_read_series
from mnts.io.dixon import *
from mnts.mnts_logger import MNTSLogger
import unittest
from pathlib import Path
import tempfile

class Test_IO(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.logger = MNTSLogger['Test IO']
        cls.logger.set_verbose(True)
        MNTSLogger.set_global_log_level('debug')


    @classmethod
    def tearDownClass(cls) -> None:
        cls.logger.cleanup()

    def setUp(self) -> None:
        self.dcm_dir = Path("./sample_data/")
        self.output = tempfile.TemporaryDirectory()
        self.output_dir = self.output.name

    def test_pydicom_read_series(self):
        d = pydicom_read_series(self.dcm_dir)
        self.logger.info(f"Found sequeces: [{','.join(d.keys())}]")
        self.assertGreater(len(d), 0)

    def test_read_dixon(self):
        d = pydicom_read_series(self.dcm_dir)
        d = DIXON_dcm_to_images(d[list(d.keys())[0]])
        self.assertGreater(len(d), 0)