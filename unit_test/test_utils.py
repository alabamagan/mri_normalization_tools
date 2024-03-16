import unittest
import tempfile
from pathlib import Path
from mnts.utils import *
from mnts.mnts_logger import MNTSLogger

class TestUtils(unittest.TestCase):
    def setUp(self):
        # Mock the iterdir() to return a list of Path objects
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)
        self.source_dir = self.tmp_dir / 'source'
        self.target_dir = self.tmp_dir / 'target'

        self.source_dir.mkdir()
        self.target_dir.mkdir()

        for i in range(4):
            (self.source_dir / f'{i}.txt').touch()
            (self.target_dir / f'{i}.txt').touch()

    def tearDown(self):
        self.tmp.cleanup()

    def test_get_unique_IDs(self):
        target_ids = ['0', '1', '2', '3']
        id_list = get_unique_IDs(self.source_dir.iterdir(), globber='\d+')
        self.assertEqual(target_ids, id_list)

    def test_load_supervised_pair_by_IDs(self):
        id_list = ['1', '2']

        # Call the function under test
        source_files, target_files = load_supervised_pair_by_IDs(
            self.source_dir,
            self.target_dir,
            id_list,
            globber=r"\d+"
        )

        # Convert Path objects to strings for assertion comparison
        source_files_str = [str(f).replace(self.tmp.name, '') for f in source_files]
        target_files_str = [str(f).replace(self.tmp.name, '') for f in target_files]

        # Assertions
        self.assertEqual(len(source_files_str), len(target_files_str))
        self.assertEqual(len(source_files_str), 2)
        self.assertIn('/source/1.txt', source_files_str)
        self.assertIn('/source/2.txt', source_files_str)
        self.assertIn('/target/1.txt', target_files_str)
        self.assertIn('/target/2.txt', target_files_str)

