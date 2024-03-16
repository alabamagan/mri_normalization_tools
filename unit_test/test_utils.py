import unittest
import tempfile
from pathlib import Path
from mnts.utils import *
from mnts.mnts_logger import MNTSLogger

class TestUtils(unittest.TestCase):
    def test_load_supervised_pair_by_IDs(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Mock the iterdir() to return a list of Path objects
            tmp_dir = Path(tmp)
            source_dir = tmp_dir / 'source'
            target_dir = tmp_dir / 'target'

            source_dir.mkdir()
            target_dir.mkdir()

            for i in range(4):
                (source_dir / f'{i}.txt').touch()
                (target_dir / f'{i}.txt').touch()

            id_list = ['1', '2']

            # Call the function under test
            source_files, target_files = load_supervised_pair_by_IDs(
                source_dir,
                target_dir,
                id_list,
                globber=r"\d+"
            )

            # Convert Path objects to strings for assertion comparison
            source_files_str = [str(f).replace(tmp, '') for f in source_files]
            target_files_str = [str(f).replace(tmp, '') for f in target_files]

            # Assertions
            self.assertEqual(len(source_files_str), len(target_files_str))
            self.assertEqual(len(source_files_str), 2)
            self.assertIn('/source/1.txt', source_files_str)
            self.assertIn('/source/2.txt', source_files_str)
            self.assertIn('/target/1.txt', target_files_str)
            self.assertIn('/target/2.txt', target_files_str)

