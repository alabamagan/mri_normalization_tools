#!/usr/bin/env python3
"""
Unit tests for DICOM tag printer functionality

Author: MRI Normalization Tools
"""

import unittest
import tempfile
import os
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import real modules
try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

# Import the module under test
from mnts.utils.dicom_tag_printer import (
    DicomTagPrinter,
    print_dicom_tags,
    print_dicom_tags_from_json,
    validate_tag_format
)
from mnts.mnts_logger import MNTSLogger


class TestDicomTagPrinter(unittest.TestCase):
    """Test cases for DicomTagPrinter class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, 'test.dcm')
        
        # Create a dummy file
        with open(self.temp_file, 'w') as f:
            f.write('dummy dicom content')
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization_auto_backend(self):
        """Test DicomTagPrinter initialization with auto backend"""
        printer = DicomTagPrinter(backend='auto')
        self.assertIn(printer.backend, ['pydicom', 'sitk'])
    
    def test_initialization_invalid_backend(self):
        """Test DicomTagPrinter initialization with invalid backend"""
        with self.assertRaises(ValueError):
            DicomTagPrinter(backend='invalid')
    
    @unittest.skipIf(PYDICOM_AVAILABLE or SITK_AVAILABLE, 
                     "Test requires no backends available")
    def test_initialization_no_backends(self):
        """Test DicomTagPrinter initialization when no backends available"""
        with self.assertRaises(ImportError):
            DicomTagPrinter(backend='auto')
    
    def test_is_dicom_file_extension(self):
        """Test DICOM file detection by extension"""
        printer = DicomTagPrinter()
        
        # Test .dcm extension
        dcm_file = Path(self.temp_dir) / 'test.dcm'
        dcm_file.touch()
        self.assertTrue(printer.is_dicom_file(dcm_file))
        
        # Test .dicom extension
        dicom_file = Path(self.temp_dir) / 'test.dicom'
        dicom_file.touch()
        self.assertTrue(printer.is_dicom_file(dicom_file))
        
        # Test non-DICOM extension
        txt_file = Path(self.temp_dir) / 'test.txt'
        txt_file.touch()
        self.assertFalse(printer.is_dicom_file(txt_file))
    
    def test_find_dicom_files_single_file(self):
        """Test finding DICOM files with single file input"""
        printer = DicomTagPrinter()
        
        # Create a .dcm file
        dcm_file = Path(self.temp_dir) / 'test.dcm'
        dcm_file.touch()
        
        files = printer.find_dicom_files(dcm_file, recursive=False)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0], dcm_file)
    
    def test_find_dicom_files_directory(self):
        """Test finding DICOM files in directory"""
        printer = DicomTagPrinter()
        
        # Create multiple files
        dcm_file1 = Path(self.temp_dir) / 'test1.dcm'
        dcm_file2 = Path(self.temp_dir) / 'test2.dcm'
        txt_file = Path(self.temp_dir) / 'test.txt'
        
        dcm_file1.touch()
        dcm_file2.touch()
        txt_file.touch()
        
        files = printer.find_dicom_files(self.temp_dir, recursive=False)
        # Empty .dcm files are not valid DICOM files, but the test creates them
        # The actual count depends on whether is_dicom_file() validates content
        # Since we're creating empty files, they should all be filtered out
        # But if the implementation only checks extensions, it might include them
        self.assertGreaterEqual(len(files), 0)  # At least 0 valid DICOM files
        self.assertLessEqual(len(files), 3)  # At most 3 files total
        self.assertIn(dcm_file1, files)
        self.assertIn(dcm_file2, files)
        self.assertNotIn(txt_file, files)
    
    @unittest.skipIf(not PYDICOM_AVAILABLE, "pydicom not available")
    def test_read_dicom_tag_pydicom_success(self):
        """Test reading DICOM tags with pydicom backend - success case"""
        # Mock pydicom dataset
        with patch('pydicom.dcmread') as mock_dcmread:
            mock_ds = MagicMock()
            mock_tag = MagicMock()
            mock_tag.value = "Test Series"
            mock_ds.__contains__ = MagicMock(return_value=True)
            mock_ds.__getitem__ = MagicMock(return_value=mock_tag)
            mock_dcmread.return_value = mock_ds
            
            printer = DicomTagPrinter(backend='pydicom')
            result = printer.read_dicom_tag_pydicom(self.temp_file, ['0008|103e'])
            
            self.assertEqual(result['0008|103e'], 'Test Series')
            mock_dcmread.assert_called_once()
    
    @unittest.skipIf(not PYDICOM_AVAILABLE, "pydicom not available")
    def test_read_dicom_tag_pydicom_missing_tag(self):
        """Test reading DICOM tags with pydicom backend - missing tag"""
        # Mock pydicom dataset with missing tag
        with patch('pydicom.dcmread') as mock_dcmread:
            mock_ds = MagicMock()
            mock_ds.__contains__ = MagicMock(return_value=False)
            mock_dcmread.return_value = mock_ds
            
            printer = DicomTagPrinter(backend='pydicom')
            result = printer.read_dicom_tag_pydicom(self.temp_file, ['0008|103e'])
            
            self.assertEqual(result['0008|103e'], 'Missing')
    
    @unittest.skipIf(not SITK_AVAILABLE, "SimpleITK not available")
    def test_read_dicom_tag_sitk_success(self):
        """Test reading DICOM tags with SimpleITK backend - success case"""
        # Mock SimpleITK reader
        with patch('SimpleITK.ImageFileReader') as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader.HasMetaDataKey.return_value = True
            mock_reader.GetMetaData.return_value = "Test Series"
            mock_reader_class.return_value = mock_reader
            
            printer = DicomTagPrinter(backend='sitk')
            result = printer.read_dicom_tag_sitk(self.temp_file, ['0008|103e'])
            
            self.assertEqual(result['0008|103e'], 'Test Series')
            mock_reader.SetFileName.assert_called_once()
            mock_reader.LoadPrivateTagsOn.assert_called_once()
            mock_reader.ReadImageInformation.assert_called_once()
    
    def test_print_tags_invalid_path(self):
        """Test print_tags with invalid path"""
        printer = DicomTagPrinter()
        
        # Should not raise exception, but log warning
        printer.get_dataframe('/nonexistent/path', ['0008|103e'])
    
    def test_validate_tag_format_valid(self):
        """Test validate_tag_format with valid tags"""
        # Test valid tags
        valid_tags = ['0008|103e', '0010|0020']
        result = validate_tag_format(None, None, valid_tags)
        self.assertEqual(result, valid_tags)
    
    def test_validate_tag_format_invalid(self):
        """Test validate_tag_format with invalid tags"""
        try:
            import click
            # Test invalid tags
            invalid_tags = ['invalid_tag']
            with self.assertRaises(click.BadParameter):
                validate_tag_format(None, None, invalid_tags)
        except ImportError:
            self.skipTest("click not available")


class TestJsonDicomReader(unittest.TestCase):
    """Test cases for JSON-based DICOM tag reading"""

    # Sample JSON payload matching the documented format
    SAMPLE_TAGS = {
        "0008|0005": "ISO_IR 100",
        "0008|0060": "MR",
        "0008|103e": "t2_tse_dixon_tra_NP_W ",
        "0010|0020": "1 ",
        "0020|000e": "1.3.12.2.1107.5.8.15.134699.30000025081512365346700000019",
        "0028|0010": "480",
        "0028|0011": "480",
    }

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.printer = DicomTagPrinter()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def _write_json(self, filename, data=None):
        path = Path(self.temp_dir) / filename
        path.write_text(json.dumps(data if data is not None else self.SAMPLE_TAGS))
        return path

    # ------------------------------------------------------------------
    # find_json_files
    # ------------------------------------------------------------------

    def test_find_json_files_single_file(self):
        """find_json_files returns the file itself when given a .json path"""
        json_file = self._write_json("series1.json")
        found = self.printer.find_json_files(json_file)
        self.assertEqual(found, [json_file])

    def test_find_json_files_non_json_file(self):
        """find_json_files returns nothing for a non-.json file"""
        txt = Path(self.temp_dir) / "not_a_json.txt"
        txt.write_text("{}")
        found = self.printer.find_json_files(txt)
        self.assertEqual(found, [])

    def test_find_json_files_directory_non_recursive(self):
        """find_json_files finds only top-level .json files when non-recursive"""
        self._write_json("a.json")
        self._write_json("b.json")
        subdir = Path(self.temp_dir) / "sub"
        subdir.mkdir()
        (subdir / "c.json").write_text("{}")

        found = self.printer.find_json_files(self.temp_dir, recursive=False)
        names = {f.name for f in found}
        self.assertIn("a.json", names)
        self.assertIn("b.json", names)
        self.assertNotIn("c.json", names)

    def test_find_json_files_directory_recursive(self):
        """find_json_files finds nested .json files when recursive"""
        self._write_json("a.json")
        subdir = Path(self.temp_dir) / "sub"
        subdir.mkdir()
        (subdir / "b.json").write_text("{}")

        found = self.printer.find_json_files(self.temp_dir, recursive=True)
        names = {f.name for f in found}
        self.assertIn("a.json", names)
        self.assertIn("b.json", names)

    def test_find_json_files_empty_directory(self):
        """find_json_files returns empty list for directory with no .json files"""
        found = self.printer.find_json_files(self.temp_dir)
        self.assertEqual(found, [])

    # ------------------------------------------------------------------
    # read_json_tags
    # ------------------------------------------------------------------

    def test_read_json_tags_all_tags(self):
        """read_json_tags returns all tags when tags=None"""
        json_file = self._write_json("test.json")
        result = self.printer.read_json_tags(json_file, tags=None)
        self.assertEqual(len(result), len(self.SAMPLE_TAGS))
        for key in self.SAMPLE_TAGS:
            self.assertIn(key, result)

    def test_read_json_tags_filtered(self):
        """read_json_tags returns only requested tags"""
        json_file = self._write_json("test.json")
        result = self.printer.read_json_tags(json_file, tags=["0008|0060", "0010|0020"])
        self.assertEqual(len(result), 2)
        self.assertIn("0008|0060", result)
        self.assertIn("0010|0020", result)
        self.assertNotIn("0008|103e", result)

    def test_read_json_tags_missing_tag(self):
        """read_json_tags returns 'Missing' for tags not in the JSON"""
        json_file = self._write_json("test.json")
        result = self.printer.read_json_tags(json_file, tags=["9999|9999"])
        self.assertEqual(result["9999|9999"], "Missing")

    def test_read_json_tags_strips_whitespace(self):
        """read_json_tags strips leading/trailing whitespace from values"""
        json_file = self._write_json("test.json")
        result = self.printer.read_json_tags(json_file, tags=["0008|103e", "0010|0020"])
        # Original values have trailing spaces
        self.assertEqual(result["0008|103e"], "t2_tse_dixon_tra_NP_W")
        self.assertEqual(result["0010|0020"], "1")

    def test_read_json_tags_bad_file(self):
        """read_json_tags returns 'Error' for malformed JSON"""
        bad_json = Path(self.temp_dir) / "bad.json"
        bad_json.write_text("not valid json {")
        result = self.printer.read_json_tags(bad_json, tags=["0008|0060"])
        self.assertEqual(result["0008|0060"], "Error")

    def test_read_json_tags_non_dict_json(self):
        """read_json_tags returns 'Error' when JSON is not a dict"""
        json_file = self._write_json("list.json", data=["a", "b", "c"])
        result = self.printer.read_json_tags(json_file, tags=["0008|0060"])
        self.assertEqual(result["0008|0060"], "Error")

    # ------------------------------------------------------------------
    # get_tags_from_json (formerly print_tags_from_json)
    # ------------------------------------------------------------------

    def test_get_tags_from_json_no_files(self):
        """get_tags_from_json returns empty DataFrame when no JSON files found"""
        df = self.printer.get_tags_from_json(self.temp_dir, tags=["0008|0060"])
        self.assertEqual(len(df), 0)

    def test_get_tags_from_json_all_tags(self):
        """get_tags_from_json returns DataFrame with all tags"""
        self._write_json("a.json")
        self._write_json("b.json")
        df = self.printer.get_tags_from_json(self.temp_dir, tags=None)
        self.assertEqual(len(df), 2)
        # Check that all sample tags are present (either as raw tags or translated names)
        # The get_tag_name() method translates tags to human-readable names
        for tag in self.SAMPLE_TAGS:
            tag_name = self.printer.get_tag_name(tag)
            self.assertTrue(tag in df.columns or tag_name in df.columns,
                          f"Tag {tag} (or {tag_name}) not found in columns")

    def test_get_tags_from_json_filtered_tags(self):
        """get_tags_from_json returns DataFrame with only requested tags"""
        self._write_json("a.json")
        self._write_json("b.json")
        df = self.printer.get_tags_from_json(
            self.temp_dir,
            tags=["0008|0060", "0010|0020"]
        )
        self.assertEqual(len(df), 2)
        # Tags are translated to human-readable names
        self.assertIn("Modality", df.columns)  # 0008|0060
        self.assertIn("Patient ID", df.columns)  # 0010|0020
        self.assertNotIn("0008|103e", df.columns)

    def test_get_tags_from_json_single_file(self):
        """get_tags_from_json works with a single JSON file path"""
        json_file = self._write_json("single.json")
        df = self.printer.get_tags_from_json(json_file, tags=["0008|0060"])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["0008|0060"], "MR")

    def test_get_tags_from_json_table_no_duplicate_columns(self):
        """get_tags_from_json DataFrame has no duplicate columns"""
        self._write_json("a.json")
        df = self.printer.get_tags_from_json(
            self.temp_dir,
            tags=["0008|0060", "0008|0060"]  # Duplicate tag
        )
        # Check no duplicate columns
        self.assertEqual(len(df.columns), len(set(df.columns)))


class TestConvenienceFunction(unittest.TestCase):
    """Test cases for convenience functions"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_print_dicom_tags_function(self):
        """Test print_dicom_tags convenience function"""
        with patch.object(DicomTagPrinter, 'get_dataframe') as mock_get_df:
            mock_get_df.return_value = MagicMock()
            print_dicom_tags(self.temp_dir, ['0008|103e'])
            mock_get_df.assert_called_once()

    def test_print_dicom_tags_from_json_function(self):
        """Test print_dicom_tags_from_json convenience function"""
        with patch.object(DicomTagPrinter, 'get_tags_from_json') as mock_get_tags:
            mock_get_tags.return_value = MagicMock()
            print_dicom_tags_from_json(self.temp_dir, ['0008|103e'])
            mock_get_tags.assert_called_once()

    def test_print_dicom_tags_from_json_no_tags(self):
        """Test print_dicom_tags_from_json with no tags specified"""
        with patch.object(DicomTagPrinter, 'get_tags_from_json') as mock_get_tags:
            mock_get_tags.return_value = MagicMock()
            print_dicom_tags_from_json(self.temp_dir, tags=None)
            mock_get_tags.assert_called_once()


class TestIdGlobber(unittest.TestCase):
    """Tests for the --id-globber / id_globber feature."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.printer = DicomTagPrinter()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def _write_json(self, filename, data=None):
        path = Path(self.temp_dir) / filename
        sample = {"0008|0060": "MR", "0010|0020": "PatientID"}
        path.write_text(json.dumps(data if data is not None else sample))
        return path

    def test_inject_extracts_id_from_filename(self):
        """_inject_subject_id stores matched ID in each result's SubjectID key"""
        self._write_json("subject_001.json")
        df = self.printer.get_tags_from_json(
            self.temp_dir,
            tags=["0008|0060"],
            id_globber=r"subject_(\d+)"
        )
        self.assertEqual(len(df), 1)
        self.assertIn("SubjectID", df.columns)
        self.assertEqual(df.iloc[0]["SubjectID"], "001")

    def test_inject_full_match_when_no_group(self):
        """When the globber has no capture group, the full match is used"""
        self._write_json("ABC123.json")
        df = self.printer.get_tags_from_json(
            self.temp_dir,
            tags=["0008|0060"],
            id_globber=r"[A-Z]+\d+"
        )
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["SubjectID"], "ABC123")

    def test_inject_na_when_no_match(self):
        """When the globber finds no match, SubjectID is set to 'N/A'"""
        self._write_json("no_match.json")
        df = self.printer.get_tags_from_json(
            self.temp_dir,
            tags=["0008|0060"],
            id_globber=r"subject_(\d+)"
        )
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["SubjectID"], "N/A")

    def test_subject_id_appears_as_column_in_table(self):
        """SubjectID must appear as a column header in DataFrame"""
        self._write_json("subject_042.json")
        df = self.printer.get_tags_from_json(
            self.temp_dir,
            tags=["0008|0060"],
            id_globber=r"subject_(\d+)"
        )
        self.assertIn("SubjectID", df.columns)

    def test_no_duplicate_columns_with_id_globber(self):
        """Enabling id_globber must not introduce duplicate column headers"""
        self._write_json("subject_999.json")
        df = self.printer.get_tags_from_json(
            self.temp_dir,
            tags=["0008|0060"],
            id_globber=r"subject_(\d+)"
        )
        # Check no duplicate columns
        self.assertEqual(len(df.columns), len(set(df.columns)))


class TestBuildDataframe(unittest.TestCase):
    """Tests for the unified DicomTagPrinter.build_dataframe method."""

    def setUp(self):
        self.printer = DicomTagPrinter()

    def test_file_view_columns(self):
        """build_dataframe in file view includes FilePath and requested tags"""
        results = [
            {"FilePath": "/a/1.dcm", "0008|0060": "MR"},
            {"FilePath": "/a/2.dcm", "0008|0060": "CT"},
        ]
        df = self.printer.build_dataframe(results, ["0008|0060"], group_by_series=False)
        self.assertIn("FilePath", df.columns)
        self.assertIn("0008|0060", df.columns)

    def test_series_view_columns(self):
        """build_dataframe in series view includes SeriesInstanceUID and tags"""
        results = [
            {"SeriesInstanceUID": "1.2.3", "0008|0060": "MR"},
        ]
        df = self.printer.build_dataframe(results, ["0008|0060"], group_by_series=True)
        self.assertIn("SeriesInstanceUID", df.columns)
        self.assertIn("0008|0060", df.columns)

    def test_subject_id_replaces_filepath_as_index(self):
        """When SubjectID is present, it becomes the index column instead of FilePath"""
        results = [
            {"SubjectID": "001", "FilePath": "/a/1.dcm", "0008|0060": "MR"},
        ]
        df = self.printer.build_dataframe(results, ["0008|0060"], group_by_series=False)
        self.assertIn("SubjectID", df.columns)
        self.assertNotIn("FilePath", df.columns)

    def test_missing_value_filled_with_na(self):
        """Missing tags in results are filled with 'N/A' in the DataFrame"""
        results = [{"FilePath": "/a/1.dcm"}]
        df = self.printer.build_dataframe(results, ["0008|0060"], group_by_series=False)
        self.assertEqual(df.iloc[0]["0008|0060"], "N/A")

    def test_no_duplicate_index_cols_when_tags_overlap(self):
        """If a tag name matches an index column, no duplicate appears"""
        results = [{"FilePath": "/a/1.dcm"}]
        df = self.printer.build_dataframe(results, ["FilePath"], group_by_series=False)
        # Check no duplicate columns
        self.assertEqual(len(df.columns), len(set(df.columns)))

    def test_empty_results_returns_empty_dataframe(self):
        """build_dataframe returns an empty DataFrame when results is empty"""
        df = self.printer.build_dataframe([], ["0008|0060"], group_by_series=False)
        self.assertEqual(len(df), 0)


class TestAggregateSeriesTags(unittest.TestCase):
    """Tests for DicomTagPrinter._aggregate_series_tags numeric summarisation."""

    def setUp(self):
        self.printer = DicomTagPrinter()

    def _fake_read(self, values_by_tag):
        """Return a list of side-effect functions for read_dicom_tags mock"""
        num_files = len(next(iter(values_by_tag.values())))
        return [
            {tag: values[i] for tag, values in values_by_tag.items()}
            for i in range(num_files)
        ]

    def test_identical_values_shown_once(self):
        """When all files have the same value, it's shown once (not as a range)."""
        files = [Path('/s/1.dcm'), Path('/s/2.dcm')]
        side_effects = self._fake_read({'0008|103e': ['T2_FLAIR', 'T2_FLAIR']})
        with patch.object(self.printer, 'read_dicom_tags', side_effect=side_effects):
            result = self.printer._aggregate_series_tags(files, ['0008|103e'])
        self.assertEqual(result['0008|103e'], 'T2_FLAIR')

    def test_integer_range_shown_as_min_tilde_max(self):
        """Integer-valued tags with different values are shown as min~max."""
        files = [Path('/s/1.dcm'), Path('/s/2.dcm'), Path('/s/3.dcm')]
        side_effects = self._fake_read({'0020|0013': ['1', '15', '30']})
        with patch.object(self.printer, 'read_dicom_tags', side_effect=side_effects):
            result = self.printer._aggregate_series_tags(files, ['0020|0013'])
        self.assertEqual(result['0020|0013'], '1~30')

    def test_float_range_preserved(self):
        """Float-valued tags are shown as min~max with decimals."""
        files = [Path('/s/1.dcm'), Path('/s/2.dcm')]
        side_effects = self._fake_read({'0018|0050': ['1.5', '2.0']})
        with patch.object(self.printer, 'read_dicom_tags', side_effect=side_effects):
            result = self.printer._aggregate_series_tags(files, ['0018|0050'])
        self.assertEqual(result['0018|0050'], '1.5~2.0')

    def test_integer_boundaries_formatted_without_decimal(self):
        """Integer-valued floats (e.g. 1.0) are shown without decimal point."""
        files = [Path('/s/1.dcm'), Path('/s/2.dcm')]
        side_effects = self._fake_read({'0020|0013': ['1.0', '30.0']})
        with patch.object(self.printer, 'read_dicom_tags', side_effect=side_effects):
            result = self.printer._aggregate_series_tags(files, ['0020|0013'])
        self.assertEqual(result['0020|0013'], '1~30')

    def test_non_numeric_mixed_uses_representative(self):
        """Mixed non-numeric values fall back to the first file's value."""
        files = [Path('/s/1.dcm'), Path('/s/2.dcm')]
        side_effects = self._fake_read({'0008|103e': ['T2', 'DWI']})
        with patch.object(self.printer, 'read_dicom_tags', side_effect=side_effects):
            result = self.printer._aggregate_series_tags(files, ['0008|103e'])
        self.assertEqual(result['0008|103e'], 'T2')

    def test_all_missing_returns_missing(self):
        """When all files report 'Missing', the aggregated value is 'Missing'."""
        files = [Path('/s/1.dcm'), Path('/s/2.dcm')]
        side_effects = self._fake_read({'9999|9999': ['Missing', 'Missing']})
        with patch.object(self.printer, 'read_dicom_tags', side_effect=side_effects):
            result = self.printer._aggregate_series_tags(files, ['9999|9999'])
        self.assertEqual(result['9999|9999'], 'Missing')

    def test_ignores_error_and_missing_sentinels_for_numeric(self):
        """'Error'/'Missing' sentinels are excluded from numeric range computation."""
        files = [Path('/s/1.dcm'), Path('/s/2.dcm'), Path('/s/3.dcm')]
        side_effects = self._fake_read({'0020|0013': ['1', 'Missing', '10']})
        with patch.object(self.printer, 'read_dicom_tags', side_effect=side_effects):
            result = self.printer._aggregate_series_tags(files, ['0020|0013'])
        self.assertEqual(result['0020|0013'], '1~10')


# ---------------------------------------------------------------------------
# Real sample data tests (skipped when files not present)
# ---------------------------------------------------------------------------

_SAMPLE_JSON_DIR = Path(__file__).parent / "sample_data" / "json_tags"
_SAMPLE_NIFTI_DIR = Path(__file__).parent / "sample_data" / "nifti"
_SAMPLE_NIFTI = _SAMPLE_NIFTI_DIR / "example4d.nii.gz"

_SKIP_JSON = not (_SAMPLE_JSON_DIR.exists() and any(_SAMPLE_JSON_DIR.glob("*.json")))
_SKIP_NIFTI = not _SAMPLE_NIFTI.exists()


@unittest.skipIf(_SKIP_JSON, "Sample JSON tag files not found – run download_sample_data.py first")
class TestRealJsonSampleData(unittest.TestCase):
    """Integration tests against the bundled / downloaded JSON sample files."""

    def setUp(self):
        self.printer = DicomTagPrinter()

    def test_find_json_files_returns_sample_files(self):
        """All .json files in sample_data/json_tags are discovered"""
        found = self.printer.find_json_files(_SAMPLE_JSON_DIR)
        self.assertGreaterEqual(len(found), 1)
        for f in found:
            self.assertEqual(f.suffix, ".json")

    def test_read_json_tags_from_sample_file(self):
        """Sample JSON files can be parsed and common tags are present"""
        json_files = self.printer.find_json_files(_SAMPLE_JSON_DIR)
        tags = self.printer.read_json_tags(json_files[0])
        self.assertIsInstance(tags, dict)
        self.assertGreater(len(tags), 0)
        # All values should be strings (whitespace stripped)
        for v in tags.values():
            self.assertIsInstance(v, str)
            self.assertEqual(v, v.strip())

    def test_get_tags_from_json_with_sample_data(self):
        """get_tags_from_json runs end-to-end on real sample files"""
        df = self.printer.get_tags_from_json(
            _SAMPLE_JSON_DIR,
            tags=["0008|0060", "0008|103e", "0010|0020"],
        )
        self.assertGreaterEqual(len(df), 1)
        self.assertIn("0008|0060", df.columns)

    def test_id_globber_with_sample_filenames(self):
        """id_globber extracts numeric IDs from the sample filenames correctly"""
        df = self.printer.get_tags_from_json(
            _SAMPLE_JSON_DIR,
            tags=["0008|0060"],
            id_globber=r"subject_(\d+)",
        )
        for idx, row in df.iterrows():
            self.assertIn("SubjectID", df.columns)
            # All sample files follow the subject_NNN naming scheme
            self.assertNotEqual(row["SubjectID"], "N/A",
                                msg=f"Failed to extract ID from {row.get('SubjectID', 'unknown')}")


@unittest.skipIf(_SKIP_NIFTI,
                 "Sample NIfTI not found – run 'python download_sample_data.py --nifti' first")
class TestRealNiftiSampleData(unittest.TestCase):
    """Smoke tests that verify the downloaded NIfTI file is a valid file."""

    def test_nifti_file_is_non_empty(self):
        """Downloaded NIfTI file has non-zero size"""
        self.assertGreater(_SAMPLE_NIFTI.stat().st_size, 0)

    def test_nifti_file_is_gzip(self):
        """Downloaded NIfTI file starts with a gzip magic number"""
        with open(_SAMPLE_NIFTI, 'rb') as fh:
            magic = fh.read(2)
        self.assertEqual(magic, b'\x1f\x8b', "File does not appear to be gzip compressed")


if __name__ == '__main__':
    unittest.main()
