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
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Stub optional third-party packages that may not be installed so that the
# test module can always be imported regardless of the runtime environment.
# ---------------------------------------------------------------------------
def _make_stub(name):
    mod = types.ModuleType(name)
    sys.modules.setdefault(name, mod)
    return mod

# SimpleITK stub (needed by mnts/__init__.py and dicom_tag_printer.py)
_sitk = _make_stub('SimpleITK')
_sitk.ProcessObject_GlobalWarningDisplayOff = lambda: None
_sitk.ImageFileReader = MagicMock

# pydicom stubs
_pydicom = _make_stub('pydicom')
_pydicom.dcmread = MagicMock()
_pydicom.tag = types.SimpleNamespace(Tag=MagicMock())
_pydicom_dd = _make_stub('pydicom.datadict')
_pydicom_dd.dictionary_description = MagicMock(return_value='Unknown')

# rich stubs
_make_stub('rich')
_rich_table = _make_stub('rich.table')
_rich_table.Table = MagicMock
_rich_console = _make_stub('rich.console')
_rich_console.Console = MagicMock
_rich_logging = _make_stub('rich.logging')
_rich_logging.RichHandler = MagicMock

# rich.progress stubs — Progress is used as a context manager with .add_task / .advance
class _FakeTask:
    pass
class _FakeProgress:
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def add_task(self, *a, **kw): return _FakeTask()
    def advance(self, *a, **kw): pass
_rich_progress = _make_stub('rich.progress')
_rich_progress.Progress = _FakeProgress
_rich_progress.SpinnerColumn = MagicMock
_rich_progress.BarColumn = MagicMock
_rich_progress.TextColumn = MagicMock
_rich_progress.MofNCompleteColumn = MagicMock

# click stub
_click = _make_stub('click')
_click.BadParameter = Exception

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Stub internal relative imports that dicom_tag_printer.py depends on.
_mnts_logger_mod = _make_stub('mnts.mnts_logger')

class _FakeLogger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
    def debug(self, *a, **k): pass
    class _inner:
        handlers = []
    _logger = _inner()

class _FakeMNTSLogger:
    def __class_getitem__(cls, item):
        return _FakeLogger()
    @staticmethod
    def set_global_log_level(level): pass

_mnts_logger_mod.MNTSLogger = _FakeMNTSLogger

_mnts_io_fmt = _make_stub('mnts.io.data_formatting')
_mnts_io_fmt.pydicom_read_series = MagicMock(return_value={})

# Load the target module directly from its file to bypass mnts/__init__.py.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    'mnts.utils.dicom_tag_printer',
    str(Path(__file__).parent.parent / 'mnts' / 'utils' / 'dicom_tag_printer.py')
)
_dtp_module = _ilu.module_from_spec(_spec)
sys.modules['mnts.utils.dicom_tag_printer'] = _dtp_module
_spec.loader.exec_module(_dtp_module)

DicomTagPrinter = _dtp_module.DicomTagPrinter
print_dicom_tags = _dtp_module.print_dicom_tags
print_dicom_tags_from_json = _dtp_module.print_dicom_tags_from_json
validate_tag_format = _dtp_module.validate_tag_format


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
    
    @patch('mnts.utils.dicom_tag_printer.PYDICOM_AVAILABLE', False)
    @patch('mnts.utils.dicom_tag_printer.SITK_AVAILABLE', False)
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
        self.assertEqual(len(files), 2)
        self.assertIn(dcm_file1, files)
        self.assertIn(dcm_file2, files)
        self.assertNotIn(txt_file, files)
    
    @patch('mnts.utils.dicom_tag_printer.pydicom')
    def test_read_dicom_tag_pydicom_success(self, mock_pydicom):
        """Test reading DICOM tags with pydicom backend - success case"""
        # Mock pydicom dataset
        mock_ds = MagicMock()
        mock_tag = MagicMock()
        mock_tag.value = "Test Series"
        mock_ds.__contains__ = MagicMock(return_value=True)
        mock_ds.__getitem__ = MagicMock(return_value=mock_tag)
        mock_pydicom.dcmread.return_value = mock_ds
        mock_pydicom.tag.Tag.return_value = 'mocked_tag'
        
        printer = DicomTagPrinter(backend='pydicom')
        result = printer.read_dicom_tag_pydicom(self.temp_file, ['0008|103e'])
        
        self.assertEqual(result['0008|103e'], 'Test Series')
        mock_pydicom.dcmread.assert_called_once()
    
    @patch('mnts.utils.dicom_tag_printer.pydicom')
    def test_read_dicom_tag_pydicom_missing_tag(self, mock_pydicom):
        """Test reading DICOM tags with pydicom backend - missing tag"""
        # Mock pydicom dataset with missing tag
        mock_ds = MagicMock()
        mock_ds.__contains__ = MagicMock(return_value=False)
        mock_pydicom.dcmread.return_value = mock_ds
        mock_pydicom.tag.Tag.return_value = 'mocked_tag'
        
        printer = DicomTagPrinter(backend='pydicom')
        result = printer.read_dicom_tag_pydicom(self.temp_file, ['0008|103e'])
        
        self.assertEqual(result['0008|103e'], 'Missing')
    
    @patch('mnts.utils.dicom_tag_printer.sitk')
    def test_read_dicom_tag_sitk_success(self, mock_sitk):
        """Test reading DICOM tags with SimpleITK backend - success case"""
        # Mock SimpleITK reader
        mock_reader = MagicMock()
        mock_reader.HasMetaDataKey.return_value = True
        mock_reader.GetMetaData.return_value = "Test Series"
        mock_sitk.ImageFileReader.return_value = mock_reader
        
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
        printer.print_tags('/nonexistent/path', ['0008|103e'])
    
    def test_console_entry_click_not_available(self):
        """Test console entry when click is not available"""
        from mnts.utils.dicom_tag_printer import console_entry
        
        with patch('mnts.utils.dicom_tag_printer.CLICK_AVAILABLE', False):
            with patch('sys.exit') as mock_exit:
                console_entry()
                mock_exit.assert_called_with(1)
    
    @patch('mnts.utils.dicom_tag_printer.click')
    def test_validate_tag_format_valid(self, mock_click):
        """Test validate_tag_format with valid tags"""
        from mnts.utils.dicom_tag_printer import validate_tag_format
        
        # Test valid tags
        valid_tags = ['0008|103e', '0010|0020']
        result = validate_tag_format(None, None, valid_tags)
        self.assertEqual(result, valid_tags)
    
    @patch('mnts.utils.dicom_tag_printer.click')
    def test_validate_tag_format_invalid(self, mock_click):
        """Test validate_tag_format with invalid tags"""
        from mnts.utils.dicom_tag_printer import validate_tag_format
        
        # Mock click.BadParameter
        mock_click.BadParameter = Exception
        
        # Test invalid tags
        invalid_tags = ['invalid_tag']
        with self.assertRaises(Exception):
            validate_tag_format(None, None, invalid_tags)


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
        """read_json_tags returns all tags when no filter is given"""
        json_file = self._write_json("series.json")
        result = self.printer.read_json_tags(json_file)
        # All keys present and values are stripped strings
        self.assertEqual(result["0008|0060"], "MR")
        self.assertEqual(result["0008|103e"], "t2_tse_dixon_tra_NP_W")
        self.assertEqual(result["0010|0020"], "1")

    def test_read_json_tags_filtered(self):
        """read_json_tags returns only the requested tags"""
        json_file = self._write_json("series.json")
        result = self.printer.read_json_tags(json_file, tags=["0008|0060", "0028|0010"])
        self.assertIn("0008|0060", result)
        self.assertIn("0028|0010", result)
        self.assertNotIn("0008|103e", result)

    def test_read_json_tags_missing_tag(self):
        """read_json_tags marks absent tags as 'Missing'"""
        json_file = self._write_json("series.json")
        result = self.printer.read_json_tags(json_file, tags=["0008|0060", "9999|9999"])
        self.assertEqual(result["0008|0060"], "MR")
        self.assertEqual(result["9999|9999"], "Missing")

    def test_read_json_tags_strips_whitespace(self):
        """read_json_tags strips leading/trailing whitespace from values"""
        json_file = self._write_json("series.json", {"0008|103e": "  padded value  "})
        result = self.printer.read_json_tags(json_file)
        self.assertEqual(result["0008|103e"], "padded value")

    def test_read_json_tags_bad_file(self):
        """read_json_tags returns Error entries for an unreadable file"""
        bad = Path(self.temp_dir) / "bad.json"
        bad.write_text("not valid json{{{{")
        result = self.printer.read_json_tags(bad, tags=["0008|0060"])
        self.assertEqual(result["0008|0060"], "Error")

    def test_read_json_tags_non_dict_json(self):
        """read_json_tags returns Error entries when JSON root is not an object"""
        json_file = Path(self.temp_dir) / "list.json"
        json_file.write_text(json.dumps([1, 2, 3]))
        result = self.printer.read_json_tags(json_file, tags=["0008|0060"])
        self.assertEqual(result["0008|0060"], "Error")

    # ------------------------------------------------------------------
    # print_tags_from_json
    # ------------------------------------------------------------------

    def test_print_tags_from_json_no_files(self):
        """print_tags_from_json logs a warning and returns when no JSON files exist"""
        # Empty directory — should not raise
        self.printer.print_tags_from_json(self.temp_dir, output_format='json')

    def test_print_tags_from_json_all_tags(self):
        """print_tags_from_json builds results for all tags when none are specified"""
        self._write_json("s1.json")
        self._write_json("s2.json")

        captured = []
        with patch('builtins.print', side_effect=captured.append):
            self.printer.print_tags_from_json(self.temp_dir, output_format='json')

        output = "\n".join(str(x) for x in captured)
        data = json.loads(output)
        self.assertEqual(len(data), 2)
        self.assertIn("FilePath", data[0])
        self.assertIn("0008|0060", data[0])

    def test_print_tags_from_json_filtered_tags(self):
        """print_tags_from_json honours the tags filter"""
        self._write_json("s1.json")

        captured = []
        with patch('builtins.print', side_effect=captured.append):
            self.printer.print_tags_from_json(
                self.temp_dir,
                tags=["0008|0060", "0028|0010"],
                output_format='json'
            )

        output = "\n".join(str(x) for x in captured)
        data = json.loads(output)
        self.assertIn("0008|0060", data[0])
        self.assertIn("0028|0010", data[0])
        self.assertNotIn("0008|103e", data[0])

    def test_print_tags_from_json_single_file(self):
        """print_tags_from_json works with a direct path to a .json file"""
        json_file = self._write_json("single.json")

        captured = []
        with patch('builtins.print', side_effect=captured.append):
            self.printer.print_tags_from_json(json_file, output_format='json')

        output = "\n".join(str(x) for x in captured)
        data = json.loads(output)
        self.assertEqual(len(data), 1)
        self.assertTrue(data[0]["FilePath"].endswith("single.json"))

    def test_print_tags_from_json_table_no_duplicate_columns(self):
        """_print_table must not produce duplicate File Path or tag columns"""
        self._write_json("s1.json")

        dtp_mod = sys.modules['mnts.utils.dicom_tag_printer']

        # Provide a fake RichHandler on the logger so _print_table takes the
        # rich_handler branch and never falls back to Console().
        mock_console = MagicMock()
        mock_handler = MagicMock()
        mock_handler.console = mock_console
        self.printer.logger._logger.handlers = [mock_handler]

        table_instance = MagicMock()

        with patch.object(dtp_mod, 'Table', return_value=table_instance):
            self.printer.print_tags_from_json(
                self.temp_dir,
                tags=["0008|0060", "0028|0010"],
                output_format='table'
            )

        headers = [call.args[0] for call in table_instance.add_column.call_args_list]
        self.assertEqual(len(headers), len(set(headers)),
                         f"Duplicate columns detected: {headers}")


class TestConvenienceFunction(unittest.TestCase):
    """Test cases for convenience functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('mnts.utils.dicom_tag_printer.DicomTagPrinter')
    def test_print_dicom_tags_function(self, mock_printer_class):
        """Test print_dicom_tags convenience function"""
        mock_printer = MagicMock()
        mock_printer_class.return_value = mock_printer

        print_dicom_tags(self.temp_dir, ['0008|103e'], output_format='csv')

        mock_printer_class.assert_called_once()
        mock_printer.print_tags.assert_called_once_with(
            self.temp_dir, ['0008|103e'], output_format='csv'
        )

    @patch('mnts.utils.dicom_tag_printer.DicomTagPrinter')
    def test_print_dicom_tags_from_json_function(self, mock_printer_class):
        """Test print_dicom_tags_from_json convenience function"""
        mock_printer = MagicMock()
        mock_printer_class.return_value = mock_printer

        print_dicom_tags_from_json(self.temp_dir, tags=['0008|0060'], output_format='csv')

        mock_printer_class.assert_called_once()
        mock_printer.print_tags_from_json.assert_called_once_with(
            self.temp_dir, ['0008|0060'], output_format='csv'
        )

    @patch('mnts.utils.dicom_tag_printer.DicomTagPrinter')
    def test_print_dicom_tags_from_json_no_tags(self, mock_printer_class):
        """print_dicom_tags_from_json passes None when tags are omitted"""
        mock_printer = MagicMock()
        mock_printer_class.return_value = mock_printer

        print_dicom_tags_from_json(self.temp_dir)

        mock_printer.print_tags_from_json.assert_called_once_with(
            self.temp_dir, None
        )


# ---------------------------------------------------------------------------
# ID-globber tests
# ---------------------------------------------------------------------------

class TestIdGlobber(unittest.TestCase):
    """Tests for the --id-globber / id_globber feature."""

    SAMPLE_TAGS = {
        "0008|0060": "MR",
        "0008|103e": "t2_tse_dixon_tra_NP_W",
        "0010|0020": "001",
    }

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.printer = DicomTagPrinter()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def _write_json(self, filename, data=None):
        path = Path(self.temp_dir) / filename
        path.write_text(json.dumps(data or self.SAMPLE_TAGS))
        return path

    # ------------------------------------------------------------------
    # _inject_subject_id
    # ------------------------------------------------------------------

    def test_inject_extracts_id_from_filename(self):
        """_inject_subject_id stores matched ID in each result's SubjectID key"""
        self._write_json("subject_001_scan.json")
        self._write_json("subject_002_scan.json")

        captured = []
        with patch('builtins.print', side_effect=captured.append):
            self.printer.print_tags_from_json(
                self.temp_dir,
                tags=["0008|0060"],
                output_format='json',
                id_globber=r"subject_(\d+)",
            )

        data = json.loads("\n".join(str(x) for x in captured))
        ids = {r['SubjectID'] for r in data}
        self.assertEqual(ids, {"001", "002"})

    def test_inject_full_match_when_no_group(self):
        """When the globber has no capture group, the full match is used"""
        self._write_json("NPC001_scan.json")

        captured = []
        with patch('builtins.print', side_effect=captured.append):
            self.printer.print_tags_from_json(
                self.temp_dir,
                tags=["0008|0060"],
                output_format='json',
                id_globber=r"NPC\d+",
            )

        data = json.loads("\n".join(str(x) for x in captured))
        self.assertEqual(data[0]['SubjectID'], "NPC001")

    def test_inject_na_when_no_match(self):
        """When the globber finds no match, SubjectID is set to 'N/A'"""
        self._write_json("unrecognised_filename.json")

        captured = []
        with patch('builtins.print', side_effect=captured.append):
            self.printer.print_tags_from_json(
                self.temp_dir,
                tags=["0008|0060"],
                output_format='json',
                id_globber=r"NPC\d+",
            )

        data = json.loads("\n".join(str(x) for x in captured))
        self.assertEqual(data[0]['SubjectID'], "N/A")

    def test_subject_id_appears_as_column_in_table(self):
        """SubjectID must appear as a column header in table output"""
        self._write_json("subject_007_scan.json")

        dtp_mod = sys.modules['mnts.utils.dicom_tag_printer']
        table_instance = MagicMock()
        mock_handler = MagicMock()
        mock_handler.console = MagicMock()
        self.printer.logger._logger.handlers = [mock_handler]

        with patch.object(dtp_mod, 'Table', return_value=table_instance):
            self.printer.print_tags_from_json(
                self.temp_dir,
                tags=["0008|0060"],
                output_format='table',
                id_globber=r"subject_(\d+)",
            )

        headers = [call.args[0] for call in table_instance.add_column.call_args_list]
        self.assertIn("SubjectID", headers)
        # SubjectID must come before the DICOM tag column
        self.assertLess(headers.index("SubjectID"), headers.index("0008|0060"))

    def test_no_duplicate_columns_with_id_globber(self):
        """Enabling id_globber must not introduce duplicate column headers"""
        self._write_json("subject_001_scan.json")

        dtp_mod = sys.modules['mnts.utils.dicom_tag_printer']
        table_instance = MagicMock()
        mock_handler = MagicMock()
        mock_handler.console = MagicMock()
        self.printer.logger._logger.handlers = [mock_handler]

        with patch.object(dtp_mod, 'Table', return_value=table_instance):
            self.printer.print_tags_from_json(
                self.temp_dir,
                tags=["0008|0060", "0008|103e"],
                output_format='table',
                id_globber=r"subject_(\d+)",
            )

        headers = [call.args[0] for call in table_instance.add_column.call_args_list]
        self.assertEqual(len(headers), len(set(headers)),
                         f"Duplicate columns: {headers}")


# ---------------------------------------------------------------------------
# build_dataframe tests
# ---------------------------------------------------------------------------

class TestBuildDataframe(unittest.TestCase):
    """Tests for the unified DicomTagPrinter.build_dataframe method."""

    def setUp(self):
        self.printer = DicomTagPrinter()

    def test_file_view_columns(self):
        """Non-series mode produces FilePath + tag columns in order."""
        results = [
            {'FilePath': '/a/1.dcm', '0008|0060': 'MR', '0010|0020': '001'},
            {'FilePath': '/a/2.dcm', '0008|0060': 'CT', '0010|0020': '002'},
        ]
        df = self.printer.build_dataframe(results, ['0008|0060', '0010|0020'],
                                          group_by_series=False)
        self.assertEqual(list(df.columns), ['FilePath', '0008|0060', '0010|0020'])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['FilePath'], '/a/1.dcm')
        self.assertEqual(df.iloc[1]['0008|0060'], 'CT')

    def test_series_view_columns(self):
        """Series mode produces SeriesUID/FileCount/RepresentativeFile + tag columns."""
        results = [
            {'SeriesUID': 'uid-1', 'FileCount': 10, 'RepresentativeFile': '/a/1.dcm',
             '0008|103e': 'T2'},
        ]
        df = self.printer.build_dataframe(results, ['0008|103e'], group_by_series=True)
        self.assertEqual(list(df.columns),
                         ['SeriesUID', 'FileCount', 'RepresentativeFile', '0008|103e'])
        self.assertEqual(df.iloc[0]['SeriesUID'], 'uid-1')
        self.assertEqual(df.iloc[0]['FileCount'], 10)

    def test_subject_id_replaces_filepath_as_index(self):
        """When SubjectID is present it replaces FilePath as the primary index column."""
        results = [
            {'FilePath': '/a/NPC001.dcm', 'SubjectID': 'NPC001', '0008|0060': 'MR'},
        ]
        df = self.printer.build_dataframe(results, ['0008|0060'],
                                          group_by_series=False)
        # SubjectID is the index; FilePath is dropped from the output
        self.assertEqual(list(df.columns), ['SubjectID', '0008|0060'])
        self.assertEqual(df.iloc[0]['SubjectID'], 'NPC001')

    def test_missing_value_filled_with_na(self):
        """Keys absent from a result dict are filled with 'N/A'."""
        results = [{'FilePath': '/a/1.dcm', '0008|0060': 'MR'}]
        df = self.printer.build_dataframe(results, ['0008|0060', '9999|9999'],
                                          group_by_series=False)
        self.assertEqual(df.iloc[0]['9999|9999'], 'N/A')

    def test_no_duplicate_index_cols_when_tags_overlap(self):
        """Index columns are never duplicated even if tags list contains them."""
        results = [{'FilePath': '/a/1.dcm', '0008|0060': 'MR'}]
        # Malformed tags list includes the index col name — must not duplicate it
        df = self.printer.build_dataframe(results, ['FilePath', '0008|0060'],
                                          group_by_series=False)
        self.assertEqual(list(df.columns).count('FilePath'), 1)

    def test_empty_results_returns_empty_dataframe(self):
        """Empty results list produces a DataFrame with correct columns but no rows."""
        df = self.printer.build_dataframe([], ['0008|0060'], group_by_series=False)
        self.assertEqual(len(df), 0)
        self.assertIn('FilePath', df.columns)
        self.assertIn('0008|0060', df.columns)


# ---------------------------------------------------------------------------
# _aggregate_series_tags tests
# ---------------------------------------------------------------------------

class TestAggregateSeriesTags(unittest.TestCase):
    """Tests for DicomTagPrinter._aggregate_series_tags numeric summarisation."""

    def setUp(self):
        self.printer = DicomTagPrinter()

    def _fake_read(self, values_by_tag):
        """Return a side_effect list for read_dicom_tags calls.

        *values_by_tag* maps tag -> list of values (one per simulated file).
        """
        n = max(len(v) for v in values_by_tag.values())
        calls = []
        for i in range(n):
            calls.append({tag: vals[i] for tag, vals in values_by_tag.items()})
        return calls

    def test_identical_values_shown_once(self):
        """Tags with the same value across all files show the single value."""
        files = [Path('/s/1.dcm'), Path('/s/2.dcm'), Path('/s/3.dcm')]
        side_effects = self._fake_read({'0008|103e': ['T2', 'T2', 'T2']})
        with patch.object(self.printer, 'read_dicom_tags', side_effect=side_effects):
            result = self.printer._aggregate_series_tags(files, ['0008|103e'])
        self.assertEqual(result['0008|103e'], 'T2')

    def test_integer_range_shown_as_min_tilde_max(self):
        """Numeric tags that differ across files are shown as 'min~max'."""
        files = [Path(f'/s/{i}.dcm') for i in range(5)]
        values = [str(i + 1) for i in range(5)]  # '1' .. '5'
        side_effects = self._fake_read({'0020|0013': values})
        with patch.object(self.printer, 'read_dicom_tags', side_effect=side_effects):
            result = self.printer._aggregate_series_tags(files, ['0020|0013'])
        self.assertEqual(result['0020|0013'], '1~5')

    def test_float_range_preserved(self):
        """Non-integer float boundaries are shown with their decimal component."""
        files = [Path('/s/1.dcm'), Path('/s/2.dcm')]
        side_effects = self._fake_read({'0018|0050': ['2.5', '7.5']})
        with patch.object(self.printer, 'read_dicom_tags', side_effect=side_effects):
            result = self.printer._aggregate_series_tags(files, ['0018|0050'])
        self.assertEqual(result['0018|0050'], '2.5~7.5')

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

    def test_print_tags_from_json_with_sample_data(self):
        """print_tags_from_json runs end-to-end on real sample files"""
        captured = []
        with patch('builtins.print', side_effect=captured.append):
            self.printer.print_tags_from_json(
                _SAMPLE_JSON_DIR,
                tags=["0008|0060", "0008|103e", "0010|0020"],
                output_format='json',
            )
        output = "\n".join(str(x) for x in captured)
        data = json.loads(output)
        self.assertGreaterEqual(len(data), 1)
        for entry in data:
            self.assertIn("FilePath", entry)
            self.assertIn("0008|0060", entry)

    def test_id_globber_with_sample_filenames(self):
        """id_globber extracts numeric IDs from the sample filenames correctly"""
        captured = []
        with patch('builtins.print', side_effect=captured.append):
            self.printer.print_tags_from_json(
                _SAMPLE_JSON_DIR,
                tags=["0008|0060"],
                output_format='json',
                id_globber=r"subject_(\d+)",
            )
        output = "\n".join(str(x) for x in captured)
        data = json.loads(output)
        for entry in data:
            self.assertIn("SubjectID", entry)
            # All sample files follow the subject_NNN naming scheme
            self.assertNotEqual(entry["SubjectID"], "N/A",
                                msg=f"Failed to extract ID from {entry.get('FilePath', entry.get('SubjectID', 'unknown'))}")


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