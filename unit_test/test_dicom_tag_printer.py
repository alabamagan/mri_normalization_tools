#!/usr/bin/env python3
"""
Unit tests for DICOM tag printer functionality

Author: MRI Normalization Tools
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mnts.utils.dicom_tag_printer import DicomTagPrinter, print_dicom_tags


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


if __name__ == '__main__':
    unittest.main()