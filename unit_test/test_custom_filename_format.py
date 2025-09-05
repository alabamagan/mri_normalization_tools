from mnts.io.data_formatting import Dcm2NiiConverter, parse_custom_filename_format
from pathlib import Path
import tempfile
import unittest
from mnts.mnts_logger import MNTSLogger

class Test_CustomFilenameFormat(unittest.TestCase):
    """Test cases for the custom filename format functionality."""
    
    @classmethod
    def setUpClass(cls) -> None:
        cls.logger = MNTSLogger['Test Custom Filename Format']
        cls.logger.set_verbose(True)
        MNTSLogger.set_global_log_level('debug')

    @classmethod
    def tearDownClass(cls) -> None:
        cls.logger.cleanup()

    def setUp(self) -> None:
        self.dcm_dir = Path("./sample_data/")
        self.output = tempfile.TemporaryDirectory()
        self.output_dir = self.output.name
        
        # Mock DICOM tags for testing
        self.mock_dicom_tags = {
            '0008|103e': 'T1_MPRAGE_SAG',  # Series Description
            '0010|0020': 'PATIENT_001',     # Patient ID
            '1030|2003': 'PROTOCOL_1',      # Protocol Name
            '0020|0011': '005',             # Series Number
            '0008|0020': '20231201',        # Study Date
        }

    def tearDown(self) -> None:
        self.output.cleanup()

    def test_parse_custom_filename_format_basic(self):
        """Test basic custom filename format parsing."""
        format_string = "`0008|103e`-`0010|0020`"
        result = parse_custom_filename_format(format_string, self.mock_dicom_tags, self.logger)
        expected = "T1_MPRAGE_SAG-PATIENT_001"
        self.assertEqual(result, expected)

    def test_parse_custom_filename_format_complex(self):
        """Test complex custom filename format parsing."""
        format_string = "`0010|0020`-`0008|103e`-`1030|2003`"
        result = parse_custom_filename_format(format_string, self.mock_dicom_tags, self.logger)
        expected = "PATIENT_001-T1_MPRAGE_SAG-PROTOCOL_1"
        self.assertEqual(result, expected)

    def test_parse_custom_filename_format_missing_tag(self):
        """Test handling of missing DICOM tags."""
        format_string = "`0008|103e`-`9999|9999`"  # 9999|9999 doesn't exist
        result = parse_custom_filename_format(format_string, self.mock_dicom_tags, self.logger)
        expected = "T1_MPRAGE_SAG-Missing"
        self.assertEqual(result, expected)

    def test_parse_custom_filename_format_special_chars(self):
        """Test handling of special characters in DICOM tag values."""
        special_tags = {
            '0008|103e': 'T1 MPRAGE / SAG',  # Contains spaces and slash
            '0010|0020': 'PATIENT<>001:*',   # Contains special characters
        }
        format_string = "`0008|103e`-`0010|0020`"
        result = parse_custom_filename_format(format_string, special_tags, self.logger)
        expected = "T1_MPRAGE___SAG-PATIENT__001__"  # Special chars replaced with underscores
        self.assertEqual(result, expected)

    def test_parse_custom_filename_format_empty_string(self):
        """Test handling of empty format string."""
        format_string = ""
        result = parse_custom_filename_format(format_string, self.mock_dicom_tags, self.logger)
        expected = ""
        self.assertEqual(result, expected)

    def test_parse_custom_filename_format_none(self):
        """Test handling of None format string."""
        format_string = None
        result = parse_custom_filename_format(format_string, self.mock_dicom_tags, self.logger)
        expected = ""
        self.assertEqual(result, expected)

    def test_dcm2nii_converter_backward_compatibility(self):
        """Test that the Dcm2NiiConverter maintains backward compatibility."""
        # Test old API without custom_filename_format parameter
        converter = Dcm2NiiConverter(
            folder="./test_folder",
            out_dir="./test_output"
        )
        self.assertIsNone(converter.custom_filename_format)

    def test_dcm2nii_converter_with_custom_format(self):
        """Test that the Dcm2NiiConverter accepts custom_filename_format parameter."""
        custom_format = "`0008|103e`-`0010|0020`"
        converter = Dcm2NiiConverter(
            folder="./test_folder",
            out_dir="./test_output",
            custom_filename_format=custom_format
        )
        self.assertEqual(converter.custom_filename_format, custom_format)

    def test_dcm2nii_converter_integration_with_sample_data(self):
        """Test integration with actual DICOM data if available."""
        sample1_path = self.dcm_dir.joinpath('sample1')
        if sample1_path.exists() and any(sample1_path.iterdir()):
            # Test with custom filename format
            converter = Dcm2NiiConverter(
                str(sample1_path),
                out_dir=self.output_dir,
                idglobber="sample[0-9]+",
                custom_filename_format="`0008|103e`-`0010|0020`"
            )
            converter.Execute()
            
            # Check that files were generated
            output_files = list(Path(self.output_dir).iterdir())
            self.assertGreater(len(output_files), 0, "No files generated with custom filename format")
            
            # Check that filenames follow the custom format (at least partially)
            # Note: We can't verify exact format without knowing actual DICOM tag values
            for file in output_files:
                if file.suffix == '.gz':
                    self.assertTrue('-' in file.stem, f"Custom format separator not found in {file.name}")
        else:
            self.skipTest("Sample DICOM data not available")

if __name__ == '__main__':
    unittest.main()