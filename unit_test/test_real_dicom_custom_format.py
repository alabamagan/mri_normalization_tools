from mnts.io.data_formatting import Dcm2NiiConverter, parse_custom_filename_format
from pathlib import Path
import tempfile
import unittest
import SimpleITK as sitk
from mnts.mnts_logger import MNTSLogger

class Test_RealDicomCustomFormat(unittest.TestCase):
    """Test cases for custom filename format functionality using real DICOM data."""
    
    @classmethod
    def setUpClass(cls) -> None:
        cls.logger = MNTSLogger['Test Real DICOM Custom Format']
        cls.logger.set_verbose(True)
        MNTSLogger.set_global_log_level('debug')

    @classmethod
    def tearDownClass(cls) -> None:
        cls.logger.cleanup()

    def setUp(self) -> None:
        self.dcm_dir = Path("./sample_data/")
        self.output = tempfile.TemporaryDirectory()
        self.output_dir = self.output.name
        
        # Read actual DICOM tags from sample data
        self.sample_dcm_file = self.dcm_dir / "sample1" / "image-00007.dcm"
        if self.sample_dcm_file.exists():
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(self.sample_dcm_file))
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            self.real_dicom_tags = {k: reader.GetMetaData(k) for k in reader.GetMetaDataKeys()}
        else:
            self.skipTest("Sample DICOM data not available")

    def tearDown(self) -> None:
        self.output.cleanup()

    def test_real_dicom_tags_available(self):
        """Test that we can read actual DICOM tags from sample data."""
        # Check that key tags are present
        expected_tags = ['0008|103e', '0010|0020', '0020|0011', '0008|0020']
        for tag in expected_tags:
            self.assertIn(tag, self.real_dicom_tags, f"Expected DICOM tag {tag} not found")
        
        print(f"Series Description (0008|103e): {self.real_dicom_tags['0008|103e']}")
        print(f"Patient ID (0010|0020): {self.real_dicom_tags['0010|0020']}")
        print(f"Series Number (0020|0011): {self.real_dicom_tags['0020|0011']}")
        print(f"Study Date (0008|0020): {self.real_dicom_tags['0008|0020']}")

    def test_parse_custom_format_with_real_tags(self):
        """Test parse_custom_filename_format with real DICOM tags."""
        # Test using actual DICOM tag values
        format_string = "`0008|103e`-`0010|0020`"
        result = parse_custom_filename_format(format_string, self.real_dicom_tags, self.logger)
        
        # Expected: "ARTERIELLE-0" based on the actual tag values
        expected = f"{self.real_dicom_tags['0008|103e']}-{self.real_dicom_tags['0010|0020']}"
        self.assertEqual(result, expected)
        print(f"Custom format result: {result}")

    def test_dcm2nii_default_vs_custom_filenames(self):
        """Test that default and custom filename formats produce different outputs."""
        sample1_path = self.dcm_dir / "sample1"
        
        # Test with default filename format (no custom format)
        output_default = tempfile.TemporaryDirectory()
        converter_default = Dcm2NiiConverter(
            str(sample1_path),
            out_dir=output_default.name,
            idglobber="sample[0-9]+",
            debug=True
        )
        converter_default.Execute()
        
        default_files = list(Path(output_default.name).glob("*.nii.gz"))
        self.assertGreater(len(default_files), 0, "No default files generated")
        
        # Test with custom filename format
        output_custom = tempfile.TemporaryDirectory()
        converter_custom = Dcm2NiiConverter(
            str(sample1_path),
            out_dir=output_custom.name,
            idglobber="sample[0-9]+",
            custom_filename_format="`0008|103e`-`0010|0020`-`0008|0020`",
            debug=True
        )
        converter_custom.Execute()
        
        custom_files = list(Path(output_custom.name).glob("*.nii.gz"))
        self.assertGreater(len(custom_files), 0, "No custom files generated")
        
        # Compare filenames - they should be different
        default_name = default_files[0].stem.replace('.nii', '')
        custom_name = custom_files[0].stem.replace('.nii', '')
        
        print(f"Default filename: {default_name}")
        print(f"Custom filename: {custom_name}")
        
        # The custom filename should contain the custom format elements
        self.assertIn("ARTERIELLE", custom_name, "Custom format not applied - missing series description")
        self.assertIn("20061012", custom_name, "Custom format not applied - missing study date")
        
        # They should be different
        self.assertNotEqual(default_name, custom_name, "Default and custom filenames should be different")
        
        # Clean up
        output_default.cleanup()
        output_custom.cleanup()

    def test_dcm2nii_custom_format_with_missing_tag(self):
        """Test custom format with a missing DICOM tag."""
        sample1_path = self.dcm_dir / "sample1"
        
        # Use a format with a non-existent tag
        converter = Dcm2NiiConverter(
            str(sample1_path),
            out_dir=self.output_dir,
            idglobber="sample[0-9]+",
            custom_filename_format="`0008|103e`-`9999|9999`",  # 9999|9999 doesn't exist
            debug=True
        )
        converter.Execute()
        
        output_files = list(Path(self.output_dir).glob("*.nii.gz"))
        self.assertGreater(len(output_files), 0, "No files generated with missing tag")
        
        # Check that "Missing" appears in the filename
        filename = output_files[0].stem.replace('.nii', '')
        print(f"Filename with missing tag: {filename}")
        self.assertIn("Missing", filename, "Missing tag not handled correctly")

    def test_command_line_integration(self):
        """Test the command-line interface with custom filename format."""
        from mnts.scripts.dicom2nii import console_entry
        
        # Test command line arguments
        args = [
            "-i", str(self.dcm_dir / "sample1"),
            "-o", self.output_dir,
            "--custom-filename-format", "`0008|103e`-`0008|0020`",
            "--debug",
            "--verbose"
        ]
        
        # This should work without errors
        try:
            console_entry(args)
            output_files = list(Path(self.output_dir).glob("*.nii.gz"))
            self.assertGreater(len(output_files), 0, "No files generated via command line")
            
            # Check filename contains expected elements
            filename = output_files[0].stem.replace('.nii', '')
            print(f"Command-line generated filename: {filename}")
            self.assertIn("ARTERIELLE", filename, "Series description not in filename")
            self.assertIn("20061012", filename, "Study date not in filename")
            
        except SystemExit as e:
            # console_entry might call sys.exit, which is normal
            if e.code != 0:
                self.fail(f"Command line execution failed with exit code {e.code}")

    def test_backward_compatibility_integration(self):
        """Test that the system maintains backward compatibility with existing functionality."""
        sample1_path = self.dcm_dir / "sample1"
        
        # Test that all existing functionality still works
        converter = Dcm2NiiConverter(
            str(sample1_path),
            out_dir=self.output_dir,
            idglobber="sample[0-9]+",
            check_im_type=True,
            use_patient_id=False,
            dump_meta_data=True,
            debug=True
        )
        converter.Execute()
        
        # Check that files are generated
        nii_files = list(Path(self.output_dir).glob("*.nii.gz"))
        json_files = list(Path(self.output_dir).glob("*.json"))
        
        self.assertGreater(len(nii_files), 0, "No NIfTI files generated")
        self.assertGreater(len(json_files), 0, "No metadata JSON files generated")
        
        print(f"Generated {len(nii_files)} NIfTI files and {len(json_files)} JSON files")
        print(f"Sample filename: {nii_files[0].name}")

if __name__ == '__main__':
    unittest.main()