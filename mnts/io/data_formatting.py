import SimpleITK as sitk
import os
import numpy as np
import re
import multiprocessing as mpi
import random
import json
import pprint
sitk.ProcessObject_GlobalWarningDisplayOff()

from functools import partial
from pathlib import Path
from mnts.utils.preprocessing import recursive_list_dir
from mnts.io.dixon import DIXON_dcm_to_images
from typing import Optional, Union, List, Tuple, Iterable, Dict
from ..mnts_logger import MNTSLogger
from tqdm.auto import tqdm

# These are optional packages installed by `pip install -e .[dicom]`
try:
    import pydicom
    import pydicom_seg
    PYDICOM_SEG_AVAILABLE = True
except:
    PYDICOM_SEG_AVAILABLE = False


__all__ = ['dicom2nii', 'batch_dicom2nii', 'pydicom_read_series']


class Dcm2NiiConverter:
    r"""A class for converting DICOM images to NIfTI format and optionally dumping metadata.

    This class provides functionality to read DICOM images from a folder, convert them to NIfTI format,
    and write them to an output directory. It also provides options to filter the images based on DICOM tags,
    and to dump the DICOM metadata to a JSON file.

    Args:
        folder (str):
            The folder where the DICOM images are located.
        out_dir (str, optional):
            The output directory where the converted NIfTI images will be saved.
        seq_filters (list or str, optional):
            Sequence filters for skipping DICOM series based on DICOM tag (0008|103e).
        idglobber (str, optional):
            Regex for extracting ID from the path.
        check_im_type (bool, optional):
            If true, check DICOM tag (0008,0008) of the series and generate indiivdual image
            for unique (0008,0008). Useful when handling DIXON. Default to `False`.
        use_patient_id (bool, optional):
            Flag indicating whether to use patient ID in the file name.
        use_top_level_fname (bool, optional):
            Flag indicating whether to use top level file name.
        add_scan_time (bool, optional):
            Flag indicating whether scan-time should be prefixed.
        root_dir (str or Path):
            Root path that is referenced if `use_top_level_fname` is specified.
        idlist (list or tuple, optional):
            List of IDs to process. If not None, only IDs in this list will be processed.
        prefix (str, optional):
            Prefix to add to the output file name.
        debug (bool, optional):
            Flag indicating whether to run in debug mode.
        dump_meta_data (bool, optional):
            Flag indicating whether to dump DICOM metadata to a JSON file.

    Attributes:
        workerid (str):
            The ID of the current worker process.
        logger (MNTSLogger):
            Logger for logging information and errors.

    Examples:
    >>> from mnts.io.data_formatting import Dcm2NiiConverter
    >>> folder = './example_folder' # DICOM images in this folder should all belong to the same scan
    >>> out_dir = "./example_out_folder" # Where the Nii files will be placed
    >>> converter = Dcm2NiiConverter(folder, out_dir)
    >>> converter.Execute()


    """
    def __init__(self,
                 folder             : str,
                 out_dir            : str                           = None,
                 seq_filters        : Optional[Union[list, str]]    = None,
                 idglobber          : Optional[str]                 = None,
                 check_im_type      : Optional[bool]                = False,
                 use_patient_id     : Optional[bool]                = False,
                 use_top_level_fname: Optional[bool]                = False,
                 add_scan_time      : Optional[bool]                = False,
                 root_dir           : Union[str, Path]              = None,
                 idlist             : Optional[Union[List, Tuple]]  = None,
                 prefix             : Optional[str]                 = "",
                 debug              : Optional[bool]                = False,
                 dump_meta_data     : Optional[bool]                = False) -> None:
        # Initialize class
        self.folder = folder
        self.out_dir = out_dir
        self.seq_filters = seq_filters
        self.idglobber = idglobber
        self.check_im_type = check_im_type
        self.use_patient_id = use_patient_id
        self.use_top_level_fname = use_top_level_fname
        self.add_scan_time = add_scan_time
        self.root_dir = root_dir
        self.idlist = idlist
        self.prefix = prefix
        self.debug = debug
        self.dump_meta_data = dump_meta_data

        # Worker ID and logger setup
        self.workerid = mpi.current_process().name
        self.logger = MNTSLogger['utils.dicom2nii-%s'%self.workerid]
        self.logger.info(f"Handling: {self.folder}")

        # initialize id globber
        if self.idglobber is None:
            self.idglobber = "(?i)(NPC|P|RHO|T1rhoNPC)?[0-9]{3,5}"

        # standardize output dir
        self.out_dir = str(Path(self.out_dir).resolve().absolute())

    def __call__(self):
        try:
            self.create_directory()
            self.process_folder()
        except Exception as e:
            self.logger.exception(e)
            return 1

    def Execute(self):
        self.__call__()

    def create_directory(self):
        r"""Makes a directory if it does not exist.

        Args:
            path (str):
                The path of the directory to be created.
            logger (logging.Logger):
                The logger to use for logging messages.
        """
        if not os.path.isdir(self.out_dir):
            self.logger.info(f"Trying to make folder at {str(self.out_dir)}")
            os.makedirs(self.out_dir, exist_ok=True)
            assert os.path.isdir(self.out_dir), "Output dir was not made."

    def pre_processing_paths(self) -> Path:
        r"""Preprocess and standardize input path. If option `use_top_level_fname` is specified, this method will also
        update the prefix to the top level folder as identifier of this data. Otherwise, the ID globbed by the specified
        `idglobber` will be used instead. If `use_patient_id` is specified, it will read the patient ID from the first
        DICOM file in the folder.

        Returns:
            f (str):
                The input path
            prefix1 (str):
                The tentative prefix of the input. This might change later.
        """
        # Construct paths
        folder = str(Path(self.folder).resolve().absolute())
        self.logger.debug(f"Processing folder: {folder}")
        f = folder

        # Search ID from path using `idglobber`
        matchobj = re.search(self.idglobber, os.path.basename(f))

        if not matchobj is None:
            id_from_path = matchobj.group()
        else:
            id_from_path = "NA"

        # Handle the case when each patient has subfolders for each series
        if self.use_top_level_fname:
            path = str(folder.replace(str(Path(self.root_dir).absolute()), '/')).lstrip(os.sep) # lstrip to make sure its not starting from /
            self.logger.debug(f"Updated folder path: {folder.replace(str(Path(self.root_dir).absolute()), '/')}")
            self.logger.debug(f"Path components: {path.split(os.sep)}")
            id_from_path = path.split(os.sep)[0]
        # If use_patient_id is enabled, extract patient ID from first DICOM file
        if self.use_patient_id:
            patient_id = None
            try:
                # Find all files in the folder and try to read first valid DICOM
                for root, dirs, files in os.walk(f):
                    for file in files:
                        filepath = os.path.join(root, file)
                        try:
                            # Try to read as DICOM using pydicom
                            if PYDICOM_SEG_AVAILABLE:
                                import pydicom
                                ds = pydicom.dcmread(filepath, specific_tags=[pydicom.tag.Tag(0x0010, 0x0020)])
                                if hasattr(ds, 'PatientID'):
                                    patient_id = ds.PatientID.strip()
                                    id_from_path = patient_id
                                    self.logger.debug(f"Using patient ID from DICOM: {patient_id}")
                                    break
                        except Exception:
                            # Skip files that can't be read as DICOM
                            continue
                if patient_id is None:
                    self.logger.warning(f"No valid DICOM files found in folder, using path-based ID")
            except Exception as e:
                self.logger.warning(f"Failed to read patient ID from DICOM: {e}, using path-based ID")

        self.logger.debug(f"Prefix ID from path: {id_from_path}")

        # set
        return f, id_from_path

    def construct_ouputpath(self, dcm_files: Iterable[Union[str, Path]]) -> Tuple[str, dict]:
        """Construct the output path for the converted NIfTI file and extract DICOM metadata.

        This method reads DICOM metadata from the first file in the series to construct
    an appropriate output filename based on standardized DICOM tags. The output filename
        follows the pattern: {prefix}-{normalized_sequence_name}+{series_number}.nii.gz

        Args:
            dcm_files (Iterable[Union[str, Path]]):
                List of DICOM file paths in the series. Only the first file is used
                for metadata extraction.

        Returns:
            Tuple[str, dict]:
                A tuple containing:
                - outname (str): The constructed output file path for the NIfTI file
                - all_dicom_tags (dict): Dictionary containing all DICOM metadata tags
                  from the first file in the series

        Raises:
            RuntimeError: If required DICOM tags are missing from the file.
            ValueError: If modality is not MR.

        Note:
            - The method extracts key DICOM tags and constructs a normalized sequence name
            - The normalized name includes: modality, sequence type, scan plan, contrast info
            - If a tag is missing, it will be marked as 'Missing' and a warning will be logged
            - Patient ID from DICOM is compared with ID extracted from folder path,
              and warnings are logged if they don't match
        """
        # Generate out file name
        headerreader = sitk.ImageFileReader()
        headerreader.SetFileName(dcm_files[0])
        headerreader.LoadPrivateTagsOn()
        headerreader.ReadImageInformation()

        # Get all the tags first
        all_dicom_tags = {k: headerreader.GetMetaData(k) for k in headerreader.GetMetaDataKeys()}

        # Define required tags for sequence name construction
        required_tags = {
            '0010|0020': None,  # Patient ID
            '0008|0060': None,  # Modality
            '0018|0020': None,  # Scanning Sequence
            '0018|0021': None,  # Sequence Variant
            '0018|0022': None,  # Scan Options
            '0018|0023': None,  # MR Acquisition Type
            '0008|103e': None,  # Series Description
            '0020|0011': None,  # Series Number
            '0008|0020': None,  # Study Date
            '0020|0037': None,  # Image Orientation Patient
            '0018|1030': None,  # Protocol Name
            '0018|0015': None,  # Body Part Examined
            '0018|0080': None,  # Repetition Time (TR)
            '0018|0081': None,  # Echo Time (TE)
            '0018|0082': None,  # Inversion Time (TI)
            '0018|0091': None,  # Echo Train Length
            '0018|0050': None,  # Slice Thickness
            '0018|1250': None,  # Receive Coil Name
            '0018|1312': None,  # In-plane Phase Encoding Direction
        }
        for dctag in required_tags:
            try:
                required_tags[dctag] = headerreader.GetMetaData(dctag).strip()
            except RuntimeError:
                self.logger.warning(f"Tag [{dctag}] missing for image {dcm_files[0]}")
                required_tags[dctag] = 'Missing'

        # Validate modality
        modality = required_tags['0008|0060']
        if modality != 'MR':
            raise ValueError(f"Expected MR modality, got: {modality}")

        # Warn if pid and original prefix is not the same
        pid = required_tags['0010|0020']
        if pid != self.id_from_path:
            if not self.id_from_path == 'NA':
                self.logger.warning(f"Prefix and patient ID are not the same! (folder: {self.id_from_path}, PID: {pid})")

        # Replace prefix if use patient id for file id
        if self.use_patient_id:
            self.logger.info(f"Replacing prefix with DICOM tag PID: {pid}")
            final_prefix = pid
        else:
            # otherwise, use ID globbed from path
            final_prefix = self.id_from_path

        # Add prefix
        if len(self.prefix) > 0:
            final_prefix = self.prefix + final_prefix
        if self.add_scan_time:
           final_prefix = final_prefix + '_' + required_tags['0008|0020']

        # Construct normalized sequence name
        normalized_name = self._construct_normalized_sequence_name(required_tags)
        # Output path structure
        outname = self.out_dir + '/%s-%s+%s.nii.gz'%(final_prefix,
                                                normalized_name,
                                                required_tags['0020|0011'])
        return outname, all_dicom_tags

    def _construct_normalized_sequence_name(self, tags: dict) -> str:
        """Construct a normalized sequence name from DICOM tags.

        Args:
            tags (dict): Dictionary of DICOM tags

        Returns:
            str: Normalized sequence name in format: MR_{sequence_type}_{plane}_{contrast}_{options}_{technique}
        """
        components = ['MR']

        # Extract sequence type (T1/T2/PD/FLAIR/etc)
        sequence_type = self._extract_sequence_type(tags)
        if sequence_type:
            components.append(sequence_type)

        # Extract scan plane (TRA/COR/SAG)
        scan_plane = self._extract_scan_plane(tags)
        if scan_plane:
            components.append(scan_plane)

        # Extract contrast information (+C for gadolinium)
        contrast_info = self._extract_contrast_info(tags)
        if contrast_info:
            components.append(contrast_info)

        # Extract other options (FS for fat suppression, etc)
        scan_options = self._extract_scan_options(tags)
        if scan_options:
            components.append(scan_options)

        # Extract technique name
        technique = self._extract_technique_name(tags)
        if technique:
            components.append(technique)

        return '_'.join(components)

    def _extract_sequence_type(self, tags: dict) -> str:
        """Extract sequence type from DICOM tags."""
        scanning_sequence = tags.get('0018|0020', '').upper()
        sequence_variant = tags.get('0018|0021', '').upper()
        tr = tags.get('0018|0080', '')
        te = tags.get('0018|0081', '')
        ti = tags.get('0018|0082', '')

        # Convert timing parameters to float for comparison
        try:
            tr_val = float(tr) if tr != 'Missing' and tr else 0
            te_val = float(te) if te != 'Missing' and te else 0
            ti_val = float(ti) if ti != 'Missing' and ti else 0
        except (ValueError, TypeError):
            tr_val = te_val = ti_val = 0

        # Determine sequence type based on scanning sequence and timing
        if 'GR' in scanning_sequence:  # Gradient Echo
            if 'SK' in sequence_variant or 'SP' in sequence_variant:  # Spoiled
                if tr_val < 50:  # Fast spoiled gradient echo
                    return 'T1_SPGR'
                else:
                    return 'T1_GRE'
            elif 'SS' in sequence_variant:  # Steady State
                return 'T2_SSFP'
            elif ti_val > 0:  # Has inversion time
                return 'T1_IR'
            else:
                return 'T1_GRE'
        elif 'SE' in scanning_sequence:  # Spin Echo
            if ti_val > 0:  # FLAIR or other inversion recovery
                if te_val > 80:  # Long TE suggests FLAIR
                    return 'FLAIR'
                else:
                    return 'T1_IR'
            elif te_val > 80:  # Long TE suggests T2
                return 'T2_SE'
            elif te_val < 30:  # Short TE suggests T1
                return 'T1_SE'
            else:
                return 'PD_SE'  # Proton density
        elif 'EP' in scanning_sequence:  # Echo Planar
            if 'DWI' in tags.get('0008|103e', '').upper():
                return 'DWI_EPI'
            elif ti_val > 0:
                return 'T1_EPI'
            else:
                return 'T2_EPI'

        # Fallback based on series description
        series_desc = tags.get('0008|103e', '').upper()
        if 'T1' in series_desc:
            return 'T1'
        elif 'T2' in series_desc:
            return 'T2'
        elif 'FLAIR' in series_desc:
            return 'FLAIR'
        elif 'DWI' in series_desc or 'DIFFUSION' in series_desc:
            return 'DWI'
        elif 'SWI' in series_desc:
            return 'SWI'
        elif 'TOF' in series_desc:
            return 'TOF'
        elif 'PERFUSION' in series_desc or 'PWI' in series_desc:
            return 'PWI'

        return 'UNK'  # Unknown sequence type

    def _extract_scan_plane(self, tags: dict) -> str:
        """Extract scan plane from DICOM tags."""
        # Check Image Orientation Patient tag
        orientation = tags.get('0020|0037', '')
        if orientation and orientation != 'Missing':
            try:
                # Parse the 6 values from Image Orientation Patient
                values = [float(x) for x in orientation.split('\\')]
                if len(values) == 6:
                    # First three values are row direction cosines
                    # Last three values are column direction cosines
                    row_x, row_y, row_z = values[0:3]
                    col_x, col_y, col_z = values[3:6]

                    # Determine primary plane based on normal vector
                    # Normal vector is cross product of row and column vectors
                    normal_x = row_y * col_z - row_z * col_y
                    normal_y = row_z * col_x - row_x * col_z
                    normal_z = row_x * col_y - row_y * col_x

                    # Find the component with largest absolute value
                    abs_x, abs_y, abs_z = abs(normal_x), abs(normal_y), abs(normal_z)

                    if abs_z > abs_x and abs_z > abs_y:
                        return 'TRA'  # Transverse/Axial
                    elif abs_y > abs_x and abs_y > abs_z:
                        return 'COR'  # Coronal
                    elif abs_x > abs_y and abs_x > abs_z:
                        return 'SAG'  # Sagittal
            except (ValueError, IndexError):
                pass

        # Fallback to series description
        series_desc = tags.get('0008|103e', '').upper()
        protocol_name = tags.get('0018|1030', '').upper()

        desc_combined = f"{series_desc} {protocol_name}"

        if any(term in desc_combined for term in ['AX', 'AXIAL', 'TRA', 'TRANS']):
            return 'TRA'
        elif any(term in desc_combined for term in ['COR', 'CORONAL']):
            return 'COR'
        elif any(term in desc_combined for term in ['SAG', 'SAGITTAL']):
            return 'SAG'

        return ''  # Unknown plane

    def _extract_contrast_info(self, tags: dict) -> str:
        """Extract contrast information from DICOM tags."""
        contrast_info = []

        series_desc = tags.get('0008|103e', '').upper()
        protocol_name = tags.get('0018|1030', '').upper()
        scan_options = tags.get('0018|0022', '').upper()

        desc_combined = f"{series_desc} {protocol_name} {scan_options}"

        # Check for gadolinium contrast
        if any(term in desc_combined for term in ['+C', 'GAD', 'GADOLINIUM', 'CONTRAST', 'POST']):
            contrast_info.append('C')

        return '+'.join(contrast_info) if contrast_info else ''

    def _extract_scan_options(self, tags: dict) -> str:
        """Extract scan options from DICOM tags."""
        options = []

        series_desc = tags.get('0008|103e', '').upper()
        protocol_name = tags.get('0018|1030', '').upper()
        scan_options = tags.get('0018|0022', '').upper()
        sequence_variant = tags.get('0018|0021', '').upper()

        desc_combined = f"{series_desc} {protocol_name} {scan_options} {sequence_variant}"

        # Fat suppression
        if any(term in desc_combined for term in ['FS', 'FAT_SAT', 'FATSAT', 'SPAIR', 'STIR']):
            options.append('FS')

        # Flow compensation
        if any(term in desc_combined for term in ['FC', 'FLOW_COMP', 'FLOWCOMP']):
            options.append('FC')

        # Magnetization transfer
        if any(term in desc_combined for term in ['MT', 'MAG_TRANS', 'MAGTRANS']):
            options.append('MT')

        # Parallel imaging
        if any(term in desc_combined for term in ['SENSE', 'GRAPPA', 'ASSET', 'IPAT']):
            options.append('PI')

        # Diffusion weighting
        if any(term in desc_combined for term in ['DWI', 'DIFFUSION', 'ADC']):
            options.append('DW')

        return '_'.join(options) if options else ''

    def _extract_technique_name(self, tags: dict) -> str:
        """Extract technique name from DICOM tags."""
        series_desc = tags.get('0008|103e', '').upper()
        protocol_name = tags.get('0018|1030', '').upper()

        desc_combined = f"{series_desc} {protocol_name}"

        # Common MR techniques
        techniques = {
            'SWI': ['SWI', 'SUSCEPTIBILITY'],
            'TOF': ['TOF', 'TIME_OF_FLIGHT'],
            'PWI': ['PWI', 'PERFUSION', 'DSC', 'ASL'],
            'MRA': ['MRA', 'ANGIO'],
            'MRV': ['MRV', 'VENOGRAPHY'],
            'DTI': ['DTI', 'TENSOR'],
            'BOLD': ['BOLD', 'FMRI'],
            'CSF': ['CSF', 'CISS', 'FIESTA'],
            'MRCP': ['MRCP', 'CHOLANGIO'],
            'SPACE': ['SPACE', 'CUBE', 'VISTA'],
            'MPR': ['MPR', 'MULTIPLANAR'],
            'VIBE': ['VIBE', 'LAVA', 'THRIVE']
        }

        for technique, keywords in techniques.items():
            if any(keyword in desc_combined for keyword in keywords):
                return technique

        # Check for specific sequence names in protocol
        if 'MPRAGE' in desc_combined:
            return 'MPRAGE'
        elif 'FSPGR' in desc_combined:
            return 'FSPGR'
        elif 'FLASH' in desc_combined:
            return 'FLASH'
        elif 'TRUFI' in desc_combined:
            return 'TRUFI'
        elif 'HASTE' in desc_combined:
            return 'HASTE'
        elif 'RARE' in desc_combined or 'TSE' in desc_combined:
            return 'TSE'
        elif 'EPI' in desc_combined:
            return 'EPI'

        return ''  # No specific technique identified

    def process_folder(self):
        r"""Process a folder to read images, filter based on DICOM tags, write the output image, and dump metadata.

        This method conducts several steps:
        1. Pre-processes paths and checks if the ID from the path is in a predefined ID list.
        2. Skips the folder if the ID is not in the list.
        3. Reads image series from the folder.
        4. For each series, it reads images, constructs the output path, and extracts DICOM tags.
        5. If a sequence filter is set, it skips the series if the DICOM tag (0008|103e) does NOT match the filter.
        6. Writes the read image to the constructed output path.
        7. If the dump_meta_data flag is set, it dumps metadata to a JSON file.

        This method uses SimpleITK for image reading and writing, and regex for sequence filtering.

        .. warning::
            If a file with the same name as the output JSON file already exists, it will be overwritten without warning.

        Raises:
            ImportError: If required libraries are not found.
            ValueError: If input arguments are in wrong format or invalid.
            FileNotFoundError: If the input folder or files do not exist.
        """
        f, self.id_from_path = self.pre_processing_paths()

        # Check ID list and drop if input is not in idlist (not None). This is based on result of idglobber
        if not self.idlist is None:
            if not self.id_from_path in self.idlist:
                self.logger.info(f"Skipping {self.id_from_path} because target not in idlist.")
                return

        self.logger.debug(f"Reading from: {f}")
        series = sitk.ImageSeriesReader_GetGDCMSeriesIDs(f)

        if not len(series):
            self.logger.warning(f"No series was found in {f}")

        for ss in series:
            # Read file
            outimage, dcm_files = self.read_images(f, ss)
            constructed_outname, dcm_tags = self.construct_ouputpath(dcm_files)

            # Skip if dicom tag (0008|103e) contains substring in seq_filters
            seq_filters = self.seq_filters
            if not seq_filters is None:
                if isinstance(seq_filters, list):
                    regex = "("
                    for i, fil in enumerate(seq_filters):
                        regex += '(.*' + fil + '{1}.*)'
                        if i != len(seq_filters) - 1:
                            regex += '|'
                    regex += ')'
                    if re.match(regex, tags['0008|103e']) is None:
                        self.logger.warning("skipping ", tags['0008|103e'], "from ", f)
                        continue
                elif isinstance(seq_filters, str):
                    if re.match(seq_filters, tags['0008|103e']) is None:
                        self.logger.warning("skipping ", tags['0008|103e'], "from ", f)
                        continue

            # Write image
            if isinstance(outimage, sitk.Image):
                self.logger.info(f"Writting: {constructed_outname}")
                outimage.SetMetaData('intent_name', dcm_tags['0010|0020'].rstrip())
                sitk.WriteImage(outimage, constructed_outname)
            elif isinstance(outimage, dict):
                self.logger.info(f"Writing multiple files: {pprint.pformat(outimage.keys())}")
                for key, img in outimage.items():
                    # alter filename to attach segmentation
                    if not constructed_outname.find("DIXON") >=0:
                        suffix = key.replace(' ', '-')
                        _outname = constructed_outname.replace('.nii.gz', f"_{suffix}.nii.gz")
                        img.SetMetaData('intent_name', dcm_tags['0010|0020'].rstrip())
                    else:
                        #! Somewhat hardcode to deal with DIXON
                        _outname = constructed_outname.replace('DIXON', 'DIXON-' + key)
                    self.logger.info(f"Writing {_outname}")
                    sitk.WriteImage(img, _outname)

            # Write metadata
            if self.dump_meta_data:
                meta_data_dir = constructed_outname.replace('.nii.gz', '.json')
                if Path(meta_data_dir).is_file():
                    self.logger.warning(f"Overwriting {str(meta_data_dir)}")
                with open(str(meta_data_dir), 'w') as jf:
                    json.dump(dcm_tags, jf)
        pass

    def read_images(self, f, ss) -> Tuple[sitk.Image, List[str]]:
        r"""Read images from folder `f` that is identified by SID `ss`

        Args:
            f (str):
                The absolute path to the folder.
            ss (str):
                The SID of the series read from DICOM.

        Returns:
            outimage (sitk.Image):
                The output image.
            reader (sitk.ImageSeriesReader):
                The reader object that is further used to read the header information.
        """
        dcm_files = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(
            f,
            ss
        )

        # Check if the files are segmentation
        if PYDICOM_SEG_AVAILABLE:
            ds = pydicom.dcmread(dcm_files[0])
            # Check if the file contains segmentation data
            if ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4':
                # warn if check im_type is also on
                if self.check_im_type:
                    self.logger.warning("Check image type option is enabled, but segmented.")

                self.logger.info(f"Detect segmentation files: {pprint.pformat(dcm_files)}")
                outimage = self.read_segmentation(dcm_files)
                return outimage, dcm_files

        if self.check_im_type:
            # perform image type check, this is mainly written for DIXON
            self.logger.info("Proceed with image type check...")
            _outimage = DIXON_dcm_to_images(dcm_files)
            if len(_outimage) > 1:
                # rewrite the dictionary keys
                outimage = {}
                for k, im in _outimage.items():
                    if len(k) < 4:
                        outimage['-'.join(k)] = im
                    if k[3] == "W":
                        # DIXON water image (fat-suppressed)
                        outimage['FS'] = im
                    elif k[3] == "IP":
                        # DIXON regular image
                        outimage['IP'] = im
                    else:
                        outimage['-'.join(k)] = im
            else:
                # if there's only one image type, just do things regularly
                outimage = list(_outimage.values())[0]
            return outimage, dcm_files


        # read image
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dcm_files)
        outimage = reader.Execute()
        return outimage, dcm_files

    def read_segmentation(self, filename: List[Union[str, Path]]):
        """Read a DICOM file with embedded segmentation. This function relies on the package `pydicom` and `pydicom-seg`
         so please make sure they are properly installed before calling.

        Args:
            filename (list):
                The path to the DICOM file. This should be a list with a single element

        Returns:
            numpy.array: The segmentation as a numpy array.
        """
        # Read the DICOM file
        if not PYDICOM_SEG_AVAILABLE:
            msg = "Segmentation is detected but loading them require `pydicom` and `pydicom-seg` packages " \
                  "that is not installed. Reinstall this package by `pip install --force-reinstall mri-nor" \
                  "malization-tools[dicom]`"
            self.logger.warning(msg)
            return

        if not len(filename) == 1:
            msg = "There is more than one file in the segmentation series."
            raise ValueError(msg)

        ds = pydicom.dcmread(filename[0])
        reader = pydicom_seg.SegmentReader()

        # Get the segmentation from the pixel data
        seg = reader.read(ds)

        # Identify the tag names
        segment_labels = {}
        for val in seg.available_segments:
            try:
                # Tag is defined by DICOM standard to be "Segment Description"
                label_name = seg.segment_infos[val][0x0062, 0x0006].value
            except:
                label_name = 'Unknown'
            segment_labels[val] = label_name

        out_dict = {}
        for val in segment_labels:
            out_dict[segment_labels[val]] = seg.segment_image(val)

        self.logger.info(f"Segmentation found: {segment_labels}")
        return out_dict


def dicom2nii(folder: str,
              out_dir: str = None,
              seq_filters: list or str = None,
              idglobber: str = None,
              check_im_type = False,
              use_patient_id: bool = False,
              use_top_level_fname: bool = False,
              add_scan_time: bool = False,
              root_dir = None,
              idlist = None,
              prefix = "",
              debug = False,
              dump_meta_data = False) -> None:
    """Covert a series under specified folder into an nii.gz image.
    This tries to assign a unique ID to each of the converted images, either based on their patient id DICOM tag or
    the folder containing the image series.

    See Also:
        :class:`Dcm2NiiConvertre`
    """

    workerid = mpi.current_process().name
    logger = MNTSLogger['utils.dicom2nii-%s'%workerid]
    logger.info(f"Handling: {folder}")
    try:
        converter = Dcm2NiiConverter(folder,
                                     out_dir=out_dir,
                                     seq_filters=seq_filters,
                                     idglobber=idglobber,
                                     check_im_type=check_im_type,
                                     use_patient_id=use_patient_id,
                                     use_top_level_fname=use_top_level_fname,
                                     add_scan_time=add_scan_time,
                                     root_dir=root_dir,
                                     idlist=idlist,
                                     prefix=prefix,
                                     debug=debug,
                                     dump_meta_data=dump_meta_data)
        converter.Execute()
    except Exception as e:
        logger.exception(e)
        return 1


def batch_dicom2nii(folderlist, out_dir,
                    workers=8,
                    **kwargs
                    ):
    r"""
    Batch version, use entry point in script instead of using this directly.

    See Also:
        func:`dicom2nii`
    """
    import multiprocessing as mpi
    logger = MNTSLogger['mpi_dicom2nii']

    if workers > 1:
        pool = mpi.Pool(workers)
        for f in folderlist:
            logger.info(f"Creating job for: {f}.")
            func = partial(dicom2nii,
                           out_dir = out_dir,
                           **kwargs
                           )
            # dicom2nii(f, out_dir, seq_filters, idglobber, use_patient_id,use_top_level_fname, input, idlist)
            # func(f)
            pool.apply_async(func, args=[f])
        pool.close()
        pool.join()
    else:
        for f in folderlist:
            dicom2nii(f, out_dir=out_dir, **kwargs)


def pydicom_read_series(dcmdir: Path, progress_bar: bool = False) -> Dict[str, List[Path]]:
    r"""Reads DICOM series from a directory and organizes them by SOP Instance UID.

    This function traverses a directory for DICOM files, reads them using pydicom,
    and returns a dictionary organizing the file paths by their SOP Instance UID.
    If `progress_bar` is enabled, it displays a progress bar during the operation.

    Args:
      dcmdir (Path):
        The directory path to search for DICOM files.
      progress_bar (bool, Optional):
        Flag to enable or disable a progress bar during processing. Defaults to False.

    Returns:
      Dict[str, List[Path]]:
        A dictionary where each key is a SOP Instance UID and the value is a list
        of file paths corresponding to that UID, sorted in ascending order.

    Raises:
      FileNotFoundError: If no DICOM files are found in the provided directory.

    .. note::
        The DICOM files are identified by attempting to read them using pydicom and
        checking for a valid SOP Instance UID. Files that cannot be read or do not
        contain a valid UID are silently ignored unless they cause the list to be empty,
        in which case a FileNotFoundError is raised.

        This function uses `pydicom.dcmread` with `specific_tags` parameter to optimize
        reading speed by only fetching the necessary tags.

    """
    dcmdir = Path(dcmdir)

    # Iterate and add all files into a list
    files = []
    for r, d, f in os.walk(dcmdir):
        if len(f) > 0:
            files.extend([Path(r) / ff for ff in f])

    # Check each file to see if they are dicom using pydicom
    dcmlist = []
    for ff in tqdm(files, disable=not progress_bar):
        try:
            dcmlist.append((ff, pydicom.dcmread(ff, specific_tags=[pydicom.tag.Tag(0x0020,0x000e)])))
        except pydicom.errors.InvalidDicomError:
            # don't add to list if can't read it
            pass

    if not len(dcmlist):
        raise FileNotFoundError(f"No DICOM files were found in directory: {dcmdir}")

    out_dict = {}
    for fpath, dcm in dcmlist:
        if not dcm.SeriesInstanceUID in out_dict:
            out_dict[dcm.SeriesInstanceUID] = []
        out_dict[dcm.SeriesInstanceUID].append(fpath)

    for keys in out_dict:
        out_dict[keys].sort()

    return out_dict

