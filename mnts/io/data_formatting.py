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
        use_patient_id (bool, optional):
            Flag indicating whether to use patient ID in the file name.
        use_top_level_fname (bool, optional):
            Flag indicating whether to use top level file name.
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
                 use_patient_id     : Optional[bool]                = False,
                 use_top_level_fname: Optional[bool]                = False,
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
        self.use_patient_id = use_patient_id
        self.use_top_level_fname = use_top_level_fname
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
        `idglobber` will be used instead.

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
        self.logger.debug(f"Prefix ID from path: {id_from_path}")

        # set
        return f, id_from_path

    def construct_ouputpath(self, dcm_files: Iterable[Union[str, Path]]) -> Tuple[str, dict]:
        # Generate out file name
        headerreader = sitk.ImageFileReader()
        headerreader.SetFileName(dcm_files[0])
        headerreader.LoadPrivateTagsOn()
        headerreader.ReadImageInformation()

        # Get all the tags first
        all_dicom_tags = {k: headerreader.GetMetaData(k) for k in headerreader.GetMetaDataKeys()}
        tags = {
            '0010|0020': None, # PID
            '0008|103e': None, # Study description, usually they put protocol name here
            '0020|0011': None, # Series Number
        }
        for dctag in tags:
            try:
                tags[dctag] = headerreader.GetMetaData(dctag).rstrip().rstrip(' ')
            except RuntimeError:
                self.logger.warning(f"Tag [{dctag}] missing for image {dcm_files[0]}")
                tags[dctag] = 'Missing'

        # Warn if pid and original prefix is not the same
        pid = tags['0010|0020'].rstrip(' ')
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

        # Add this to path for clarity
        description = re.sub(' +', '_', tags['0008|103e'])

        # Remove slash because some people are stupid and put it in description that will mess with path
        description = re.sub('\/+', '_', description)

        # Output path structure
        outname = self.out_dir + '/%s-%s+%s.nii.gz'%(final_prefix,
                                                description,
                                                tags['0020|0011'])  # Some series has the same series name, need this
                                                                    # to differentiate
        return outname, all_dicom_tags

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
                self.logger.info(f"Writing segmentation files: {pprint.pformat(outimage.keys())}")
                for key, seg in outimage.items():
                    # alter filename to attach segmentation
                    suffix = key.replace(' ', '-')
                    _outname = constructed_outname.replace('.nii.gz', f"_{suffix}.nii.gz")
                    seg.SetMetaData('intent_name', dcm_tags['0010|0020'].rstrip())
                    self.logger.info(f"Writing {_outname}")
                    sitk.WriteImage(seg, _outname)

            # Write metadata
            if self.dump_meta_data:
                meta_data_dir = constructed_outname.replace('.nii.gz', '.json')
                if Path(meta_data_dir).is_file():
                    self.logger.warning(f"Overwriting {str(meta_data_dir)}")
                with open(str(meta_data_dir), 'w') as jf:
                    json.dump(dcm_tags, jf)
        pass

    def read_images(self, f, ss) -> Tuple[sitk.Image, List[str]]:
        r"""Read iamges from folder `f` that is identified by SID `ss`

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
                self.logger.info(f"Detect segmentation files: {pprint.pformat(dcm_files)}")
                outimage = self.read_segmentation(dcm_files)
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
              use_patient_id: bool = False,
              use_top_level_fname: bool = False,
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
                                     out_dir,
                                     seq_filters,
                                     idglobber,
                                     use_patient_id,
                                     use_top_level_fname,
                                     root_dir,
                                     idlist,
                                     prefix,
                                     debug,
                                     dump_meta_data)
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

    .. notes::
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

