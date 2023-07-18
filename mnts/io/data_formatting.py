import SimpleITK as sitk
import os
import numpy as np
import re
import multiprocessing as mpi
import random
import json
sitk.ProcessObject_GlobalWarningDisplayOff()

from functools import partial
from pathlib import Path
from ..mnts_logger import MNTSLogger
from mnts.utils.preprocessing import recursive_list_dir
from typing import Optional, Union, List, Tuple
import pydicom

__all__ = ['dicom2nii', 'batch_dicom2nii']


class DicomConverter:
    def __init__(self,
                 folder             : str,
                 out_dir            : str                           = None,
                 seq_filters        : Optional[Union[list, str]]    = None,
                 idglobber          : Optional[str]                 = None,
                 use_patient_id     : Optional[bool]                = False,
                 use_top_level_fname: Optional[bool]                = False,
                 input              : Union[str, Path]              = None,
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
        self.input = input
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
            path = str(folder.replace(str(Path(input).absolute()), '/')).lstrip(os.sep) # lstrip to make sure its not starting from /
            self.logger.debug(f"Updated folder path: {folder.replace(str(Path(input).absolute()), '/')}")
            self.logger.debug(f"Path components: {path.split(os.sep)}")
            id_from_path = path.split(os.sep)[0]
        self.logger.debug(f"Prefix ID from path: {id_from_path}")

        # set
        return f, id_from_path

    def construct_ouputpath(self, reader: sitk.ImageSeriesReader) -> Tuple[str, dict]:
        # Generate out file name
        headerreader = sitk.ImageFileReader()
        headerreader.SetFileName(reader.GetFileNames()[0])
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
                logger.warning(f"Tag [{dctag}] missing for image {f}")
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
        f, self.id_from_path = self.pre_processing_paths()

        # Check ID list and drop if input is not in idlist (not None). This is based on result of idglobber
        if not self.idlist is None:
            if not self.id_from_path in self.idlist:
                self.logger.info(f"Skipping {self.id_from_path} because target not in idlist.")
                return

        self.logger.debug(f"Reading from: {f}")
        series = sitk.ImageSeriesReader_GetGDCMSeriesIDs(f)

        for ss in series:
            # Read file
            outimage, reader = self.read_images(f, ss)
            constructed_outname, dcm_tags = self.construct_ouputpath(reader)

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
            self.logger.info(f"Writting: {constructed_outname}")
            outimage.SetMetaData('intent_name', dcm_tags['0010|0020'].rstrip())
            sitk.WriteImage(outimage, constructed_outname)

            # Write metadata
            if self.dump_meta_data:
                meta_data_dir = constructed_outname.replace('.nii.gz', '.json')
                if Path(meta_data_dir).is_file():
                    self.logger.warning(f"Overwriting {str(meta_data_dir)}")
                with open(str(meta_data_dir), 'w') as jf:
                    json.dump(dcm_tags, jf)

            del reader
        pass

    def read_images(self, f, ss) -> Tuple[sitk.Image, sitk.ImageSeriesReader]:
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
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(
            f,
            ss
        ))
        outimage = reader.Execute()
        return outimage, reader

    def read_segmentation(self, f, ss):
        """Read a DICOM file with embedded segmentation.

        Args:
            filename (str):
                The path to the DICOM file.

        Returns:
            numpy.array: The segmentation as a numpy array.
        """
        # Read the DICOM file
        ds = pydicom.dcmread(filename)

        # Check if the file contains segmentation data
        if ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.66.4':
            raise ValueError("The file does not contain segmentation data.")

        # Get the segmentation from the pixel data
        segmentation = ds.pixel_array

        # If the DICOM file has a palette color lookup table, apply it to the segmentation
        if 'PixelPresentation' in ds and ds.PixelPresentation == 'COLOR':
            segmentation = apply_color_lut(segmentation, ds)

        return segmentation


def dicom2nii(folder: str,
              out_dir: str = None,
              seq_filters: list or str = None,
              idglobber: str = None,
              use_patient_id: bool = False,
              use_top_level_fname: bool = False,
              input = None,
              idlist = None,
              prefix = "",
              debug = False,
              dump_meta_data = False) -> None:
    """
    Covert a series under specified folder into an nii.gz image.
    This tries to assign a unique ID to each of the converted images, either based on their patient id DICOM tag or
    the folder containing the image series.
    """

    workerid = mpi.current_process().name
    logger = MNTSLogger['utils.dicom2nii-%s'%workerid]
    logger.info(f"Handling: {folder}")


    try:
        if not os.path.isdir(out_dir):
            logger.info(f"Trying to make folder at {str(out_dir)}")
            os.makedirs(out_dir, exist_ok=True)
            assert os.path.isdir(out_dir), "Output dir was not made."

        if not os.path.isdir(folder):
            logger.error(f"Cannot locate specified folder: {folder}")
            raise IOError("Cannot locate specified folder!")

        # Default globber
        if not isinstance(idglobber, str):
            idglobber = "(?i)(NPC|P|RHO|T1rhoNPC)?[0-9]{3,5}"

        # Construct paths
        folder = Path(folder)
        folder = str(folder.resolve().absolute())
        logger.debug(f"{folder}")
        f = folder
        out_dir = str(Path(out_dir).resolve().absolute())
        matchobj = re.search(idglobber, os.path.basename(f))

        if not matchobj is None:
            prefix1 = matchobj.group()
        else:
            prefix1 = "NA"

        if use_top_level_fname:
            path = str(folder.replace(str(Path(input).absolute()), '/')).lstrip(os.sep) # lstrip to make sure its not starting from /
            logger.debug(f"{folder.replace(str(Path(input).absolute()), '/')}")
            logger.debug(f"{path.split(os.sep)}")
            prefix1 = path.split(os.sep)[0]
        logger.debug(f"ID: {prefix1}")

        # Check ID list and drop if input is not in idlist (not None)
        if not idlist is None:
            if not prefix1 in idlist:
                logger.info(f"Skipping {prefix1} because target not in idlist.")
                return

        # Read file
        logger.debug(f"Reading from: {f}")
        series = sitk.ImageSeriesReader_GetGDCMSeriesIDs(f)

        for ss in series:
            logger.debug(f"{ss}")
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(
                f,
                ss
            ))
            outimage = reader.Execute()

            # Generate out file name
            headerreader = sitk.ImageFileReader()
            headerreader.SetFileName(reader.GetFileNames()[0])
            headerreader.LoadPrivateTagsOn()
            headerreader.ReadImageInformation()

            # Get all the tags first
            tags = {
                '0010|0020': None, # PID
                '0008|103e': None, # Study description, usually they put protocol name here
                '0020|0011': None, # Series Number
            }
            for dctag in tags:
                try:
                    tags[dctag] = headerreader.GetMetaData(dctag).rstrip().rstrip(' ')
                except RuntimeError:
                    logger.warning(f"Tag [{dctag}] missing for image {f}")
                    tags[dctag] = 'Missing'

            # Warn if pid and original prefix is not the same
            pid = tags['0010|0020'].rstrip(' ')
            if pid != prefix1:
                logger.warning(f"Prefix and patient ID are not the same! (folder: {prefix1}, PID: {pid})")

            # Replace prefix if use patient id for file id
            if use_patient_id:
                logger.info(f"Replacing prefix with DICOM tag PID: {pid}")
                prefix2 = pid
            else:
                prefix2 = prefix1

            # Add prefix
            if len(prefix) > 0:
                prefix2 = prefix + prefix2

            description = re.sub(' +', '_', tags['0008|103e'])
            description = re.sub('\/+', '_', description)       # Remove slash because some people are stupid and put it in description
            outname = out_dir + '/%s-%s+%s.nii.gz'%(prefix2,
                                                    description,
                                                    tags['0020|0011']) # Some series has the same series name, need this to differentiate

            # Skip if dicom tag (0008|103e) contains substring in seq_filters
            if not seq_filters is None:
                if isinstance(seq_filters, list):
                    regex = "("
                    for i, fil in enumerate(seq_filters):
                        regex += '(.*' + fil + '{1}.*)'
                        if i != len(seq_filters) - 1:
                            regex += '|'
                    regex += ')'
                    if re.match(regex, tags['0008|103e']) is None:
                        logger.warning("skipping ", tags['0008|103e'], "from ", f)
                        continue
                elif isinstance(seq_filters, str):
                    if re.match(seq_filters, tags['0008|103e']) is None:
                        logger.warning("skipping ", tags['0008|103e'], "from ", f)
                        continue

            # Write image
            logger.info(f"Writting: {outname}")
            outimage.SetMetaData('intent_name', headerreader.GetMetaData('0010|0020').rstrip())
            sitk.WriteImage(outimage, outname)

            # Write metadata
            if dump_meta_data:
                meta_data_dir = outname.replace('.nii.gz', '.json')
                all_dicom_tags = {k: headerreader.GetMetaData(k) for k in headerreader.GetMetaDataKeys()}
                if Path(meta_data_dir).is_file():
                    logger.warning(f"Overwriting {str(meta_data_dir)}")
                with open(str(meta_data_dir), 'w') as jf:
                    json.dump(all_dicom_tags, jf)

            del reader
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


