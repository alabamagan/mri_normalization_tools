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

__all__ = ['dicom2nii', 'batch_dicom2nii']

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


