from ..mnts_logger import MNTSLogger
from ..utils import preprocessing
from ..io import batch_dicom2nii
import argparse
import os
from pathlib import Path

__all__ = ['dicom2nii']

def dicom2nii(a, logger):
    if not Path(a.input).is_dir():
        raise FileNotFoundError(f"Cannot open {a.input}.")
    if not Path(a.output).is_dir():
        logger.warning(f"Target directory does not exist, trying to create: {a.output}")
        Path(a.output).mkdir(exist_ok=True)
        if not Path(a.output).is_dir():
            logger.error("Error making output directory.")
            return
    if a.verbose:
        MNTSLogger.set_global_log_level('debug')

    logger.info(f"Specified ID globber: {a.idglobber}")
    logger.info(f"Use patient ID: {a.usepid}")
    ids = a.idlist.split(',') if not a.idlist is None else None
    if not ids is None:
        logger.info(f"Use patient ids for filtering: {ids}")
    dicom_dirs = preprocessing.recursive_list_dir(a.depth, a.input)
    logger.info(f"Dirs:\n{dicom_dirs}")

    try:
        batch_dicom2nii(dicom_dirs,
                        out_dir = a.output,
                        workers = a.num_workers,
                        seq_filters = None,
                        idglobber = a.idglobber,
                        check_im_type = a.check_image_type_tag,
                        use_patient_id = a.usepid,
                        use_top_level_fname = a.usefname,
                        add_scan_time = a.addtime,
                        root_dir = a.input,
                        idlist = ids,
                        prefix = a.prefix,
                        debug = a.debug,
                        dump_meta_data = a.dump_dicom_tags)
    except KeyboardInterrupt as e:
        logger.warning("Keyboard interrupt detected. Exiting.")
        return


def console_entry(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, action='store', dest='input', required=True,
                        help='Input directory that contains the DICOMs.')
    parser.add_argument('-o', '--output', type=str, action='store', dest='output', required=True,
                        help='Output directory to hold the nii files.')
    parser.add_argument('-d', '--depth', type=int, action='store', default=3, dest='depth',
                        help='Depth of DICOM file search.')
    parser.add_argument('-g', '--idglobber', action='store', default=None, dest='idglobber',
                        help='Specify the globber to glob the ID from the DICOM paths.')
    parser.add_argument('-n', '--num-workers', type=int, default=None,
                        help="Specify number of workers. If not specified, use all CPU cores.")
    parser.add_argument('--check-image-type-tag', action='store_true',
                        help='If specified, check the dicom tag (0008,0008) for differences. This is implmented to '
                             'deal with DIXON scans mainly.')
    parser.add_argument('--dump-dicom-tags', action='store_true',
                        help="If this option is specified, the dicom tags will be generated to a json text file.")
    parser.add_argument('--debug', action='store_true',
                        help="Debug mode.")
    parser.add_argument('--prefix', default="", type=str,
                        help="Add a preffix to the patient's ID.")
    parser.add_argument('--idlist', action='store', dest='idlist', default=None, type=str,
                        help='Only do conversion if the globbed ID is in the list. e.g. ["ID1", "ID2", ...]')
    parser.add_argument('--use-top-level-fname', action='store_true', dest='usefname',
                        help='Use top level file name immediately after the input directory as ID.')
    parser.add_argument('--use-patient-id', action='store_true', dest='usepid',
                        help='Use patient id as file id.')
    parser.add_argument('--add-scan-time', action='store_true', dest='addtime',
                        help='Specify this to include timestamp to filename. Useful when same patient has '
                             'multiple scans with the same protocol.')
    parser.add_argument('--log', action='store_true', dest='log',
                        help='Keep log file under ./dicom2nii.log')
    parser.add_argument('--verbose', action='store_true',
                        help='Debug log messages')
    a = parser.parse_args(raw_args)

    with MNTSLogger('./dicom2nii.log', logger_name='dicom2nii', verbose=True, keep_file=a.log) as logger:
        logger.info("Recieve argumetns: {}".format(a))
        dicom2nii(a, logger)

if __name__ == '__main__':
    console_entry()