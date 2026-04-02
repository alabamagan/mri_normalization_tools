from ..mnts_logger import MNTSLogger
from ..utils import preprocessing
from ..io import batch_dicom2nii
import argparse
import os
import ast
import click
from pathlib import Path

__all__ = ['dicom2nii']

def _str_to_dict(in_str:str):
    if in_str is None:
        return None
    out = ast.literal_eval(in_str)
    assert type(out) == dict, f"Output eval from {in_str} is not a dictionary"
    return out

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
    dicom_dirs = preprocessing.recursive_list_dir(a.depth, a.input)
    logger.info(f"Dirs:\n{dicom_dirs}")

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
                    regex_replace = _str_to_dict(a.regex_replace),
                    custom_filename_format=a.custom_filename_format,
                    debug = a.debug,
                    dump_meta_data = a.dump_dicom_tags)




@click.command()
@click.option('-i', '--input', 'input_dir', required=True, type=str,
              help='Input directory that contains the DICOMs.')
@click.option('-o', '--output', 'output_dir', required=True, type=str,
              help='Output directory to hold the nii files.')
@click.option('-d', '--depth', default=3, type=int,
              help='Depth of DICOM file search.')
@click.option('-g', '--idglobber', default=None, type=str,
              help='Specify the globber to glob the ID from the DICOM paths.')
@click.option('-n', '--num-workers', default=None, type=int,
              help='Specify number of workers. If not specified, use all CPU cores.')
@click.option('--check-image-type-tag', is_flag=True,
              help='If specified, check the dicom tag (0008,0008) for differences. This is implemented to deal with DIXON scans mainly.')
@click.option('--dump-dicom-tags', is_flag=True,
              help='If this option is specified, the dicom tags will be generated to a json text file.')
@click.option('--debug', is_flag=True,
              help='Debug mode.')
@click.option('--prefix', default='', type=str,
              help="Add a prefix to the patient's ID.")
@click.option('--idlist', default=None, type=str,
              help='Only do conversion if the globbed ID is in the list. e.g. ["ID1", "ID2", ...]')
@click.option('--use-top-level-fname', 'usefname', is_flag=True,
              help='Use top level file name immediately after the input directory as ID.')
@click.option('--use-patient-id', 'usepid', is_flag=True,
              help='Use patient id as file id.')
@click.option('--add-scan-time', 'addtime', is_flag=True,
              help='Specify this to include timestamp to filename. Useful when same patient has multiple scans with the same protocol.')
@click.option('--regex-replace', type=str, default=None,
              help='Dictionary where key are regex matcher and key is replace value. Default to None')
@click.option('--custom-filename-format', type=str, default=None,
              help='Custom format for building name of output file by dicom tags. Example: "`0008|103e`-`1030|2003`" '
                   'creates "PROTOCOL_NAME-TAG_VALUE" format.')
@click.option('--log', is_flag=True,
              help='Keep log file under ./dicom2nii.log')
@click.option('--verbose', is_flag=True,
              help='Debug log messages')
def console_entry(input_dir, output_dir, depth, idglobber, num_workers,
                  check_image_type_tag, dump_dicom_tags, debug, prefix,
                  idlist, usefname, usepid, addtime, regex_replace,custom_filename_format, log, verbose):
    """Convert DICOM files to NIfTI format."""
    # Create a namespace object similar to argparse.Namespace for compatibility
    from types import SimpleNamespace
    a = SimpleNamespace(
        input=input_dir,
        output=output_dir,
        depth=depth,
        idglobber=idglobber,
        num_workers=num_workers,
        check_image_type_tag=check_image_type_tag,
        dump_dicom_tags=dump_dicom_tags,
        debug=debug,
        prefix=prefix,
        idlist=idlist,
        usefname=usefname,
        usepid=usepid,
        addtime=addtime,
        regex_replace=regex_replace,
        custom_filename_format=custom_filename_format,
        log=log,
        verbose=verbose
    )

    with MNTSLogger('./dicom2nii.log', logger_name='dicom2nii', verbose=True, keep_file=log) as logger:
        logger.info("Receive arguments: {}".format(a))
        dicom2nii(a, logger)

if __name__ == '__main__':
    console_entry()