from .console_entry import MNTS_ConsoleEntry
from ..filters import MNTSFilterGraph, mpi_wrapper
from ..mnts_logger import MNTSLogger

from pathlib import Path
from typing import Union, Sequence
from mnts.utils import repeat_zip
import SimpleITK as sitk
import argparse
import tempfile

__all__ = ['_train_normalization', '_inference_normalization', 'run_graph_train', 'run_graph_inference']

def _train_normalization(G: MNTSFilterGraph,
                         input_dir: Union[Path, str],
                         output_dir: Union[Path, str],
                         num_worker=0) -> None:
    r"""
    Run training

    .. note::
        Call this function through the console entry and avoid using it directly. This function involve threading
        and could cause deadlock if called in a sub-thread.

    Args:
        G (MNTSFilterGraph):
            Must be a MNTSFilterGraph instance already created.
        input_dir (Path or str):
            The path to a directory hosting multiple NIFTI files or a path to a single NIFTI file. However, note that
            for training, one file is not enough, this support is just for legacy compatibility.
            TODO: Add support for multiple files.
        output_dir (Path or str):
            The path to hold the intermediate output. The trained states will also be placed under this directory inside
            a sub-directory `Trained_states`.
        num_worker (int, Optional):
            If > 1, the normalization is trained in parallel. Default to 0.

    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    # Check if input is one file or a whole directory
    if in_path.is_dir():
        images = [f for f in in_path.iterdir() if f.name.find('nii') != -1]
        out_names = [f.name for f in images]
    else:
        assert in_path.suffix.find('nii') >= 0, "Can only handle nii files currently."
        images = [str(in_path)]
        out_names = [in_path.name]

    # multi-processing
    z = ([None], out_names, [out_path], images)
    if num_worker > 1:
        result = mpi_wrapper(G.prepare_training_files, z, num_worker=num_worker)
    else:
        result = [G.prepare_training_files(*row) for row in repeat_zip(*z)]

    # Train and save the states.
    G.train_node(None, out_path, out_path.joinpath("Trained_states"))
    return 0

def _inference_normalization(G: MNTSFilterGraph,
                             state_dir: Union[Path, str],
                             input_dir: Union[Path, str],
                             output_dir: Sequence[Union[Path, str]],
                             num_worker=0,
                             force_segment=False) -> None:
    r"""
    Run inference of the trained network. For more see `_train_normalization`.

    Args:
        state_dir (Path or str):
            The directory holding the trained states.
        output_dir (List of path or str):
            A list of directories where the normalized images will be stored. The number of elements in this should
            match the number of exit nodes of `G`. If only one directory is passed, sub-directories will be created to
            save the output of each exit node.
        force_segment (bool, Optional):
            If specified, convert all images in the input dir into uint8 type, i.e. labels, before proceeding.

    Returns:
        0
    """
    state_path, in_path = [Path(p) for p in [state_dir, input_dir]]
    if isinstance(output_dir, (list, tuple)):
        out_path = [Path(p) for p in output_dir]
    elif isinstance(output_dir, str):
        out_path = [Path(output_dir)] # make this a list because repeat zip need list


    # Check if input is one file or a whole directory
    if in_path.is_dir():
        images = [f for f in in_path.iterdir() if f.name.find('nii') != -1]
        out_names = [f.name for f in images]
    else:
        assert in_path.suffix in (".nii", ".gz"), f"Can only handle nii files currently, got: {in_path.suffix}"
        images = [str(in_path)]
        out_names = [in_path.name]

    with tempfile.TemporaryDirectory() as f:
        if force_segment:
            for i, im in enumerate(images):
                _new_path = Path(f).joinpath(Path(im).name)
                sitk.WriteImage(sitk.Cast(sitk.ReadImage(str(im)), sitk.sitkUInt8),
                                str(_new_path))
                images[i] = str(_new_path)

        # Load the trained normalization states
        G.load_node_states(None, str(state_path))

        # Prepare arguments
        z = [out_names, [out_path], images]
        if num_worker > 1:
            mpi_wrapper(G.mpi_execute, z, num_worker=num_worker)
        else:
            for row in repeat_zip(*z):
                print(row)
            [G.mpi_execute(*row) for row in repeat_zip(*z)]
    return 0


def run_graph_train(raw_args=None):
    r"""
    Script to initiate graph training
    Args:
        raw_args:

    Returns:

    """

    parser = MNTS_ConsoleEntry('ionvl')
    parser.add_argument('-f', '--file', action='store', type=str,
                        help="Specify a yaml file for creating the normalization graph.")
    a = parser.parse_args(raw_args)

    yaml_file = Path(a.file)
    assert yaml_file.is_file(), f"Cannot open yaml file at {yaml_file}"

    G = MNTSFilterGraph.CreateGraphFromYAML(yaml_file)

    log_save = Path(a.save_log).suffix == ".log"
    log_dir = a.save_log if log_save else None
    with MNTSLogger(log_dir, keep_file=log_save, verbose=a.verbose) as logger:
        _train_normalization(G,
                             a.input,
                             a.output,
                             a.numworker)

def run_graph_inference(raw_args=None):
    parser = MNTS_ConsoleEntry('ionvl')
    parser.add_argument('-s', '--state-dir', action='store', type=str,
                        help="Directory which holds the saved states.")
    parser.add_argument('-f', '--file', action='store', type=str,
                        help="Specify a yaml file for creating the normalization graph.")
    parser.add_argument('--force-segment', action='store_true',
                        help="If specified, force the input as uint8 such that they are treated as segmentation.")
    a = parser.parse_args(raw_args)

    MNTSLogger['run_graph_inference'].info(f"Recieved arguments: {a}")

    yaml_file = Path(a.file)
    assert yaml_file.is_file(), f"Cannot open yaml file at {yaml_file}"

    log_save = Path(a.save_log).suffix == ".log"
    log_dir = a.save_log if log_save else None
    with MNTSLogger(log_dir, keep_file=False, verbose=a.verbose) as logger:
        G = MNTSFilterGraph.CreateGraphFromYAML(yaml_file)
        _inference_normalization(G,
                                 a.state_dir,
                                 a.input,
                                 [a.output],
                                 a.numworker,
                                 a.force_segment)

