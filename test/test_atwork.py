import os
from pathlib import Path
from mnts.filters.geom import *
from mnts.filters.intensity import *
from mnts.filters.mnts_filters import MNTSFilterGraph
import SimpleITK as sitk

from tqdm.auto import tqdm
from mnts.utils import repeat_zip
from mnts.filters import mpi_wrapper

# Define paths
image_folder = Path(r'./0A.NIFTI_Image')
segment_folder = Path(r'./0B.Segmentation')
out_folder = Path(r'./10.Pre-processed/')
out_temp_folder = out_folder.joinpath('.training_outputs')
state_dir = out_temp_folder.joinpath('Trained_states')


def create_graph() -> MNTSFilterGraph:
    r"""Create the normalization graph"""
    G = MNTSFilterGraph()

    # Add filter nodes to the graph.
    G.add_node(SpatialNorm(out_spacing=[0.4492, 0.4492, 4]))
    G.add_node(HuangThresholding(closing_kernel_size=10), 0, is_exit=True)  # Use mask to better match teh histograms
    G.add_node(N4ITKBiasFieldCorrection(), [0, 1])
    G.add_node(NyulNormalizer(), [2, 1])
    G.add_node(RangeRescale(0, 5000), 3, is_exit=True)
    G.add_node(SignalIntensityRebinning(num_of_bins=256), 3, is_exit=True)
    return G


def train_nyul():
    # Create training outputs folders
    out_temp_folder.mkdir(exist_ok=True, parents=True)

    # Define inputs for preparing training materials
    images = [f for f in image_folder.iterdir() if f.name.find('nii') != -1]
    out_names = [i.name for i in images]
    z = ([3], out_names, [out_temp_folder], images)

    # Create the graph
    G = create_graph()
    result = mpi_wrapper(G.prepare_training_files, z)

    G.train_node(3, out_temp_folder, out_temp_folder.joinpath('Trained_states'))


def normalize_images(folder):
    G = create_graph()
    # G.load_node_states(3, state_dir)

    # Set the directories to deposit the outputs
    float_path = out_folder.joinpath('01.Normalized')
    rebbined_path = out_folder.joinpath('02.Rebinned')
    float_path.mkdir(parents=True, exist_ok=True)
    rebbined_path.mkdir(parents=True, exist_ok=True)

    """
    Single-thread
    """
    # images = [f for f in folder.iterdir() if f.name.find('nii') != -1]
    # for im in tqdm(images):
    #     save_im = G.execute(im)
    #     float_name = out_folder.joinpath('01.Normalized').joinpath(im.name).resolve().__str__()
    #     rebinned_name = out_folder.joinpath('02.Rebinned').joinpath(im.name).resolve().__str__()
    #     print(f"Saving to {float_name} and {rebinned_name}")
    #     sitk.WriteImage(save_im[4], float_name)
    #     sitk.WriteImage(save_im[5], rebinned_name)

    """
    Multi-thread
    """
    images = [f for f in folder.iterdir() if f.name.find('nii') != -1]
    output_prefix = [im.name for im in images]
    # output_directories = (out_folder.joinpath('00.OtsuMask'),
    #                       out_folder.joinpath('01.Normalized'),
    #                       out_folder.joinpath('02.Rebinned'))

    output_directories = [out_folder.joinpath('00.OtsuMask')]

    # Prepare
    z = [output_prefix, [output_directories], images]
    mpi_wrapper(G.mpi_execute, z, num_worker=4)


# If this protector is absent, windows python might go into recursive import loop.
if __name__ == '__main__':
    train_nyul()
    normalize_images(image_folder)