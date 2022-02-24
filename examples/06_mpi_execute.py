import os
from pathlib import Path
from mnts.filters.geom import *
from mnts.filters.intensity import *
from mnts.filters.mnts_filters_graph import MNTSFilterGraph
from functools import partial
import SimpleITK as sitk

from tqdm.auto import tqdm
from mnts.utils import repeat_zip
from mnts.filters import mpi_wrapper, TypeCastNode

# Define paths
image_folder = Path(r'/home/lwong/Storage/Source/Repos/NPC_Segmentation/NPC_Segmentation/60.Large-Study/v1-All-Data/./Original/T2WFS_TRA')
segment_folder = Path(r'/home/lwong/Storage/Source/Repos/NPC_Segmentation/NPC_Segmentation/60.Large-Study/v1-All-Data/../../0B.Segmentations/T2WFS_TRA/00.First')
out_folder = Path(r'/home/lwong/Storage/Source/Repos/NPC_Segmentation/NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_Seg/T2WFS_TRA')
out_temp_folder = out_folder.joinpath('.training_outputs')
state_dir = out_temp_folder.joinpath('Trained_states')

"""
To allow graph to be pickled, it need a wrapper function
"""
def create_graph() -> MNTSFilterGraph:
    r"""Create the normalization graph"""
    G = MNTSFilterGraph()

    # Add filter nodes to the graph.
    G.add_node(SpatialNorm(out_spacing=[0.4492, 0.4492, 4]))
    G.add_node(HuangThresholding(closing_kernel_size=10), 0, is_exit=True)  # Use mask to better match the histograms
    G.add_node(N4ITKBiasFieldCorrection(), [0, 1])
    G.add_node(NyulNormalizer(), [2, 1], is_exit=True)
    return G

"""
If training is need, not necessary to wrap it around a function though
"""
def train_nyul():
    # Create training outputs folders
    out_temp_folder.mkdir(exist_ok=True, parents=True)

    # Define inputs for preparing training materials
    images = [f for f in image_folder.iterdir() if f.name.find('nii') != -1]
    out_names = [i.name for i in images]
    z = ([3], out_names, [out_temp_folder], images)

    # Create the graph
    G = create_graph()
    #result = mpi_wrapper(G.prepare_training_files, z)

    G.train_node(out_temp_folder.joinpath('Trained_states'), out_temp_folder, 3)

"""
'main'
"""
def normalize_images(folder):
    G = create_graph()
    G.load_node_states(3, state_dir)


    images = [f for f in folder.iterdir() if f.name.find('nii') != -1]
    segs   = [f for f in segment_folder.iterdir() if f.name.find('nii') != -1]
    output_prefix = [im.name for im in segs]
    output_directories = (out_folder.joinpath('00.HuangMask'),
                          out_folder.joinpath('01.NyulNormalized'),
                          )

    # Prepare
    # z = [output_prefix, [output_directories], segs]
    seg_z = [output_prefix, [out_folder], segs]

    """
    Single-thread
    """
    for row in repeat_zip(*seg_z):
        G.mpi_execute(*row, force_request=0)

    """
    Multi-thread
    """
    mpi_wrapper(partial(G.mpi_execute, force_request=0), seg_z, num_worker=10)
    return 0

# If this protector is absent, windows python might go into recursive import loop.
def main():
    train_nyul()
    normalize_images(image_folder)

if __name__ == '__main__':
    main()
