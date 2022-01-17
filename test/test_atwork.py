import os
from pathlib import Path
from mnts.filters.geom import *
from mnts.filters.intensity import *
from mnts.filters.mnts_filters import MNTSFilterGraph
import SimpleITK as sitk

from tqdm.auto import tqdm
from mnts.utils import repeat_zip
from mnts.filters import mpi_wrapper, TypeCastNode

# Define paths
image_folder = Path(r'./0A.NIFTI_Image')
segment_folder = Path(r'./0B.Segmentation')
out_folder = Path(r'./10.Pre-processed-v2/')
out_temp_folder = out_folder.joinpath('.training_outputs')
state_dir = out_temp_folder.joinpath('Trained_states')


def create_graph() -> MNTSFilterGraph:
    r"""Create the normalization graph"""
    G = MNTSFilterGraph()

    # Add filter nodes to the graph.
    G.add_node(SpatialNorm(out_spacing=[0.4492, 0.4492, 4]))
    G.add_node(HuangThresholding(closing_kernel_size=10), 0, is_exit=True)  # Use mask to better match the histograms
    G.add_node(N4ITKBiasFieldCorrection(), [0, 1])
    G.add_node(NyulNormalizer(), [2, 1], is_exit=True)
    G.add_node(LinearRescale(mean=5000., std=2500.), [3, 1])
    G.add_node(TypeCastNode(sitk.sitkUInt16), 4, is_exit=True)
    G.add_node(SignalIntensityRebinning(num_of_bins=256, quantiles=[0.01, 0.99]), [4, 1], is_exit=True)
    G.add_node(SignalIntensityRebinning(num_of_bins=256, quantiles=[0.01, 0.99]), [3, 1], is_exit=True)
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
    import matplotlib.pyplot as plt
    G = create_graph()
    G.load_node_states(3, state_dir)
    # G.plot_graph()
    # plt.show()

    """
    Multi-thread
    """
    images = [f for f in folder.iterdir() if f.name.find('nii') != -1]
    output_prefix = [im.name for im in images]
    output_directories = (out_folder.joinpath('00.HuangMask'),
                          out_folder.joinpath('01.NyulNormalized'),
                          out_folder.joinpath('02.NyulNormRescaled'),
                          out_folder.joinpath('03.NyulNormRescaledBinned'),
                          out_folder.joinpath('04.NyulNormBinned')
                          )

    # Prepare mpi inputs
    z = [output_prefix, [output_directories], images]
    # mpi_wrapper(G.mpi_execute, z, num_worker=4)
    for row in repeat_zip(*z):
        G.mpi_execute(*row)




# If this protector is absent, windows python might go into recursive import loop.
if __name__ == '__main__':
    os.chdir(r'/home/lwong/Storage/Data/NPC_Radiomics')
    # train_nyul()
    normalize_images(image_folder)
