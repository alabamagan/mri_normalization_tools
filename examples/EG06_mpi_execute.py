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
from mnts.mnts_logger import MNTSLogger

"""
To allow graph to be pickled, it need a wrapper function
"""
def create_graph() -> MNTSFilterGraph:
    r"""Create the normalization graph"""
    G = MNTSFilterGraph()

    # Add filter nodes to the graph.
    G.add_node(SpatialNorm(out_spacing=[0.5, 0.5, 0.5]))
    G.add_node(HuangThresholding(closing_kernel_size=10), 0, is_exit=True)  # Use mask to better match the histograms
    G.add_node(N4ITKBiasFieldCorrection(), [0, 1])
    G.add_node(NyulNormalizer(), [2, 1], is_exit=True)
    return G

"""
If training is need (not necessary to wrap it around a function though)
"""
def train_nyul():
    with MNTSLogger['train_nyul'] as logger:
        # Create training outputs folders
        out_temp_folder = Path('./example_data/output/.EG_06_temp')
        out_temp_folder.mkdir(exist_ok=True, parents=True)

        # Define inputs for preparing training materials
        images = [f for f in Path('./example_data/').glob("MRI*.nii.gz")]
        logger.info(f"Training with images: {images}")
        out_names = [i.name for i in images]
        z = ([3], out_names, [out_temp_folder], images)

        # Create the graph
        G = create_graph()
        result = mpi_wrapper(G.prepare_training_files, z)  # Sometimes training node require normalized input,
                                                           # which are prepared by this function

        G.train_node(3, out_temp_folder, out_temp_folder.joinpath('Trained_states'))

"""
'main'
"""
def normalize_images():
    with MNTSLogger['normmalize_images'] as logger:
        state_dir = Path('./example_data/output/.EG_06_temp/Trained_states')
        eg_input = Path('./example_data')
        out_folder = Path('./example_data/output/EG_06_temp/')

        G = create_graph()
        G.load_node_states(3, str(state_dir))

        images = [f for f in eg_input.glob("MRI*nii.gz")]
        output_prefix = [i.name for i in images]
        logger.info(f"Globbed: {images}")

        # Prepare
        seg_z = [output_prefix, [out_folder], images]

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
    with MNTSLogger('default', keep_file=False, verbose=True) as logger:
        train_nyul()
        normalize_images()

if __name__ == '__main__':
    main()
