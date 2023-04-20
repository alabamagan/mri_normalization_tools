from pathlib import Path
from mnts.filters.geom import *
from mnts.filters.intensity import *
from mnts.filters.mnts_filters_graph import MNTSFilterGraph
import networkx as nx
import matplotlib.pyplot as plt
import SimpleITK as sitk

from mnts.utils import repeat_zip
from mnts.filters import mpi_wrapper
from mnts.filters.intensity import NyulNormalizer
from mnts.mnts_logger import MNTSLogger
import pprint

# If this protector is absent, windows python might go into recursive import loop.
def main():
    with MNTSLogger('default', keep_file=False, verbose=True) as _:
        # Create the normalization graph.
        G = MNTSFilterGraph()

        # Add filter nodes to the graph.
        G.add_node(SpatialNorm(out_spacing=[1, 1, 0]))
        G.add_node(OtsuThresholding(), 0)    # Use mask to better match teh histograms
        G.add_node(NyulNormalizer(), [0, 1])
        G.add_node(RangeRescale(0, 5000), [2,1], is_exit=True) # Label this as the

        """
        Prepare the arguments
        """
        image_folder = Path(r'./example_data')
        temp_output_folder = Path(r'./example_data/output/.EG_04_temp/')
        images = [f for f in image_folder.iterdir() if f.name.find('nii') != -1]
        out_names = [f.name for f in images]

        # Single-thread
        #--------------
        # This zip function accepts a tuple of iterables, different length is allowed.
        # See more in `repeat_zip`.
        # Arguments of `prepare_training_files` method are:
        #   ([[node(s) to train], [output names], [output_folder(s)], [input_images]])
        z = ([2], out_names, [temp_output_folder], images)
        for args in repeat_zip(*z):
            G.prepare_training_files(*args)

        # Multi-thread
        #-------------
        result = mpi_wrapper(G.prepare_training_files, z)
        # [None, None, None], this function returns nothing.

        """
        Training
        """
        # Train and save the states.
        G.train_node(2, temp_output_folder, temp_output_folder.joinpath("EG_04_States"))

        """
        Inference
        """
        output_save_dir = Path(r'./example_data/output/EG_04')
        output_save_dir.mkdir(parents=True, exist_ok=True)
        G.load_node_states(2, temp_output_folder.joinpath("EG_04_States"))
        for im in images:
            save_im = G.execute(im)
            fname = output_save_dir.joinpath(im.name).resolve().__str__()
            print(f"Saving to {fname}")
            sitk.WriteImage(save_im[3], fname)

    return 0

if __name__ == '__main__':
    main()