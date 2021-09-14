from pathlib import Path
from mnts.filters.geom import *
from mnts.filters.intensity import *
from mnts.filters.mnts_filters import MNTSFilterGraph
import networkx as nx
import matplotlib.pyplot as plt
import SimpleITK as sitk

from mnts.utils import repeat_zip
from mnts.filters import mpi_wrapper
from mnts.filters.intensity import NyulNormalizer

import pprint
# If this protector is absent, windows python might go into recursive import loop.
if __name__ == '__main__':
    # Create the normalization graph.
    G = MNTSFilterGraph()

    # Add filter nodes to the graph.
    G.add_node(SpatialNorm(out_spacing=[1, 1, 0]))
    G.add_node(OtsuTresholding(), 0)    # Use mask to better match teh histograms
    G.add_node(N4ITKBiasFieldCorrection(), [0, 1])
    G.add_node(NyulNormalizer(), [2, 1])
    G.add_node(RangeRescale(0, 5000), 3, is_exit=True)
    G.add_node(SignalIntensityRebinning(num_of_bins=256), 3, is_exist=True)

    # Plot the graph
    G.plot_graph()
    plt.show()

    # Borrow the trained features, please run example 04 if this reports error.
    state_path = Path(r'./example_data/output/.EG_04_temp/EG_04_States/2_NyulNormalizer.npz')
    G.load_node_states(3, state_path)

    # Write output images
    image_folder = Path(r'./example_data')
    images = [f for f in image_folder.iterdir() if f.name.find('nii') != -1]
    output_save_dir = Path(r'./example_data/output/EG_05')
    output_save_dir.mkdir(parents=True, exist_ok=True)
    for im in images:
        save_im = G.execute(im)
        fname = output_save_dir.joinpath(im.name).resolve().__str__()
        print(f"Saving to {fname}")
        sitk.WriteImage(save_im[3], fname)
