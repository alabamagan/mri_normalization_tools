from mnts.filters.geom import *
from mnts.filters.intensity import *
from mnts.filters.mnts_filters_graph import MNTSFilterGraph
import networkx as nx
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path

# Create the normalization graph.
G = MNTSFilterGraph()

# Add filter nodes to the graph.
G.add_node(SpatialNorm(out_spacing=[1, 1, 0]))
G.add_node(OtsuThresholding(), 0)    # Use mask to better match teh histograms
G.add_node(ZScoreNorm(), [0, 1])
G.add_node(RangeRescale(0, 5000), [2,1], is_exit=True) # Label this as the output node

# Plot and show the graph
G.plot_graph()
plt.show()

# Load image
eg_input = Path(r"./example_data/MRI_01.nii.gz")
if not eg_input.is_file():
    raise IOError("Error opening example data.")
im = sitk.ReadImage(eg_input.resolve().__str__())
orig_dtype = im.GetPixelID()

# Execute the graph
im = G.execute(im)[3] # node 3 is the only output node.

# Cast the image back into its original datatype
im = sitk.Cast(im, orig_dtype)

# Save the image
eg_output = Path(r"./example_data/output/EG_03.nii.gz")
eg_output.parent.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(im, eg_output.resolve().__str__())
