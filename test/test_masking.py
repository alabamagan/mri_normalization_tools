from mnts.filters.geom import *
from mnts.filters.intensity import *
from mnts.filters.mnts_filters import MNTSFilterGraph
import networkx as nx
import matplotlib.pyplot as plt
import SimpleITK as sitk

t = ZScoreNorm()
G = MNTSFilterGraph()
G.add_node(SpatialNorm(out_spacing=1))
G.add_node(OtsuTresholding(), 0)
G.add_node(ZScoreNorm(), [0, 1])
G.add_node(RangeRescale(0, 5000), [2,1], True)
G.plot_graph()
plt.show()

image = sitk.ReadImage(r"Z:\Shared\2.Projects\8.NPC_Segmentation\0A.NIFTI_ALL\Malignant\CE-T1WFS_TRA\726-T1FS+C_TRA.nii.gz")
x = G.execute(image)
# # x = Z.filter(image)
# # sitk.WriteImage(x, r"D:\temp\diudiu.nii.gz")
sitk.WriteImage(sitk.Cast(x[3], sitk.sitkUInt16), r"D:\temp\diudiu.nii.gz")
# #TODO: Keep original datatype?
# #TODO: Keep original slice thickness!!!