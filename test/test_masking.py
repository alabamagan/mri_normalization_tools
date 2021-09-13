from mnts.filters.geom import *
from mnts.filters.intensity import *
from mnts.filters.mnts_filters import MNTSFilterGraph
import networkx as nx
import matplotlib.pyplot as plt
import SimpleITK as sitk

from netgraph import Graph

G = nx.DiGraph()
G.add_nodes_from([0, 1, 2, 3, 4])
G.add_edges_from([(0, 1),
                  (1, 2),
                  (0, 2),
                  (1, 3),
                  (3, 4)])
Graph(graph=G, arrows=True, node_layout='dot', node_labels=True,
      node_label_fontdict={'size':14}, node_label_offset=0.1)
plt.show()
