import unittest
from mnts.filters import MNTSFilterGraph
from mnts.filters.data_node import *
from mnts.filters.intensity import *
from mnts.filters.geom import *
from pathlib import Path

test_yaml =\
"""
DataNode: 

SpatialNorm:
    out_spacing: [0.5, 0.5, 0]
    _ext:
        upstream: 0

HuangThresholding:
    closing_kernel_size: 10
    _ext:
        upstream: 1 
        is_exit: True

N4ITKBiasFieldCorrection:
    _ext:
        upstream: [1, 2]
    
NyulNormalizer:
    _ext:
        upstream: [3, 2]
        is_exit: True

"""

def create_graph() -> MNTSFilterGraph:
    r"""Create the normalization graph"""
    G = MNTSFilterGraph()

    # Add filter nodes to the graph.
    G.add_node(DataNode())
    G.add_node(SpatialNorm(out_spacing=[0.4492, 0.4492, 4]), 0)
    G.add_node(HuangThresholding(closing_kernel_size=10), 1, is_exit=True)  # Use mask to better match the histograms
    G.add_node(N4ITKBiasFieldCorrection(), [1, 2])
    G.add_node(NyulNormalizer(), [3, 2], is_exit=True)
    return G

class TestGraph(unittest.TestCase):
    def test_graph_print(self):
        G = create_graph()
        self.assertIsInstance(str(G), str)
        print(G)

    def test_graph_from_yaml(self):
        # Create file from str
        with open('_test_graph.yaml', 'w') as f:
            f.write(test_yaml)
        G = MNTSFilterGraph.CreateGraphFromYAML('_test_graph.yaml')
        self.assertTrue(isinstance(G,
                                   MNTSFilterGraph))
        print(G)
        Path('_test_graph.yaml').unlink()

    def test_running_graph(self):
        pass

if __name__ == '__main__':
    unittest.main()