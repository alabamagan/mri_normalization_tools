import unittest
from mnts.filters import MNTSFilterGraph
from mnts.filters.intensity import *
from mnts.filters.geom import *
from pathlib import Path

test_yaml =\
"""
SpatialNorm:
    out_spacing: [0.5, 0.5, 0]

HuangThresholding:
    closing_kernel_size: 10
    _ext:
        upstream: 0 
        is_exit: True

N4ITKBiasFieldCorrection:
    _ext:
        upstream: [0, 1]
    
NyulNormalizer:
    _ext:
        upstream: [2, 1]
        is_exit: True

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

class TestGraph(unittest.TestCase):
    def test_graph_print(self):
        G = create_graph()
        self.assertIsInstance(str(G), str)

    def test_graph_from_yaml(self):
        # Create file from str
        with open('_test_graph.yaml', 'w') as f:
            f.write(test_yaml)
        G = MNTSFilterGraph.CreateGraphFromYAML('_test_graph.yaml')
        self.assertTrue(isinstance(G,
                                   MNTSFilterGraph))
        print(G)
        Path('_test_graph.yaml').unlink()

if __name__ == '__main__':
    unittest.main()