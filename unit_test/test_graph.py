import unittest
import SimpleITK as sitk
import tempfile

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

test_yaml_no_train = \
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
    
ZScoreNorm:
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

    def test_properties(self):
        # Nyul requires training
        graph_1 = create_graph()
        self.assertTrue(graph_1.requires_training)

        # Null graph should return None and warning message
        graph_2 = MNTSFilterGraph()
        self.assertIsNone(graph_2.requires_training)

        # This graph does not require training
        graph_3 = MNTSFilterGraph()
        graph_3.add_node(DataNode())
        graph_3.add_node(SpatialNorm(out_spacing=[0.4492, 0.4492, 4]), 0)
        graph_3.add_node(HuangThresholding(closing_kernel_size=10), 1, is_exit=True)
        self.assertFalse(graph_3.requires_training)

    def test_running_graph(self):
        pass

    def test_running_graph_no_training(self):
        """Test that a graph that doesn't require training skips the state loading step."""
        # Create a graph that doesn't require training
        with open('_test_graph_no_train.yaml', 'w') as f:
            f.write(test_yaml_no_train)

        G = MNTSFilterGraph.CreateGraphFromYAML('_test_graph_no_train.yaml')
        self.assertFalse(G.requires_training)

        # Create a spy on the load_node_states method
        original_load = G.load_node_states
        load_called = [False]  # Using a list for modification in nested function

        def spy_load(*args, **kwargs):
            load_called[0] = True
            return original_load(*args, **kwargs)

        G.load_node_states = spy_load

        # Create temporary directories for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            state_dir = Path(temp_dir) / "state"

            input_dir.mkdir()
            output_dir.mkdir()
            state_dir.mkdir()

            # Create a dummy input file
            dummy_input = input_dir / "test.nii"
            dummy_input.touch()

            # Patch mpi_execute to avoid actual execution
            original_execute = G.mpi_execute
            execute_called = [False]

            def spy_execute(*args, **kwargs):
                execute_called[0] = True
                return None

            G.mpi_execute = spy_execute

            # Import and call the function
            from mnts.scripts.normalization import _inference_normalization

            _inference_normalization(
                G=G,
                state_dir=state_dir,
                input_dir=input_dir,
                output_dir=[output_dir],
                num_worker=0
            )

            # Verify that load_node_states was not called, but mpi_execute was
            self.assertFalse(load_called[0], "load_node_states should not be called for a graph that doesn't require training")
            self.assertTrue(execute_called[0], "mpi_execute should be called regardless of training requirements")

        # Restore original methods
        G.load_node_states = original_load
        G.mpi_execute = original_execute

        # Clean up
        Path('_test_graph_no_train.yaml').unlink()

if __name__ == '__main__':
    unittest.main()