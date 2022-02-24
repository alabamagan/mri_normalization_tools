import unittest
import shutil
import SimpleITK as sitk
import numpy as np

from typing import Union, Sequence
from mnts.scripts.dicom2nii import *
from mnts.scripts.normalization import *
from mnts.filters import MNTSFilterGraph
from mnts.filters.intensity import *
from mnts.filters.geom import *
from mnts.mnts_logger import MNTSLogger
from pathlib import Path

def create_graph() -> MNTSFilterGraph:
    r"""Create the normalization graph"""
    G = MNTSFilterGraph()

    # Add filter nodes to the graph.
    G.add_node(SpatialNorm(out_spacing=[0.4492, 0.4492, 0]))
    G.add_node(HuangThresholding(closing_kernel_size=10), 0, is_exit=True)  # Use mask to better match the histograms
    G.add_node(N4ITKBiasFieldCorrection(), [0, 1])
    G.add_node(NyulNormalizer(), [2, 1], is_exit=True)
    return G

def create_random_boxes(size: Sequence[int], box_size: Sequence[int], intensity: int):
    r"""Create an sitk image of size with a random box placed within the image"""
    x = np.zeros(size)
    corner = [np.random.randint(0, size[i] - box_size[i]) for i in range(len(size))]
    s = tuple([slice(corner[i], corner[i] + box_size[i], 1) for i in range(len(size))])
    x[s] = intensity
    return sitk.GetImageFromArray(x)


"""
Test settings
"""
N = 3 # create 3 images
out_path = Path('./temp_output')
fnames = [f"_temp{i}.nii.gz" for i in range(N)]
test_yaml =\
r"""
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

class TestScript(unittest.TestCase):
    CLEAN_FLAG = True
    def __init__(self, *args, **kwargs):
        super(TestScript, self).__init__(*args, **kwargs)
        TestScript.create_samples()

    def test_logger(self):
        with MNTSLogger('./default.log', keep_file=False, verbose=True, log_level='debug') as logger:
            logger.info("Info")
            logger.debug("debug")

            logger2 = MNTSLogger['logger2']
            logger2.info("Info")
            logger2.debug("debug")


    def test_norm_0_train(self):
        # Create graph
        G = create_graph()
        G._logger.set_verbose(1)
        _train_normalization(G, '.', str(out_path), 0)

        # Halt clean dir
        TestScript.CLEAN_FLAG = False

    def test_norm_2_train_mpi(self):
        # Create graph
        G = create_graph()
        G._logger.set_verbose(1)
        _train_normalization(G, '.', str(out_path), 16)

        # Halt clean dir
        TestScript.CLEAN_FLAG = False

    def test_norm_1_inference(self):
        G = create_graph()
        G._logger.set_verbose(1)
        _inference_normalization(G, str(out_path.joinpath("Trained_states")), ".", str(out_path), 0)

        TestScript.CLEAN_FLAG = True

    def test_norm_3_inference_mpi(self):
        G = create_graph()
        G._logger.set_verbose(1)
        _inference_normalization(G, str(out_path.joinpath("Trained_states")), ".", str(out_path), 16)

        TestScript.CLEAN_FLAG = True

    def test_console_entry_train(self):
        r"""Run this after """
        with open('_temp.yaml', 'w') as f:
            f.write(test_yaml)
        run_graph_train(f"-i . -f ./_temp.yaml -n 16 -v -o {str(out_path)}".split())
        Path('_temp.yaml').unlink()

    @staticmethod
    def create_samples():
        x = [create_random_boxes([128, 128, 30], [64, 64, 20], 255) for i in range(N)]
        [sitk.WriteImage(sitk.Cast(xx, sitk.sitkInt16), fnames[i]) for i, xx in enumerate(x)]

    @staticmethod
    def clean_dir():
        # Delete temp images and generated files
        if TestScript.CLEAN_FLAG:
            [Path(f).unlink() for f in fnames]
            shutil.rmtree(str(out_path))
            MNTSLogger.cleanup()

    def __del__(self):
        TestScript.clean_dir()
        pass

if __name__ == '__main__':
    unittest.main()
