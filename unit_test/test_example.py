import importlib
import unittest
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Suppress interactive matplotlib windows before any mnts/example imports.
import matplotlib
matplotlib.use('Agg')

_EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))


class TestExamples(unittest.TestCase):
    """Run example scripts in numeric order.

    EG05 requires the trained model states produced by EG04, so this class
    tracks whether EG04 succeeded and skips EG05 if it did not.
    """

    _eg04_passed = False

    @classmethod
    def setUpClass(cls):
        cls._orig_cwd = Path.cwd()
        os.chdir(_EXAMPLES_DIR)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._orig_cwd)

    # ------------------------------------------------------------------
    # plt.show() appears in several examples; patch it to be a no-op so
    # tests don't block waiting for a GUI window.
    # ------------------------------------------------------------------

    def _run(self, module_name: str):
        with patch("matplotlib.pyplot.show"):
            mod = importlib.import_module(module_name)
            importlib.reload(mod)   # re-execute in case module was already cached
            mod.main()

    def test_example_01(self):
        self._run("EG01_using_the_filters")

    def test_example_02(self):
        self._run("EG02_using_filter_w_pipeline")

    def test_example_03(self):
        self._run("EG03_using_filter_w_graph")

    def test_example_04(self):
        self._run("EG04_using_filters_that_require_train")
        TestExamples._eg04_passed = True

    def test_example_05(self):
        if not TestExamples._eg04_passed:
            self.skipTest("EG04 (training step) must pass before EG05 can run")
        self._run("EG05_another_graph_example")

    def test_example_06(self):
        self._run("EG06_mpi_execute")

    def test_example_07(self):
        self._run("EG07_create_graph_from_yaml")

    def test_example_08(self):
        self._run("EG08_geom_mask_crop_example")

    def test_example_09(self):
        # EG09 demos print usage and attempt paths that may not exist;
        # main() handles missing paths gracefully.
        self._run("EG09_dicom_tag_printing")


if __name__ == '__main__':
    unittest.main()
