import unittest
import os
import subprocess
import sys
sys.path.append(os.path.abspath("../examples"))
# import importlib

class TestScript(unittest.TestCase):
    def setUp(self):
        os.chdir("../examples")

    def tearDown(self):
        os.chdir("../unit_test")

    def test_example_1(self):
        from EG01_using_the_filters import main
        main()

    def test_example_2(self):
        from EG02_using_filter_w_pipeline import main
        main()

    def test_example_3(self):
        from EG03_using_filter_w_graph import main
        main()

    def test_example_4(self):
        from EG04_using_filters_that_require_train import main
        main()

    def test_example_5(self):
        from EG05_another_graph_example import main
        main()

    def test_example_6(self):
        from EG06_mpi_execute import main
        main()

    def test_example_7(self):
        from EG07_create_graph_from_yaml import main
        main()


if __name__ == '__main__':
    unittest.main()