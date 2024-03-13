import unittest
import os
import subprocess
import sys
sys.path.append(os.path.abspath("../examples"))
# import importlib

class TestScript(unittest.TestCase):
    def test_example_1(self):
        os.chdir("../examples")
        from EG01_using_the_filters import main
        main()

    def test_example_2(self):
        os.chdir("../examples")
        from EG02_using_filter_w_pipeline import main
        main()

    def test_example_3(self):
        os.chdir("../examples")
        from EG03_using_filter_w_graph import main
        main()

    def test_example_4(self):
        os.chdir("../examples")
        from EG04_using_filters_that_require_train import main
        main()

    def test_example_5(self):
        os.chdir("../examples")
        from EG05_another_graph_example import main
        main()

    def test_example_6(self):
        os.chdir("../examples")
        from EG06_mpi_execute import main
        main()

    def test_example_7(self):
        os.chdir("../examples")
        from EG07_create_graph_from_yaml import main
        main()


if __name__ == '__main__':
    unittest.main()