import unittest
import os
import subprocess
# import importlib

class TestScript(unittest.TestCase):
    def test_example_1(self):
        from examples.EG01_using_the_filters import main
        os.chdir("../examples")
        main()

    def test_example_2(self):
        from examples.EG02_using_filter_w_pipeline import main
        os.chdir("../examples")
        main()

    def test_example_3(self):
        from examples.EG03_using_filter_w_graph import main
        os.chdir("../examples")
        main()

    def test_example_4(self):
        from examples.EG04_using_filters_that_require_train import main
        os.chdir("../examples")
        main()

    def test_example_5(self):
        from examples.EG05_another_graph_example import main
        os.chdir("../examples")
        main()

    def test_example_6(self):
        from examples.EG06_mpi_execute import main
        os.chdir("../examples")
        main()

    def test_example_7(self):
        from examples.EG07_create_graph_from_yaml import main
        os.chdir("../examples")
        main()


if __name__ == '__main__':
    unittest.main()