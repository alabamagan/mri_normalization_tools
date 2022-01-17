import unittest
import os
import subprocess

class TestScript(unittest.TestCase):
    def test_example_1(self):
        os.chdir("../examples")
        subprocess.Popen("01_using_the_filters.py", cwd="../examples")

    def test_example_2(self):
        os.chdir("../examples")
        exec(open("02_using_filter_w_pipeline.py").read())

    def test_example_3(self):
        os.chdir("../examples")
        exec(open("03_using_filter_w_graph.py").read())

    def test_example_4(self):
        os.chdir("../examples")
        exec(open("04_using_filters_that_require_train.py").read())

    def test_example_5(self):
        os.chdir("../examples")
        exec(open("05_another_graph_example.py").read())

    def test_example_6(self):
        os.chdir("../examples")
        exec(open("06_mpi_execute.py").read())

    def test_example_7(self):
        os.chdir("../examples")
        exec(open("07_create_graph_from_yaml.py").read())


if __name__ == '__main__':
    unittest.main()