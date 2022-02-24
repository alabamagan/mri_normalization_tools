from pathlib import Path
import matplotlib.pyplot as plt

from mnts.filters.mnts_filters_graph import MNTSFilterGraph
from mnts.mnts_logger import MNTSLogger
# This is the format for the yaml file
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

def main():
    with MNTSLogger('./default.log', keep_file=False, verbose=True) as _:
        with open('_test_graph.yaml', 'w') as f:
            f.write(test_yaml)

        G = MNTSFilterGraph.CreateGraphFromYAML('_test_graph.yaml')
        G.plot_graph()
        plt.show()
        Path('_test_graph.yaml').unlink()

    return 0

if __name__ == '__main__':
    main()