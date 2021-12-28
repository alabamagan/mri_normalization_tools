from pathlib import Path

from mnts.filters.mnts_filters_graph import MNTSFilterGraph

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

if __name__ == '__main__':
    with open('_test_graph.yaml', 'w') as f:
        f.write(test_yaml)

    G = MNTSFilterGraph.CreateGraphFromYAML('_test_graph.yaml')
    print(G)
    Path('default.log').unlink()
    Path('_test_graph.yaml').unlink()