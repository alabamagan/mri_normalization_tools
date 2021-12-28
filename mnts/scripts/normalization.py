from ..filters import MNTSFilterGraph
from pathlib import Path
from typing import Union
from mnts.utils import repeat_zip

import argparse


def train_normalization(G: MNTSFilterGraph,
                        input_dir: Union[Path, str],
                        output_dir: Union[Path, str],
                        num_worker=0):
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    G.train_node(None)