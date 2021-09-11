"""
Wrap functions implemented in: jcreinhold/intensity-normalization
https://github.com/jcreinhold/intensity-normalization

Note that the original design was based on images of the brain, but is generally recognized to be suitable for
other body regions.
"""

from intensity_normalization.normalize.nyul import NyulNormalize
from ..mnts_filters import MNTSFilterRequireTraining
from .intensity_base import MNTSIntensityBase

class NyulNorm(MNTSFilterRequireTraining):
    def __init__(self):
        super(NyulNorm, self).__init__()

    def train(self, images, masks):
        normalizer = NyulNormalize()
        normalizer.fit_from_directories(images, mask_dir=masks)