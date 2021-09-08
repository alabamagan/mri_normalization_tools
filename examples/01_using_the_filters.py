from mnts.filters.geom import SpatialNorm
from mnts.filters.intensity import ZScoreNorm, RangeRescale
from mnts.filters.mnts_filters import MNTSFilterPipeline
import SimpleITK as sitk

im = sitk.ReadImage(r"")

