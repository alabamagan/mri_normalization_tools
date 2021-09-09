from mnts.filters.geom import SpatialNorm
from mnts.filters.intensity import ZScoreNorm, RangeRescale
from mnts.filters.mnts_filters import MNTSFilterPipeline
import SimpleITK as sitk
from pathlib import Path

eg_input = Path(r"./example_data/MRI_01.nii.gz")
if not eg_input.is_file():
    raise IOError("Error opening example data.")


im = sitk.ReadImage(eg_input.resolve().__str__())
orig_dtype = im.GetPixelID()

# Setting up the filters
F0 = SpatialNorm(out_spacing=[1, 1, 0]) # 0 or negative value to keep original spacing.
F1 = ZScoreNorm()
F2 = RangeRescale(0, 5000)

print(f"{F0}\n{F1}\n{F2}")

# Execute the filters. (There are better way to do this in later examples)
im = F0.filter(im)
im = F1.filter(im) # Image is casted to float when doing linear rescales
im = F2.filter(im)

# Cast the image back into its original datatype
im = sitk.Cast(im, orig_dtype)

# Save the image
eg_output = Path(r"./example_data/output/EG_01.nii.gz")
eg_output.parent.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(im, eg_output.resolve().__str__())