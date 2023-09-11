import numpy as np
import SimpleITK as sitk
from typing import Optional, Tuple
def create_dummy_image(size   : Optional[Tuple[int  , int  , int ]]  = (10, 10, 10),
                       origin : Optional[Tuple[float, float, float]] = (0., 0., 0.),
                       spacing: Optional[Tuple[float, float, float]] = (1., 1., 1.)) -> sitk.Image:
    # Create a 3D image with dimensions 10x10x10
    image = sitk.GetImageFromArray(np.random.rand(*size) * 100)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    return image

def create_dummy_segmentation(size   : Optional[Tuple[int  , int  , int]]   = (10 , 10 , 10),
                              spacing: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
                              origin : Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0),
                              center : Optional[Tuple[int  , int  , int]]   = (5  , 5  , 5),
                              cube_size : Optional[int] = 3) -> sitk.Image:
    # Create a segmentation for the image where each voxel is labeled with a 1 or 0
    segmentation_array = np.zeros(size)

    # Compute the coordinates of the cube in the image
    x_min = max(0, center[0] - cube_size // 2)
    x_max = min(size[0], center[0] + cube_size // 2)
    y_min = max(0, center[1] - cube_size // 2)
    y_max = min(size[1], center[1] + cube_size // 2)
    z_min = max(0, center[2] - cube_size // 2)
    z_max = min(size[2], center[2] + cube_size // 2)

    # Set the voxels in the cube to 1
    segmentation_array[x_min:x_max, y_min:y_max, z_min:z_max] = 1

    # Convert the numpy array to a SimpleITK image
    segmentation = sitk.GetImageFromArray(segmentation_array)
    segmentation.SetSpacing(spacing)
    segmentation.SetOrigin(origin)
    return segmentation