import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
from ..mnts_filters import MNTSFilter


__all__ = ['RemoveShoulder']


class RemoveShoulder(MNTSFilter):
    r"""
    A geometric filter that takes a mask and calculates the slice area along a specified dimension,
    then crops the 3D volume to a single slice along that dimension.

    This filter finds the slice with the smallest mask area along the specified dimension and
    reduces the 3D volume to just that single slice. The search starts from the end of the volume
    and works backwards by the barrier amount to avoid unwanted regions like shoulder areas.

    The filter only crops along the specified dimension, leaving the other two dimensions at full size.
    This is useful for extracting representative slices for analysis or preprocessing.

    Attributes:
        min_area_threshold (float):
            Minimum area threshold as a fraction of the maximum area. Slices with area below
            this threshold will be ignored to avoid noise. Default is 0.1 (10% of max area).

        barrier (int):
            Number of slices to count downwards from when searching for the minimum area slice.
            The search starts from the end of the volume and works backwards by this many slices.
            Default is 10 slices.

        dimension (int):
            The dimension along which to calculate slice areas and perform cropping.
            0 for axial (z-axis), 1 for coronal (y-axis), 2 for sagittal (x-axis).
            Default is 0 (axial).

    Example:
        >>> from mnts.filters.geom import RemoveShoulder
        >>> crop_filter = RemoveShoulder(barrier=15, dimension=0)
        >>> cropped_image = crop_filter.filter(image, mask)
    """

    def __init__(self,
                  min_area_threshold: float = 0.1,
                  barrier: int = 10,
                  dimension: int = 0):
        super(RemoveShoulder, self).__init__()
        self.min_area_threshold = min_area_threshold
        self.barrier = barrier
        self.dimension = dimension

    @property
    def min_area_threshold(self):
        return self._min_area_threshold

    @min_area_threshold.setter
    def min_area_threshold(self, threshold: float):
        if not 0 <= threshold <= 1:
            raise ValueError("Min area threshold must be between 0 and 1")
        self._min_area_threshold = threshold
        
    @property
    def barrier(self):
        return self._barrier
    
    @barrier.setter
    def barrier(self, val: float):
        assert 0 <= val
        self._barrier = val

    @property
    def dimension(self):
        return self._dimension

    @dimension.setter
    def dimension(self, val: int):
        if not 0 <= val <= 2:
            raise ValueError("Dimension must be 0, 1, or 2")
        self._dimension = val
        
    def filter(self,
               image: Union[str, Path, sitk.Image],
               mask: Union[str, Path, sitk.Image]) -> sitk.Image:
        """
        Apply geometric mask-based cropping to the input image.
        """
        image = self.read_image(image)
        mask = self.read_image(mask)
        size_before = image.GetSize()

        if image.GetSize() != mask.GetSize():
            raise RuntimeError("Image and mask must have the same dimensions")

        if image.GetSpacing() != mask.GetSpacing():
            self._logger.warning("Image and mask have different spacing. Using image spacing for output.")
        if image.GetDirection() != mask.GetDirection():
            self._logger.warning("Image and mask have different direction. Using image direction for output.")
        if image.GetOrigin() != mask.GetOrigin():
            self._logger.warning("Image and mask have different origin. Using image origin for output.")

        # Cache geometry
        self._image_spacing = image.GetSpacing()   # [sx, sy, sz]
        self._image_origin = image.GetOrigin()     # [ox, oy, oz]
        self._image_direction = image.GetDirection()  # 3x3 flattened

        mask_array = sitk.GetArrayFromImage(mask)  # numpy order [z, y, x]
        if mask_array.ndim != 3:
            raise ValueError("Mask must be 3D.")
        if np.count_nonzero(mask_array) == 0:
            raise ValueError("Mask is empty (all zeros).")

        slice_areas = self._calculate_slice_areas(mask_array)
        if slice_areas.size == 0 or np.all(slice_areas == 0):
            raise ValueError("All slice areas are zero; cannot determine cropping slice.")
        self._logger.info(f"slice areas: {slice_areas}")

        min_slice_idx = self._find_minimum_area_slice(slice_areas)
        if min_slice_idx is None:
            raise ValueError("No valid slice found with sufficient area. Check mask quality or thresholds.")

        crop_bounds = self._get_cropping_bounds(mask_array, min_slice_idx)
        cropped_image = self._crop_image(image, crop_bounds)

        self._logger.info(f"Cropped image at slice {min_slice_idx} with bounds: {crop_bounds}. Original {size_before} -> New {cropped_image.GetSize()}")
        return cropped_image

    def _calculate_slice_areas(self, mask_array: np.ndarray) -> np.ndarray:
        """
        Calculate physical area (mm^2) per slice along given dimension.
        numpy array is [z, y, x]; dimension: 0=z, 1=y, 2=x
        """
        # Spacing in SITK is [sx, sy, sz]; array dims are [z, y, x]
        sx, sy, sz = self._image_spacing if hasattr(self, '_image_spacing') else (1.0, 1.0, 1.0)

        # Pixel area is the product of the spacings in the two axes orthogonal to the slicing axis
        if self.dimension == 0:      # slicing along z => area in y-x plane
            pixel_area = sy * sx
        elif self.dimension == 1:    # slicing along y => area in z-x plane
            pixel_area = sz * sx
        elif self.dimension == 2:    # slicing along x => area in z-y plane
            pixel_area = sz * sy
        else:
            raise ValueError(f"Invalid dimension {self.dimension}")

        num_slices = mask_array.shape[self.dimension]
        areas = np.zeros(num_slices, dtype=np.float64)

        for i in range(num_slices):
            if self.dimension == 0:
                slice_data = mask_array[i, :, :]     # [y, x]
            elif self.dimension == 1:
                slice_data = mask_array[:, i, :]     # [z, x]
            else:  # self.dimension == 2
                slice_data = mask_array[:, :, i]     # [z, y]
            areas[i] = np.count_nonzero(slice_data) * pixel_area

        # Replace 0 with max value to ignore empty slices.
        areas[areas == 0] = areas.max()
        self._logger.debug(f"{areas = }")
        return areas

    def _find_minimum_area_slice(self, slice_areas: np.ndarray) -> Optional[int]:
        """
        Apply threshold and barrier to find minimal area slice index.
        """
        if slice_areas.size == 0:
            return None

        max_area = float(np.max(slice_areas))
        if max_area <= 0.0:
            self._logger.warning("Maximum slice area is zero.")
            return None

        min_threshold = max_area * self.min_area_threshold

        # Barrier logic: count downwards from the end of the volume
        n = len(slice_areas)
        if self.barrier >= n:
            self._logger.warning(f"Barrier ({self.barrier}) >= number of slices ({n}), no valid indices.")
            return None
        end_idx = n - int(self.barrier)  # count backwards from the end
        candidate_indices = np.arange(0, end_idx)

        valid_mask = slice_areas[candidate_indices] >= min_threshold
        if not np.any(valid_mask):
            self._logger.warning("No slices meet the minimum area threshold after applying barrier.")
            return None

        valid_indices = candidate_indices[valid_mask]
        min_idx_rel = int(np.argmin(slice_areas[valid_indices]))
        min_slice_idx = int(valid_indices[min_idx_rel])

        self._logger.info(
            f"Found minimum area slice at index {min_slice_idx} "
            f"(area={slice_areas[min_slice_idx]:.3f}, max={max_area:.3f}, barrier={self.barrier})"
        )
        return min_slice_idx

    def _get_cropping_bounds(self, mask_array: np.ndarray, min_slice_idx: int) -> Tuple[list, list]:
        """
        Build crop bounds as (lower_crop, upper_crop) for SimpleITK.Crop().
        Crops from top (max index in LPS convention) to the minimum area slice.

        Returns:
            Tuple of (lower_crop, upper_crop) where each is a list [x, y, z] of integers
            representing the number of pixels to crop from start and end respectively.
        
        Note: mask_array is in numpy order [z, y, x] but SimpleITK uses [x, y, z]
        dimension 0 = z-axis (numpy index 0), dimension 1 = y-axis (numpy index 1), dimension 2 = x-axis (numpy index 2)
        In LPS convention, top (superior) corresponds to max index.
        """
        # Step 1: Extract the 2D slice along the chosen dimension (for validation only)
        if self.dimension == 0:      # Axial: extract slice at z=min_slice_idx, result shape [y, x]
            min_slice = mask_array[min_slice_idx, :, :]
        elif self.dimension == 1:    # Coronal: extract slice at y=min_slice_idx, result shape [z, x]
            min_slice = mask_array[:, min_slice_idx, :]
        else:                        # Sagittal: extract slice at x=min_slice_idx, result shape [z, y]
            min_slice = mask_array[:, :, min_slice_idx]

        # Step 2: Validate that the selected slice has some mask content
        if np.count_nonzero(min_slice) == 0:
            raise ValueError(f"No mask pixels found in slice {min_slice_idx}")

        # Step 3: Calculate crop parameters for SimpleITK.Crop()
        # Crop from top (max index) to min_slice_idx (inclusive)
        # SimpleITK expects [x, y, z] order for crop parameters
        # mask_array.shape is [z, y, x] in numpy order
        
        if self.dimension == 0:
            # Dimension 0 (z-axis): crop from top (max z) to min_slice_idx
            # Keep from min_slice_idx to max index (top)
            lower_crop = [0, 0, min_slice_idx]  # crop from start to min_slice_idx
            upper_crop = [0, 0, 0]  # don't crop from end (keep to top/max)
        elif self.dimension == 1:
            # Dimension 1 (y-axis): crop from top (max y) to min_slice_idx
            lower_crop = [0, min_slice_idx, 0]  # crop from start to min_slice_idx
            upper_crop = [0, 0, 0]  # don't crop from end (keep to top/max)
        else:  # dimension == 2
            # Dimension 2 (x-axis): crop from top (max x) to min_slice_idx
            lower_crop = [min_slice_idx, 0, 0]  # crop from start to min_slice_idx
            upper_crop = [0, 0, 0]  # don't crop from end (keep to top/max)

        return (lower_crop, upper_crop)

    def _crop_image(self, image: sitk.Image, crop_bounds: Tuple[list, list]) -> sitk.Image:
        """
        Crop using SimpleITK Crop with lower and upper crop parameters and update origin.
        """
        # Expecting crop_bounds as (lower_crop, upper_crop) from _get_cropping_bounds
        lower_crop, upper_crop = crop_bounds

        # Apply cropping in index space; spacing/direction preserved automatically.
        cropped_img = sitk.Crop(image, lower_crop, upper_crop)

        # Update origin explicitly to match the shifted index start.
        sx, sy, sz = image.GetSpacing()
        ox, oy, oz = image.GetOrigin()
        new_origin = [
            ox + lower_crop[0] * sx,  # x offset
            oy + lower_crop[1] * sy,  # y offset
            oz + lower_crop[2] * sz,  # z offset
        ]
        cropped_img.SetOrigin(new_origin)
        return cropped_img

# ... rest of code ...

