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
    reduces the 3D volume to just that single slice. The search of slice is done from inferior to superior
    for identifying the shoulder. To prevent top slices (top of the head) to be accidentedly selected, a
    barrier setting is implement that protects the top n slices from being selected.


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

    Example:
        >>> from mnts.filters.geom import RemoveShoulder
        >>> crop_filter = RemoveShoulder(barrier=15, dimension=0)
        >>> cropped_image = crop_filter.filter(image, mask)
    """

    def __init__(self,
                  min_area_threshold: float = 0.1,
                  barrier: int = 10):
        super(RemoveShoulder, self).__init__()
        self.min_area_threshold = min_area_threshold
        self.barrier = barrier
        self.dimension = 0 # This is now fixed

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

        # Determine orientation for the specified dimension
        orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection())
        # Map dimension to orientation string index (dimension 0->z->2, 1->y->1, 2->x->0)
        orient_idx = 2 - self.dimension
        axis_orientation = orientation[orient_idx]

        # Determine if Superior is at high or low index
        # For anatomy: S (Superior) should be at high index in standard orientation
        # If axis is I (Inferior), then Superior is at low index (reversed)
        self._is_reversed = axis_orientation == 'I' if self.dimension == 0 else \
                            axis_orientation == 'A' if self.dimension == 1 else \
                            axis_orientation == 'L'

        self._logger.info(f"Orientation: {orientation}, Dimension {self.dimension} axis: {axis_orientation}, Reversed: {self._is_reversed}")

        mask_array = sitk.GetArrayFromImage(mask)  # numpy order [z, y, x]
        if mask_array.ndim != 3:
            raise ValueError("Mask must be 3D.")
        if np.count_nonzero(mask_array) == 0:
            raise ValueError("Mask is empty (all zeros).")

        slice_areas = self._calculate_slice_areas(mask_array)
        if slice_areas.size == 0 or np.all(slice_areas == 0):
            raise ValueError("All slice areas are zero; cannot determine cropping slice.")
        self._logger.info(f"Slice areas: {slice_areas}")

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
        sx, sy, sz = self._image_spacing

        # Pixel area is the product of the spacings in the two axes orthogonal to the slicing axis
        if self.dimension == 0:      # slicing along z => area in y-x plane
            pixel_area = sy * sx
        elif self.dimension == 1:    # slicing along y => area in z-x plane
            pixel_area = sz * sx
        else:                         # slicing along x => area in z-y plane
            pixel_area = sz * sy

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

        # Replace 0 with max value to ignore empty slices
        max_area = areas.max() if areas.max() > 0 else 1.0
        areas[areas == 0] = max_area
        self._logger.debug(f"Calculated areas: {areas}")
        return areas

    def _find_minimum_area_slice(self, slice_areas: np.ndarray) -> Optional[int]:
        """
        Apply threshold and barrier to find minimal area slice index.
        Barrier is applied from the Superior end (where shoulders typically are).
        """
        if slice_areas.size == 0:
            return None

        max_area = float(np.max(slice_areas))
        if max_area <= 0.0:
            self._logger.warning("Maximum slice area is zero.")
            return None

        min_threshold = max_area * self.min_area_threshold

        # Determine Superior end based on orientation
        n = len(slice_areas)
        if self.barrier >= n:
            self._logger.warning(f"Barrier ({self.barrier}) >= number of slices ({n}), no valid indices.")
            return None

        # Superior (where head are) determination:
        # If NOT reversed: Superior is at high index (end of array), crop from low index
        # If reversed: Superior is at low index (start of array), crop from high index
        if not self._is_reversed:
            # S -> Superior at high index, apply barrier from end
            superior_barrier_idx = n - int(self.barrier)
            candidate_indices = np.arange(0, superior_barrier_idx)
        else:
            # I -> Superior at low index, apply barrier from start
            superior_barrier_idx = int(self.barrier)
            candidate_indices = np.arange(superior_barrier_idx, n)

        if len(candidate_indices) == 0:
            self._logger.warning("No candidate indices after applying barrier.")
            return None

        valid_mask = slice_areas[candidate_indices] >= min_threshold
        if not np.any(valid_mask):
            self._logger.warning("No slices meet the minimum area threshold after applying barrier.")
            return None

        valid_indices = candidate_indices[valid_mask]
        min_idx_rel = int(np.argmin(slice_areas[valid_indices]))
        min_slice_idx = int(valid_indices[min_idx_rel])

        self._logger.info(
            f"Found minimum area slice at index {min_slice_idx} "
            f"(area={slice_areas[min_slice_idx]:.3f}, max={max_area:.3f}, barrier={self.barrier}, reversed={self._is_reversed})"
        )
        return min_slice_idx

    def _get_cropping_bounds(self, mask_array: np.ndarray, min_slice_idx: int) -> Tuple[list, list]:
        """
        Build crop bounds as (lower_crop, upper_crop) for SimpleITK.Crop().
        Crops from Superior (shoulder side) to the minimum area slice.

        Returns:
            Tuple of (lower_crop, upper_crop) where each is a list [x, y, z] of integers.
        """
        # Validate the selected slice has mask content
        if self.dimension == 0:
            min_slice = mask_array[min_slice_idx, :, :]
        elif self.dimension == 1:
            min_slice = mask_array[:, min_slice_idx, :]
        else:
            min_slice = mask_array[:, :, min_slice_idx]

        if np.count_nonzero(min_slice) == 0:
            raise ValueError(f"No mask pixels found in slice {min_slice_idx}")

        # Calculate crop bounds in SimpleITK order [x, y, z]
        # mask_array.shape is [z, y, x] in numpy order
        n_slices = mask_array.shape[self.dimension]

        # Determine which end to crop from based on orientation
        if not self._is_reversed:
            # S -> Superior at high index: crop from low index up to min_slice_idx
            # Keep slices [min_slice_idx, end], remove [0, min_slice_idx-1]
            crop_from_start = min_slice_idx
            crop_from_end = 0
        else:
            # I -> Superior at low index: crop from high (inferior) index down to min_slice_idx
            # Keep slices [0, min_slice_idx], remove [min_slice_idx+1, end]
            crop_from_end = n_slices - min_slice_idx - 1
            crop_from_start = 0

        # Build crop parameters in SimpleITK order [x, y, z]
        if self.dimension == 0:  # z-axis
            lower_crop = [0, 0, crop_from_start]
            upper_crop = [0, 0, crop_from_end]
        elif self.dimension == 1:  # y-axis
            lower_crop = [0, crop_from_start, 0]
            upper_crop = [0, crop_from_end, 0]
        else:  # x-axis
            lower_crop = [crop_from_start, 0, 0]
            upper_crop = [crop_from_end, 0, 0]

        self._logger.debug(f"Cropping bounds: lower={lower_crop}, upper={upper_crop} (keeping slice {min_slice_idx})")
        return (lower_crop, upper_crop)

    def _crop_image(self, image: sitk.Image, crop_bounds: Tuple[list, list]) -> sitk.Image:
        """
        Crop using SimpleITK Crop with lower and upper crop parameters.
        """
        lower_crop, upper_crop = crop_bounds
        cropped_img = sitk.Crop(image, lower_crop, upper_crop)
        return cropped_img