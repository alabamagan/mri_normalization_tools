#!/usr/bin/env python3
"""
Unit tests for the geom_mask_crop filter functionality.
"""

import unittest
import numpy as np
import SimpleITK as sitk
from mnts.filters.geom import RemoveShoulder
from mnts.mnts_logger import MNTSLogger



class TestGeomMaskCrop(unittest.TestCase):
    """Test cases for the RemoveShoulder (geom_mask_crop) filter."""

    def setUp(self):
        """Set up test data."""
        self.logger = MNTSLogger['test']
        MNTSLogger.set_global_log_level('debug')

        # Create a simple 3D test image (10x10x10)
        self.test_image = sitk.Image([10, 10, 10], sitk.sitkFloat32)
        self.test_image.SetSpacing([1.0, 1.0, 1.0])
        self.test_image.SetOrigin([0.0, 0.0, 0.0])

        # Fill with ones
        self.test_image = sitk.Add(self.test_image, 1.0)

    def test_barrier_zero(self):
        """Test barrier functionality with barrier=0 (no barrier)."""
        # Create a mask where slice 2 has the smallest area
        mask_array = np.ones((10, 10, 10))
        mask_array[0:2, :, :] = 0  # Remove first 2 slices
        mask_array[8:10, :, :] = 0  # Remove last 2 slices
        mask_array[2, 5:10, 5:10] = 0  # Make slice 2 smaller
        mask = sitk.GetImageFromArray(mask_array.astype(np.uint8))

        # Test with barrier=0
        filter_obj = RemoveShoulder(barrier=0, min_area_threshold=0.0)
        min_slice = filter_obj._find_minimum_area_slice(
            filter_obj._calculate_slice_areas(mask_array)
        )

        # Should find slice 2 (index 2) as it has the smallest area
        self.assertEqual(min_slice, 2)

    def test_barrier_nonzero(self):
        """Test barrier functionality with barrier > 0."""
        # Create a mask where slice 2 has the smallest area but we set barrier=3
        mask_array = np.ones((10, 10, 10))
        mask_array[0:2, :, :] = 0  # Remove first 2 slices
        mask_array[8:10, :, :] = 0  # Remove last 2 slices
        mask_array[2, 5:10, 5:10] = 0  # Make slice 2 smaller (area = 25)
        mask_array[4, 3:10, 3:10] = 0  # Make slice 4 smaller (area = 49) but after barrier

        # Calculate slice areas
        temp_filter = RemoveShoulder()
        slice_areas = temp_filter._calculate_slice_areas(mask_array)

        # Test with barrier=3 (ignore first 3 slices)
        filter_obj = RemoveShoulder(barrier=3, min_area_threshold=0.0)
        min_slice = filter_obj._find_minimum_area_slice(slice_areas)

        # Should find slice 4 (index 4) as it has the smallest area after barrier
        self.assertEqual(min_slice, 4)

    def test_barrier_larger_than_array(self):
        """Test barrier functionality when barrier >= array length."""
        mask_array = np.ones((5, 10, 10))
        slice_areas = np.array([100, 50, 30, 80, 20])

        # Test with barrier larger than array
        filter_obj = RemoveShoulder(barrier=10, min_area_threshold=0.0)
        min_slice = filter_obj._find_minimum_area_slice(slice_areas)

        # Should return None as no valid slices after barrier
        self.assertIsNone(min_slice)

    def test_min_area_threshold_with_barrier(self):
        """Test interaction between barrier and min_area_threshold."""
        mask_array = np.ones((10, 10, 10))
        mask_array[0:2, :, :] = 0  # Remove first 2 slices
        mask_array[8:10, :, :] = 0  # Remove last 2 slices

        slice_areas = np.array([100, 10, 25, 80, 49, 60, 70, 80, 90, 100])

        # Test with barrier=3 and min_area_threshold that excludes slice 4
        filter_obj = RemoveShoulder(barrier=3, min_area_threshold=0.5)  # threshold = 50
        min_slice = filter_obj._find_minimum_area_slice(slice_areas)
        self.assertEqual(min_slice, 5) # Should ignore 10 & 25

    def test_filter_full_pipeline(self):
        """Test the complete filter pipeline with barrier."""
        # Create a test mask with known minimum area slice
        mask_array = np.ones((10, 10, 10))
        mask_array[0:3, :, :] = 0  # Remove first 3 slices
        mask_array[7:10, :, :] = 0  # Remove last 3 slices
        mask_array[5, 3:8, 3:8] = 0  # Make slice 5 smaller (area = 25)
        mask = sitk.GetImageFromArray(mask_array.astype(np.uint8))

        # Apply filter with barrier=4 (ignore first 4 slices)
        filter_obj = RemoveShoulder(barrier=4, padding=0, min_area_threshold=0.0)
        cropped_image = filter_obj.filter(self.test_image, mask)

        # The filter should crop around slice 5
        cropped_size = cropped_image.GetSize()
        self.assertEqual(cropped_size[0], 1)  # Should be 1 slice thick due to padding=0


if __name__ == '__main__':
    unittest.main()