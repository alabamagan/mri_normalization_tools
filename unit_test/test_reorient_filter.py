#!/usr/bin/env python3
"""
Unit tests for the ReorientFilter.
"""

import unittest
import numpy as np
import SimpleITK as sitk
from mnts.filters.geom import ReorientFilter
from mnts.mnts_logger import MNTSLogger


def _make_image(size=(10, 12, 14), spacing=(1.0, 2.0, 3.0), orientation='LPS'):
    """Create a synthetic 3D image with a known orientation."""
    img = sitk.Image(list(size), sitk.sitkFloat32)
    img.SetSpacing(list(spacing))
    img.SetOrigin([0.0, 0.0, 0.0])
    img = sitk.DICOMOrient(img, orientation)
    # Fill with deterministic signal so we can verify content is preserved
    arr = np.arange(np.prod(size), dtype=np.float32).reshape(size[::-1])  # numpy [z,y,x]
    filled = sitk.GetImageFromArray(arr)
    filled.CopyInformation(img)
    return filled


def _make_mask(size=(10, 12, 14), orientation='LPS'):
    """Create a binary mask with the same orientation."""
    mask = sitk.Image(list(size), sitk.sitkUInt8)
    mask.SetSpacing([1.0, 2.0, 3.0])
    mask.SetOrigin([0.0, 0.0, 0.0])
    mask = sitk.DICOMOrient(mask, orientation)
    arr = np.ones(size[::-1], dtype=np.uint8)
    filled = sitk.GetImageFromArray(arr)
    filled.CopyInformation(mask)
    return filled


class TestReorientFilterInit(unittest.TestCase):
    """Tests for constructor and property validation."""

    def test_default_orientation(self):
        f = ReorientFilter()
        self.assertEqual(f.target_orientation, 'RAI')

    def test_custom_orientation(self):
        f = ReorientFilter('LPS')
        self.assertEqual(f.target_orientation, 'LPS')

    def test_orientation_stored_uppercase(self):
        f = ReorientFilter('lps')
        self.assertEqual(f.target_orientation, 'LPS')

    def test_setter_updates_value(self):
        f = ReorientFilter('RAI')
        f.target_orientation = 'RPI'
        self.assertEqual(f.target_orientation, 'RPI')

    def test_invalid_type_raises(self):
        with self.assertRaises(AssertionError):
            ReorientFilter(123)

    def test_invalid_chars_raises(self):
        with self.assertRaises(AssertionError):
            ReorientFilter('XYZ')

    def test_wrong_length_raises(self):
        with self.assertRaises(AssertionError):
            ReorientFilter('RA')

    def test_too_long_raises(self):
        with self.assertRaises(AssertionError):
            ReorientFilter('RAIS')

    def test_all_valid_axes_accepted(self):
        """Spot-check a variety of valid orientation strings."""
        for orient in ('LPS', 'RAS', 'LAS', 'RPS', 'AIL', 'PIL', 'AIR', 'PIR'):
            with self.subTest(orient=orient):
                f = ReorientFilter(orient)
                self.assertEqual(f.target_orientation, orient)


class TestReorientFilterOutputOrientation(unittest.TestCase):
    """Tests that the filter correctly reorients images."""

    def setUp(self):
        MNTSLogger.set_global_log_level('warning')

    def test_output_orientation_matches_target(self):
        """After filtering, the image orientation code should equal the target."""
        src_orient = 'LPS'
        tgt_orient = 'RAS'
        img = _make_image(orientation=src_orient)

        f = ReorientFilter(tgt_orient)
        out = f.filter(img)

        actual = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            out.GetDirection()
        )
        self.assertEqual(actual, tgt_orient)

    def test_noop_when_already_correct_orientation(self):
        """Filtering to the same orientation should leave the image unchanged."""
        orient = 'RAS'
        img = _make_image(orientation=orient)

        f = ReorientFilter(orient)
        out = f.filter(img)

        actual = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            out.GetDirection()
        )
        self.assertEqual(actual, orient)

    def test_image_size_preserved_up_to_axis_permutation(self):
        """Total voxel count must be conserved after reorientation."""
        img = _make_image(size=(10, 12, 14), orientation='LPS')
        f = ReorientFilter('RAS')
        out = f.filter(img)
        self.assertEqual(np.prod(out.GetSize()), np.prod(img.GetSize()))

    def test_callable_interface(self):
        """__call__ should behave identically to filter()."""
        img = _make_image(orientation='LPS')
        f = ReorientFilter('RAS')
        out_call = f(img)
        out_filter = f.filter(img)
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(out_call),
            sitk.GetArrayFromImage(out_filter)
        )


class TestReorientFilterWithMask(unittest.TestCase):
    """Tests for the two-argument (image + mask) variant."""

    def setUp(self):
        MNTSLogger.set_global_log_level('warning')

    def test_returns_tuple_when_mask_provided(self):
        img = _make_image(orientation='LPS')
        mask = _make_mask(orientation='LPS')
        f = ReorientFilter('RAS')
        result = f.filter(img, mask)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_returns_image_when_no_mask(self):
        img = _make_image(orientation='LPS')
        f = ReorientFilter('RAS')
        result = f.filter(img)
        self.assertIsInstance(result, sitk.Image)

    def test_both_outputs_have_correct_orientation(self):
        tgt = 'RAS'
        img = _make_image(orientation='LPS')
        mask = _make_mask(orientation='LPS')
        f = ReorientFilter(tgt)
        out_img, out_mask = f.filter(img, mask)

        img_orient = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            out_img.GetDirection()
        )
        mask_orient = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            out_mask.GetDirection()
        )
        self.assertEqual(img_orient, tgt)
        self.assertEqual(mask_orient, tgt)

    def test_mask_none_returns_image(self):
        img = _make_image(orientation='LPS')
        f = ReorientFilter('RAS')
        result = f.filter(img, None)
        self.assertIsInstance(result, sitk.Image)

    def test_image_and_mask_same_size_after_filter(self):
        img = _make_image(size=(10, 12, 14), orientation='LPS')
        mask = _make_mask(size=(10, 12, 14), orientation='LPS')
        f = ReorientFilter('RAS')
        out_img, out_mask = f.filter(img, mask)
        self.assertEqual(out_img.GetSize(), out_mask.GetSize())


class TestReorientFilterRoundTrip(unittest.TestCase):
    """Reorienting A->B->A should recover the original image data."""

    def setUp(self):
        MNTSLogger.set_global_log_level('warning')

    def test_round_trip_image_data(self):
        src_orient = 'LPS'
        tgt_orient = 'RAS'
        img = _make_image(size=(8, 10, 12), orientation=src_orient)

        f_fwd = ReorientFilter(tgt_orient)
        f_rev = ReorientFilter(src_orient)

        mid = f_fwd.filter(img)
        recovered = f_rev.filter(mid)

        np.testing.assert_array_almost_equal(
            sitk.GetArrayFromImage(recovered),
            sitk.GetArrayFromImage(img),
            decimal=5
        )

    def test_round_trip_mask_data(self):
        src_orient = 'LPS'
        tgt_orient = 'RAS'
        img = _make_image(size=(8, 10, 12), orientation=src_orient)
        mask = _make_mask(size=(8, 10, 12), orientation=src_orient)

        f_fwd = ReorientFilter(tgt_orient)
        f_rev = ReorientFilter(src_orient)

        mid_img, mid_mask = f_fwd.filter(img, mask)
        rec_img, rec_mask = f_rev.filter(mid_img, mid_mask)

        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(rec_mask),
            sitk.GetArrayFromImage(mask)
        )


if __name__ == '__main__':
    unittest.main()
