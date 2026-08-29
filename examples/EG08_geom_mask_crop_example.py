#!/usr/bin/env python3
"""
Example 8: Using GeomMaskCrop Filter for ROI Extraction

This example demonstrates how to use the GeomMaskCrop filter to:
1. Take a mask and calculate axial area slice-by-slice
2. Find the slice with the smallest area
3. Crop the image to that region of interest (ROI)

This is particularly useful for head/neck imaging where you want to crop
to the narrowest part of the anatomy.
"""

from pathlib import Path
import SimpleITK as sitk
from mnts.filters.geom import RemoveShoulder
from mnts.filters.intensity import OtsuThresholding
import matplotlib.pyplot as plt

def main():
    # Set up paths
    example_data_dir = Path("./examples/example_data")
    output_dir = example_data_dir / "output" / "EG08_geom_mask_crop"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if example data exists
    if not example_data_dir.exists():
        print("Example data not found. Please run this from the project root directory.")
        return

    nii_files = list(example_data_dir.glob("*.nii.gz"))
    if not nii_files:
        print("No .nii.gz files found in example_data directory.")
        return

    # Use the first available image
    input_image_path = nii_files[0]
    print(f"Processing image: {input_image_path}")

    # Step 1: Load the image
    image = sitk.ReadImage(str(input_image_path))

    # Step 2: Create a mask using Otsu thresholding
    # In practice, you would use a proper segmentation mask
    otsu_filter = OtsuThresholding()
    mask = otsu_filter.filter(image)

    # Step 3: Apply GeomMaskCrop filter
    print("Applying GeomMaskCrop filter...")

    # Create the crop filter with custom parameters
    crop_filter = RemoveShoulder(
        min_area_threshold=0.05,  # Ignore slices with area < 5% of maximum
        barrier=5  # Count backwards 5 slices from the end when searching
    )

    # Apply the filter
    cropped_image = crop_filter.filter(image, mask)

    # Step 4: Save the results
    cropped_path = output_dir / "cropped_image.nii.gz"
    mask_path = output_dir / "mask.nii.gz"
    original_path = output_dir / "original_image.nii.gz"

    sitk.WriteImage(cropped_image, str(cropped_path))
    sitk.WriteImage(mask, str(mask_path))
    sitk.WriteImage(image, str(original_path))

    print("Results saved to:")
    print(f"  Original: {original_path}")
    print(f"  Mask: {mask_path}")
    print(f"  Cropped: {cropped_path}")

    # Step 5: Print statistics
    print("\nImage Statistics:")
    print(f"  Original size: {image.GetSize()}")
    print(f"  Cropped size: {cropped_image.GetSize()}")
    print(f"  Original spacing: {image.GetSpacing()}")
    print(f"  Cropped spacing: {cropped_image.GetSpacing()}")

    # Calculate compression ratio
    original_voxels = image.GetSize()[0] * image.GetSize()[1] * image.GetSize()[2]
    cropped_voxels = cropped_image.GetSize()[0] * cropped_image.GetSize()[1] * cropped_image.GetSize()[2]
    compression_ratio = (1 - cropped_voxels / original_voxels) * 100

    print(".1f")

    # Step 6: Optional visualization (requires matplotlib)
    try:
        # Get middle slices for visualization
        original_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        cropped_array = sitk.GetArrayFromImage(cropped_image)

        middle_slice = original_array.shape[0] // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_array[middle_slice, :, :], cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(mask_array[middle_slice, :, :], cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')

        # For cropped image, we need to find the corresponding slice
        # This is a simplified visualization - in practice you'd need to align the slices
        cropped_middle = cropped_array.shape[0] // 2
        axes[2].imshow(cropped_array[cropped_middle, :, :], cmap='gray')
        axes[2].set_title('Cropped Image')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / "comparison.png", dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Visualization saved to: {output_dir}/comparison.png")

    except ImportError:
        print("Matplotlib not available, skipping visualization")

    print("\nExample completed successfully!")

if __name__ == '__main__':
    main()