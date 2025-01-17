import SimpleITK as sitk
import numpy as np
import random
import os
import json

def load_mhd_images(file_paths):
    """Load mhd images and return them as numpy arrays."""
    images = []
    for path in file_paths:
        image = sitk.ReadImage(path)
        images.append(sitk.GetArrayFromImage(image))
    return images

def extract_patches(hr_images, label_images, hr_file_paths, patch_size=(256, 256), roi_bounds=(200, 1700, 400, 1500), min_label_pixels=500, patches_per_image=5):
    """
    Extract patches from HR images and label images with specified conditions.

    Args:
        hr_images (list of np.ndarray): High-resolution images.
        label_images (list of np.ndarray): Corresponding label images.
        patch_size (tuple): Size of the patch (height, width).
        roi_bounds (tuple): Bounds for random cropping (y_min, y_max, x_min, x_max).
        min_label_pixels (int): Minimum number of label pixels required in a patch.
        patches_per_image (int): Number of patches to extract per image.

    Returns:
        list of tuples: List of (hr_patch, label_patch) pairs.
    """
    y_min, y_max, x_min, x_max = roi_bounds
    patch_height, patch_width = patch_size
    extracted_patches = []

    for hr_image, label_image, hr_path in zip(hr_images, label_images, hr_file_paths):
        patches_extracted = 0
        attempts = 0

        while patches_extracted < patches_per_image and attempts < 100:
            # Randomly select the top-left corner for the patch
            top_left_y = random.randint(y_min, y_max - patch_height)
            top_left_x = random.randint(x_min, x_max - patch_width)

            # Extract patches
            hr_patch = hr_image[top_left_y:top_left_y + patch_height, top_left_x:top_left_x + patch_width]
            label_patch = label_image[top_left_y:top_left_y + patch_height, top_left_x:top_left_x + patch_width]

            # Check conditions
            if np.any(hr_patch == 0):
                attempts += 1
                continue

            if np.sum(label_patch > 0) < min_label_pixels:
                attempts += 1
                continue

            metadata = {
                "file_path": hr_path,
                "top_left_y": top_left_y,
                "top_left_x": top_left_x,
                "patch_size": patch_size
            }
            extracted_patches.append((hr_patch, label_patch, metadata))
            patches_extracted += 1

    return extracted_patches

def save_patches_as_mhd(patches, output_dir, prefix="patch"):
    """Save patches as .mhd files along with metadata."""
    os.makedirs(output_dir, exist_ok=True)
    metadata_list = []

    for i, (hr_patch, label_patch, metadata) in enumerate(patches):
        hr_image = sitk.GetImageFromArray(hr_patch)
        label_image = sitk.GetImageFromArray(label_patch)

        hr_output_path = os.path.join(output_dir, f"{prefix}_hr_{i}.mhd")
        label_output_path = os.path.join(output_dir, f"{prefix}_label_{i}.mhd")

        sitk.WriteImage(hr_image, hr_output_path)
        sitk.WriteImage(label_image, label_output_path)

        # Add file paths to metadata
        metadata["hr_patch_path"] = hr_output_path
        metadata["label_patch_path"] = label_output_path
        metadata_list.append(metadata)

    # Save metadata as JSON
    metadata_path = os.path.join(output_dir, f"{prefix}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata_list, f, indent=4)

# Example usage

name_list = ["100", "200", "300", "400", "500", "600", "700", "800"]# 読み込むMHDファイルのパス
hr_file_paths = [f"/take/dataset/microCT_slices_1792_2/hr/{name}.mhd" for name in name_list]

label_file_paths = [f"/take/dataset/microCT_slices_1792_2/interlobular_septum_mask/tophat/{name}_0.03.mhd" for name in name_list]

output_dir = "/take/dataset/microCT_slices_1792_2/patches"
os.makedirs(output_dir, exist_ok=True)

# Load images
hr_images = load_mhd_images(hr_file_paths)
label_images = load_mhd_images(label_file_paths)

# Extract patches
patches = extract_patches(hr_images, label_images, hr_file_paths)

# Save patches
save_patches_as_mhd(patches, output_dir=output_dir)
