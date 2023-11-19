import os

import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
from PIL import Image


def resample_mhd(input_path, output_path, target_size):
    mhd_files = [f for f in os.listdir(input_path) if f.endswith(".mhd")]

    for mhd_file in mhd_files:
        input_file_path = os.path.join(input_path, mhd_file)
        image = sitk.ReadImage(input_file_path)
        resampled_image = sitk.Resample(image, target_size, interpolator=sitk.sitkBSpline)
        output_file_path = os.path.join(output_path, mhd_file)
        sitk.WriteImage(resampled_image, output_file_path)


def resample_png(input_path, output_path, target_size):
    png_files = [f for f in os.listdir(input_path) if f.endswith(".png")]
    resampled_images = []

    for png_file in png_files:
        input_file_path = os.path.join(input_path, png_file)
        image = cv2.imread(input_file_path)
        resampled_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
        resampled_images.append(resampled_image)
        output_file_path = os.path.join(output_path, png_file)
        cv2.imwrite(output_file_path, resampled_image)

    fig, axarr = plt.subplots(1, 5, figsize=(15, 3))
    for k, ax in enumerate(axarr):
        ax.imshow(resampled_images[k * 36], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path + ".png")


if __name__ == "__main__":
    hr_size = (64, 64)
    lr_size = (16, 16)
    for i in range(5, 8):
        input_folder = "images/noisy_rotated_{}".format(2**i)
        input_folder_mhd = "images/noisy_rotated_{}_mhd".format(2**i)

        output_folder = "images/hr_{}".format(2**i)
        output_folder_mhd = "images/hr_{}_mhd".format(2**i)
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(output_folder_mhd, exist_ok=True)
        resample_png(input_folder, output_folder, hr_size)
        resample_mhd(input_folder_mhd, output_folder_mhd, hr_size)

        lr_folder = "images/lr_{}".format(2**i)
        lr_folder_mhd = "images/lr_{}_mhd".format(2**i)
        os.makedirs(lr_folder, exist_ok=True)
        os.makedirs(lr_folder_mhd, exist_ok=True)
        resample_png(input_folder, lr_folder, lr_size)
        resample_mhd(input_folder_mhd, lr_folder_mhd, lr_size)

        output_folder = "images/sr_{}".format(2**i)
        output_folder_mhd = "images/sr_{}_mhd".format(2**i)
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(output_folder_mhd, exist_ok=True)
        resample_png(lr_folder, output_folder, hr_size)
        resample_mhd(lr_folder_mhd, output_folder_mhd, hr_size)
