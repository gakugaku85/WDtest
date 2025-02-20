import math
import os

import numpy as np
import SimpleITK as sitk
from PIL import Image


def calculate_psnr(gt, input):
    # gt and input have range [0, 255]
    gt = gt.astype(np.float64)
    input = input.astype(np.float64)
    mse = np.mean((gt - input)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def mhd_to_png(input_mhd, output_png):
    # Read the .mhd file using SimpleITK
    image = sitk.ReadImage(input_mhd)

    # Convert the image to a numpy array
    image_array = sitk.GetArrayFromImage(image)

    # Normalize the image array to the range [0, 255]
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255
    image_array = image_array.astype(np.uint8)
    # x, y = 720, 1195
    x, y = 820, 650
    image_size = (256, 256)
    # image_size = (64, 64)
    clip_image = image_array[x:x+image_size[0], y:y+image_size[1]]

    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(clip_image)

    # Save the PIL image as a PNG file
    pil_image.save(output_png)

    return clip_image

# Example usage
input_mhd = 'images_1/original_val1/0.mhd'

input_mhd_list = ["/take/gaku/SR3/SR3-chestCT/experiments/val_ori_best/results/0_7_hr.mhd",
                  "/take/gaku/SR3/SR3-chestCT/experiments/val_ori_best/results/0_7_inf.mhd",
                  "/take/gaku/SR3/SR3-chestCT/experiments/val_ori_best/results/0_7_sr.mhd",
                  "/take/gaku/SR3/SR3-chestCT/experiments/val_wd_10_best/results/0_7_sr.mhd"]

out_path = "/take/gaku/SR3/SR3-chestCT/experiments/forslide/"

input_list = ["hr", "inf", "sr_ori", "sr_wd_10"]

img_list = []

os.makedirs(out_path, exist_ok=True)
for i, input_mhd in enumerate(input_mhd_list):
    output_png = os.path.join(out_path, input_mhd.split("/")[-3] + "_" + input_mhd.split("/")[-1].split(".")[0] + ".png")
    img_list.append(mhd_to_png(input_mhd, output_png))
    # mhd_to_png(input_mhd, output_png)

# Calculate PSNR
psnr_list = []
for i, img in enumerate(img_list):
    psnr = calculate_psnr(img_list[0], img)
    psnr_list.append(psnr)
    print(f"PSNR between hr and {input_list[i]}: {psnr:.2f} dB")