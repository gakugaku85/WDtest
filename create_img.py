import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm


def calculate_snr(image, noise_sigma):
    signal = image.max() - image.min()
    snr = signal / noise_sigma
    return snr


def save_image(image, filename):
    Image.fromarray(image.astype(np.uint8)).save("{}.png".format(filename))


def save_image_mhd(image, filename):
    sitk.WriteImage(sitk.GetImageFromArray(image), "{}.mhd".format(filename))


def save_image_hist(image, filename):
    plt.clf()
    plt.hist(image.ravel(), bins=256, color="black", alpha=0.7)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Count")
    plt.savefig("{}_hist.png".format(filename))


def concentration_profile(image, filename):
    # y direction
    x_vals = np.linspace(0, 0, 13)
    y_vals = np.linspace(25, 38, 13)
    pixel_values = []
    for x, y in zip(x_vals, y_vals):
        av_pixel = 0
        for i in range(image.shape[1]):
            av_pixel += image[int(y), int(i)]
        av_pixel /= image.shape[1]
        pixel_values.append(av_pixel)

    plt.clf()
    plt.plot(pixel_values, label="Pixel values along the line", color="blue")
    plt.title("Pixel Values along the Center Line")
    plt.xlabel("Position along the line")
    plt.ylabel("Pixel Value")
    plt.savefig("{}_gram_y.png".format(filename))


def create_noisy_image(image, blur_sigma, noise_sigma):
    kennel_size = 7
    mean = 0

    blurred_image = cv2.GaussianBlur(image, (kennel_size, kennel_size), blur_sigma)

    noise = np.random.normal(mean, noise_sigma, blurred_image.shape).astype(np.float64)
    noisy_image = cv2.add(blurred_image, noise)

    noisy_snr = calculate_snr(image, noise_sigma)

    return noisy_image


def rotate_image(image, angle):
    center = tuple(np.array(image.shape) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        image.shape[1::-1],
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    h, w = rotated_image.shape
    center_h, center_w = h // 2, w // 2

    rotated_image = rotated_image[
        center_h - 32 : center_h + 32, center_w - 32 : center_w + 32
    ]

    return rotated_image


def resize_mhd_save(image_array, output_path, target_size, j):
    resized_mhd = cv2.resize(image_array, target_size, interpolation=cv2.INTER_CUBIC)
    resample_mhd_image = sitk.GetImageFromArray(resized_mhd)
    sitk.WriteImage(resample_mhd_image, output_path + "/{}.mhd".format(j))

    return resized_mhd

def normalize_image(image_array):
    image = sitk.GetImageFromArray(image_array)
    upper_percentile = np.percentile(image_array, 99.95)
    lower_percentile = np.percentile(image_array, 0.05)
    normalized_image_array = np.clip((image_array - lower_percentile) / (upper_percentile - lower_percentile), 0, 1)
    normalized_image_array = (normalized_image_array * 255).astype(np.uint8)
    normalized_image = sitk.GetImageFromArray(normalized_image_array)
    normalized_image.SetOrigin(image.GetOrigin())
    normalized_image.SetSpacing(image.GetSpacing())
    normalized_image.SetDirection(image.GetDirection())

    norm_image = sitk.GetArrayFromImage(normalized_image)

    return norm_image


os.makedirs("images", exist_ok=True)

image = np.zeros((100, 100)) + 64
center_line = image.shape[0] // 2
image[center_line - 1 : center_line + 2, :] = 192
hr_size = (64, 64)
lr_size = (16, 16)
line_space = 3

for i in range(5, 8):
    # os.makedirs("images/noisy_rotated_{}_mhd_{}".format(2**i, line_space), exist_ok=True)
    img_sigma_path = "images/sigma{}".format(2**i)
    hr_folder_mhd = img_sigma_path + "/hr_64"
    lr_folder_mhd = img_sigma_path + "/lr_16"
    sr_folder_mhd = img_sigma_path + "/sr_64"
    hr_folder_mhd_val1 = img_sigma_path + "val1/hr_64"
    lr_folder_mhd_val1 = img_sigma_path + "val1/lr_16"
    sr_folder_mhd_val1 = img_sigma_path + "val1/sr_64"
    hr_folder_mhd_val2 = img_sigma_path + "val2/hr_64"
    lr_folder_mhd_val2 = img_sigma_path + "val2/lr_16"
    sr_folder_mhd_val2 = img_sigma_path + "val2/sr_64"
    os.makedirs(hr_folder_mhd, exist_ok=True)
    os.makedirs(lr_folder_mhd, exist_ok=True)
    os.makedirs(sr_folder_mhd, exist_ok=True)
    os.makedirs(hr_folder_mhd_val1, exist_ok=True)
    os.makedirs(lr_folder_mhd_val1, exist_ok=True)
    os.makedirs(sr_folder_mhd_val1, exist_ok=True)
    os.makedirs(hr_folder_mhd_val2, exist_ok=True)
    os.makedirs(lr_folder_mhd_val2, exist_ok=True)
    os.makedirs(sr_folder_mhd_val2, exist_ok=True)

    noisy_images = []
    for j in tqdm(range(0, 180), desc="{}sigma Creating images".format(2**i)):
        rotated_image = rotate_image(image, j)
        noisy_image = create_noisy_image(rotated_image, 1, 2**i)

        norm_image = normalize_image(noisy_image)

        if j % 3 == 0:
            if j % 6 == 0:
                hr_image = resize_mhd_save(norm_image, hr_folder_mhd_val1, hr_size, j)
                lr_image = resize_mhd_save(hr_image, lr_folder_mhd_val1, lr_size, j)
                sr_image = resize_mhd_save(lr_image, sr_folder_mhd_val1, hr_size, j)
            else:
                hr_image = resize_mhd_save(norm_image, hr_folder_mhd_val2, hr_size, j)
                lr_image = resize_mhd_save(hr_image, lr_folder_mhd_val2, lr_size, j)
                sr_image = resize_mhd_save(lr_image, sr_folder_mhd_val2, hr_size, j)
        else:
            hr_image = resize_mhd_save(norm_image, hr_folder_mhd, hr_size, j)
            lr_image = resize_mhd_save(hr_image, lr_folder_mhd, lr_size, j)
            sr_image = resize_mhd_save(lr_image, sr_folder_mhd, hr_size, j)
