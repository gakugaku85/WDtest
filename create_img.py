import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from PIL import Image


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


os.makedirs("images", exist_ok=True)

image = np.zeros((100, 100)) + 64
center_line = image.shape[0] // 2
image[center_line - 2 : center_line + 1, :] = 192

for i in range(5, 8):
    os.makedirs("images/noisy_rotated_{}".format(2**i), exist_ok=True)
    os.makedirs("images/noisy_rotated_{}_mhd".format(2**i), exist_ok=True)
    noisy_images = []
    for j in range(0, 180):
        rotated_image = rotate_image(image, j)
        noisy_image = create_noisy_image(rotated_image, 1, 2**i)
        noisy_images.append(noisy_image)
        save_image(noisy_image, "images/noisy_rotated_{}/rotated_{}".format(2**i, j))
        save_image_mhd(
            noisy_image, "images/noisy_rotated_{}_mhd/rotated_{}".format(2**i, j)
        )

    # Display the first 5 rotated images as a sample
    fig, axarr = plt.subplots(1, 5, figsize=(15, 3))
    for k, ax in enumerate(axarr):
        ax.imshow(noisy_images[k * 36], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("images/rotated_images_{}.png".format(2**i))
