import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import SimpleITK as sitk

def calculate_snr(image, original_image):
    signal_mean = np.mean(image)
    noise = image - original_image
    noise_std = np.std(noise)
    snr = 20 * np.log10(signal_mean / noise_std)
    return snr

def save_image_hist(image, filename):
    plt.clf()
    plt.hist(image.ravel(), bins=256, color="black", alpha=0.7)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Count")
    plt.savefig("{}_hist.png".format(filename))

def concentration_profile(image, filename):
    # # x direction
    # x_vals = np.linspace(0, image.shape[0] - 1, image.shape[0])
    # y_vals = np.linspace(32, 32, image.shape[1])
    # pixel_values = [image[int(y), int(x)] for x, y in zip(x_vals, y_vals)]

    # plt.clf()
    # plt.plot(pixel_values, label="Pixel values along the line", color='blue')
    # plt.title("Pixel Values along the Center Line")
    # plt.xlabel("Position along the line")
    # plt.ylabel("Pixel Value")
    # plt.savefig("{}_gram_x.png".format(filename))

    # y direction
    x_vals = np.linspace(0, 0, 13)
    y_vals = np.linspace(25, 38, 13)
    pixel_values = []
    for x, y in zip(x_vals, y_vals):
        av_pixel = 0
        for i in range(64):
            av_pixel += image[int(y), int(i)]
        av_pixel /= 10
        pixel_values.append(av_pixel)

    plt.clf()
    plt.plot(pixel_values, label="Pixel values along the line", color='blue')
    plt.title("Pixel Values along the Center Line")
    plt.xlabel("Position along the line")
    plt.ylabel("Pixel Value")
    plt.savefig("{}_gram_y.png".format(filename))

os.makedirs("images", exist_ok=True)

image = np.zeros((64, 64)) + 64
center_line = image.shape[0] // 2
image[center_line - 2 : center_line + 1, :] = 192

Image.fromarray(image.astype(np.uint8)).save("images/original_pil.png")
save_image_hist(image, "images/original_image")
concentration_profile(image, "images/original_image")
sitk.WriteImage(sitk.GetImageFromArray(image), "images/original_image.mhd")

kennel_size = 7
mean = 0
blur_sigma = 1
noise_sigma = 32

blurred_image = cv2.GaussianBlur(image, (kennel_size, kennel_size), blur_sigma)

noise = np.random.normal(mean, noise_sigma, blurred_image.shape).astype(np.float64)
noisy_image = cv2.add(blurred_image, noise)
#0~255に正規化
# noisy_image = (noisy_image - np.min(noisy_image)) / (np.max(noisy_image) - np.min(noisy_image)) * 255
# noisy_image = np.clip(noisy_image, 0, 255).astype(np.float64)

Image.fromarray(noisy_image.astype(np.uint8)).save("images/noisy" + "_bsigma=" + str(blur_sigma) + "_nsigma=" + str(noise_sigma) + "_pil.png")
save_image_hist(noisy_image, "images/noisy" + "_bsigma=" + str(blur_sigma) + "_nsigma=" + str(noise_sigma))
concentration_profile(noisy_image, "images/noisy" + "_bsigma=" + str(blur_sigma) + "_nsigma=" + str(noise_sigma))
sitk.WriteImage(sitk.GetImageFromArray(noisy_image), "images/noisy_" + "_bsigma=" + str(blur_sigma) + "_nsigma=" + str(noise_sigma) + ".mhd")

# original_snr = calculate_snr(image, image)
# print(f"元の画像のSN比: {original_snr} dB")
noisy_snr = calculate_snr(noisy_image, image)
print(f"ノイズを加えた後のSN比: {noisy_snr} dB")


# noisy_image = Image.fromarray(noisy_image.astype(np.uint8))

# Rotate the original image by 1 degree at a time and save each rotation
# rotated_images = []

# os.makedirs("images/rotated_images", exist_ok=True)
# for i in range(180):
#     rotated_image = noisy_image.rotate(
#         i, resample=Image.BICUBIC, center=(32, 32)
#     )
#     rotated_images.append(rotated_image)
#     save_image_and_hist(
#         np.array(rotated_image), "rotated_images/{:03d}".format(i + 1)
#     )

# Display the first 5 rotated images as a sample
# fig, axarr = plt.subplots(1, 5, figsize=(15, 3))
# for i, ax in enumerate(axarr):
#     ax.imshow(rotated_images[i * 36], cmap="gray")
#     ax.axis("off")
# plt.tight_layout()
# plt.savefig("images/rotated_images.png")
