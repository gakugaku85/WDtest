import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_image(image, filename):
    plt.clf()
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.savefig("{}.png".format(filename))

def calculate_snr(image):
    signal_mean = np.mean(image)
    noise = image - signal_mean
    noise_std = np.std(noise)
    snr = 20 * np.log10(signal_mean / noise_std)
    return snr

def AddGaussianNoise(image, mean, sigma):
    noise = np.random.normal(mean, sigma, np.shape(image))
    noisy_image = image + noise
    noisy_image[noisy_image > 255] = 255
    noisy_image[noisy_image < 0] = 0
    noisy_image = noisy_image.astype(np.uint8)    # Float -> Uint
    return noisy_image

image = np.zeros((64, 64)) + 64
center_line = image.shape[0] // 2
image[center_line-2:center_line+2, :] = 196

noisy_image = AddGaussianNoise(image, 0, 4)
Image.fromarray(noisy_image.astype(np.uint8)).save("images/noisy_center_line.png")

original_snr = calculate_snr(image)
print(f"元の画像のSN比: {original_snr} dB")

Image.fromarray(image.astype(np.uint8)).save("images/center_line.png")

mean = 0
var = 4  # ノイズの分散を大きくする
sigma = var ** 0.5
gaussian = np.random.normal(mean, sigma, image.shape)
noisy_image = np.clip(image + gaussian, 0, 255).astype(np.uint8)

# SNRの計算と出力
noisy_snr = calculate_snr(noisy_image)
print(f"ノイズを加えた後のSN比: {noisy_snr} dB")

os.makedirs("images", exist_ok=True)
Image.fromarray(noisy_image.astype(np.uint8)).save("images/noisy_center_line.png")



