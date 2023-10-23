import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_image_and_hist(image, filename):
    plt.clf()
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.savefig("images/{}.png".format(filename))

    plt.clf()
    plt.hist(image.ravel(), bins=256, color="black", alpha=0.7)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Count")
    plt.savefig("images/{}_hist.png".format(filename))


os.makedirs("images", exist_ok=True)

# Create a 64x64 black image
image = np.zeros((64, 64))

# Draw a white horizontal bar in the center of the image
start_row = (image.shape[0] - 10) // 2
end_row = start_row + 10
image[start_row:end_row, :26] = 196
image[start_row:end_row, 36:] = 196
image[start_row:end_row, 26:36] = 128

kennel_sizes = [7]
blur_sigmas = [0, 5, 10, 15, 20]
means = [0]
noise_sigmas = [3, 5, 7]

for kennel_size in kennel_sizes:
    for blur_sigma in blur_sigmas:
        for mean in means:
            for noise_sigma in noise_sigmas:
                blurred_image = cv2.GaussianBlur(
                    image, (kennel_size, kennel_size), blur_sigma
                )

                noise = np.random.normal(
                    mean, noise_sigma, blurred_image.shape
                ).astype(np.float64)
                noisy_image = cv2.add(blurred_image, noise)
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.float64)

                filename = "test_images/noisy_image_"
                +"kennel={}_mean={}_b_sigma={}_n_sigma={}".format(
                    kennel_size, mean, blur_sigma, noise_sigma
                )
                save_image_and_hist(noisy_image, filename)
