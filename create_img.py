import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_image_and_hist(image, filename):
    plt.clf()
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.savefig("images/{}.png".format(filename))

    plt.clf()
    plt.hist(image.ravel(), bins=256, color="red", alpha=0.7)
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
image[start_row:end_row, 26:36] = 64

# Convert the NumPy array to a PIL image for easier rotation
original_image = Image.fromarray(image)

save_image_and_hist(image, "original_image")

blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

save_image_and_hist(blurred_image, "blurred_image")

noise = np.random.normal(1, 3, blurred_image.shape).astype(np.float64)
noisy_image = cv2.add(blurred_image, noise)
# 0から256の範囲に収める
noisy_image = np.clip(noisy_image, 0, 255).astype(np.float64)

save_image_and_hist(noisy_image, "noisy_image")

noisy_image = Image.fromarray(noisy_image.astype(np.uint8))

# Rotate the original image by 1 degree at a time and save each rotation
rotated_images = []

os.makedirs("images/rotated_images", exist_ok=True)
for i in range(180):
    rotated_image = noisy_image.rotate(
        i, resample=Image.BICUBIC, center=(32, 32)
    )
    rotated_images.append(rotated_image)

    save_image_and_hist(
        np.array(rotated_image), "rotated_images/{:03d}".format(i + 1)
    )

# Display the first 5 rotated images as a sample
# fig, axarr = plt.subplots(1, 5, figsize=(15, 3))

# for i, ax in enumerate(axarr):
#     ax.imshow(rotated_images[i], cmap='gray')
#     ax.axis('off')

# plt.tight_layout()
# plt.savefig('images/rotated_images.png')
