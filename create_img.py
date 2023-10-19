import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

os.makedirs('images', exist_ok=True)
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

# Display the original image with the white bar
plt.imshow(original_image, cmap='gray')
plt.axis('off')
plt.savefig('images/original_image.png')

plt.clf()

plt.hist(image.ravel(), bins=256, color="red", alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Count')
plt.savefig('images/original_histogram.png')

blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

plt.clf()
plt.imshow(blurred_image, cmap='gray')
plt.axis('off')
plt.savefig('images/blurred_image.png')

plt.clf()
plt.hist(blurred_image.ravel(), bins=256, color='green', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Count')
plt.savefig('images/blurred_histogram.png')


noise = np.random.normal(1, 3, blurred_image.shape).astype(np.float64)
noisy_image = cv2.add(blurred_image, noise)



plt.clf()
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')
plt.savefig('images/noisy_image.png')

plt.clf()
plt.hist(noisy_image.ravel(), bins=256, color='black', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Count')
plt.savefig('images/noisy_histogram.png')

# Rotate the original image by 1 degree at a time and save each rotation
rotated_images = []

for i in range(180):
    rotated_image = original_image.rotate(i, resample=Image.BICUBIC, center=(32, 32))
    rotated_images.append(rotated_image)
    # plt.imshow(rotated_image, cmap='gray')
    # plt.axis('off')
    # plt.savefig('images/rotated_image_{:03d}.png'.format(i))

# Display the first 5 rotated images as a sample
# fig, axarr = plt.subplots(1, 5, figsize=(15, 3))

# for i, ax in enumerate(axarr):
#     ax.imshow(rotated_images[i], cmap='gray')
#     ax.axis('off')

# plt.tight_layout()
# plt.savefig('images/rotated_images.png')

