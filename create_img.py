import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

os.makedirs('images', exist_ok=True)
# Create a 64x64 black image
image = np.zeros((64, 64))

# Draw a white horizontal bar in the center of the image
start_row = (image.shape[0] - 10) // 2
end_row = start_row + 10
image[start_row:end_row, :] = 255

# Convert the NumPy array to a PIL image for easier rotation
original_image = Image.fromarray(image)

# Display the original image with the white bar
plt.imshow(original_image, cmap='gray')
plt.axis('off')
plt.savefig('images/original_image.png')

# Rotate the original image by 1 degree at a time and save each rotation
rotated_images = []

for i in range(180):
    rotated_image = original_image.rotate(i, resample=Image.BICUBIC, center=(32, 32))
    rotated_images.append(rotated_image)

# Display the first 5 rotated images as a sample
fig, axarr = plt.subplots(1, 5, figsize=(15, 3))

for i, ax in enumerate(axarr):
    ax.imshow(rotated_images[i], cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.savefig('images/rotated_images.png')

