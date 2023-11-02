import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import gudhi as gd


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

def get_pixel(img, filename):
    x_vals = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_vals = np.linspace(28, 28, img.shape[1])
    pixel_values = [img[int(y), int(x)] for x, y in zip(x_vals, y_vals)]

    plt.clf()
    plt.plot(pixel_values, label="Pixel values along the line", color='blue')
    plt.title("Pixel Values along the Center Line")
    plt.xlabel("Position along the line")
    plt.ylabel("Pixel Value")
    plt.savefig("images/{}_gram.png".format(filename))

def plot_persistence_diagram(persistence, output_file_name="output"):
    """Plots the persistence diagram."""
    plt.clf()
    print(persistence)
    gd.plot_persistence_diagram(persistence)
    plt.title("Persistence Diagram")
    plt.savefig("images/" + output_file_name + "_diagram.png")

def plot_persistence_barcode(persistence, output_file_name="output"):
    """Plots the persistence barcode."""
    plt.clf()
    gd.plot_persistence_barcode(persistence, max_intervals=0, inf_delta=100)
    plt.xlim(0, 255)
    plt.ylim(-1, len(persistence))
    plt.xticks(ticks=np.linspace(0, 255, 6), labels=np.round(np.linspace(255, 0, 6), 2))
    plt.yticks([])
    plt.title("Persistence Barcode")
    plt.savefig("images/" + output_file_name + "_barcode.png")

def plot_image_array(image_array, output_file_name="output"):
    """Plots the grayscale image based on the image array."""
    plt.clf()
    plt.imshow(image_array, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.savefig(
        "images/" + output_file_name + ".png",
        dpi=300,
        bbox_inches="tight",
    )

def persistent_homology(image_data, output_file_name="output"):
    """Computes and visualizes the persistent homology for the given image data."""
    cc = gd.CubicalComplex(
        dimensions=image_data.shape, top_dimensional_cells=255 - image_data.flatten()
    )
    persistence = cc.persistence()

    for idx, (birth, death) in enumerate(persistence):
        if death[1] == float("inf"):
            persistence[idx] = (birth, (death[0], image_data.max()))

    # Visualization
    plot_image_array(image_data, output_file_name)
    plot_persistence_diagram(persistence, output_file_name)
    plot_persistence_barcode(persistence, output_file_name)

    return persistence

os.makedirs("images", exist_ok=True)
os.makedirs("images/test_images", exist_ok=True)

# Create a 64x64 black image
image = np.zeros((64, 64)) + 64

# Draw a white horizontal bar in the center of the image
start_row = (image.shape[0] - 10) // 2
end_row = start_row + 4
image[start_row:end_row, 4:26] = 196
image[start_row:end_row, 36:-4] = 196
image[start_row:end_row, 26:36] = 128

kennel_sizes = [7]
blur_sigmas = [0.9, 1.03, 2]
means = [0]
noise_sigmas = [5 ,10 ,22.2]

image_persistent = persistent_homology(image, "original_image")
save_image_and_hist(image, "test_images/original_image")
get_pixel(image, "test_images/original_image")

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

                filename = "test_images/noisy_image_" + "kennel={}_mean={}_b_sigma={}_n_sigma={}".format(
                            kennel_size, mean, blur_sigma, noise_sigma)
                persistent_homology(noisy_image, filename)
                save_image_and_hist(noisy_image, filename)
                get_pixel(noisy_image, filename)
