import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
# import SimpleITK as sitk
# import torch
# import cv2

# from torch import nn

import time
from torch_topological.nn import CubicalComplex, WassersteinDistance

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_persistence_diagram(persistence, output_file_name="output"):
    """Plots the persistence diagram."""
    plt.clf()
    gd.plot_persistence_diagram(persistence)
    plt.title("Persistence Diagram " + output_file_name)
    plt.savefig(output_file_name + "_diagram.png")

def save_image_list(image_list, output_file_name="output"):
    """Saves the image list to a file."""
    plt.clf()
    fig, axs = plt.subplots(1, len(image_list))
    for i, img in enumerate(image_list):
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')
        axs[i].set_title("Image " + str(i))
    plt.savefig(output_file_name + "_images.png")

start_time = time.time()

image_data = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0.8, 0],
        [0, 0.4, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

image_data_2 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0.4, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

image_data_3 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0.3, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0.5, 0.5, 0, 1, 0],
        [0, 1, 0, 0.5, 0.5, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

image_list = [image_data, image_data_2, image_data_3]
# image_list = [image_data]
# combined_images = np.array(image_list)

# save_image_list(image_list, "image_list")

# # Convert the numpy array to a PyTorch tensor
# img_batch = torch.tensor(combined_images, device=device, dtype=torch.float32).unsqueeze(1)

# print(image_data.size, image_data.shape)

persistences = []
for i, img in enumerate(image_list):
    cc = gd.CubicalComplex(
        dimensions=image_data.shape, top_dimensional_cells=1-img.flatten()
    )
    persistence = cc.persistence()
    print("gudhi:", persistence)
    print("vertices:", cc.vertices(), cc.vertices().shape)
    print("top:", cc.top_dimensional_cells(), cc.top_dimensional_cells().shape)
    print("betti:", cc.betti_numbers())
    print("bettis:", cc.persistent_betti_numbers(from_value=0, to_value=1))
    print("cofaces:", cc.cofaces_of_persistence_pairs())
    persistences.append(persistence)
    # plot_persistence_diagram(persistence, "gudhi_" + str(i))

print("gudhi:", persistences)

end_time = time.time()
print("Time taken:", end_time - start_time)
# cofaces = cc.cofaces_of_persistence_pairs()

# for idx, (birth, death) in enumerate(persistence):
#     if death[1] == float("inf"):
#         persistence[idx] = (birth, (death[0], image_data.max()))

# print("gudhi:", persistence)
# print("cofaces:", cofaces)
# plot_persistence_diagram(persistence, "cubical")

cubical = CubicalComplex()
wd_loss = WassersteinDistance(q=2)
per_cc = cubical(image_data)
print(len(per_cc))
print(len(per_cc[0]))
print(len(per_cc[0][0]))
print(per_cc[0])
print(per_cc[0][0])

# loss = wd_loss(per_cc, per_cc)