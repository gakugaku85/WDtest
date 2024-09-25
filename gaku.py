# png画像を読み込んで、np.arrayに変換する
# 画像は0から1に正規化される

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import SimpleITK as sitk
import gudhi as gd
import cv2

def match_cofaces_with_gudhi(image_data, cofaces, gudhi_persistence):
    height, width = image_data.shape
    result = []

    # Regular pairs (finite persistence)
    regular_pairs = []
    for dim, pairs in enumerate(cofaces[0]):
        for birth, death in pairs:
            birth_y, birth_x = np.unravel_index(birth, (height, width))
            death_y, death_x = np.unravel_index(death, (height, width))
            regular_pairs.append((dim, ((birth_y, birth_x), (death_y, death_x))))

    # Infinite pairs
    infinite_pairs = []
    for dim, births in enumerate(cofaces[1]):
        for birth in births:
            birth_y, birth_x = np.unravel_index(birth, (height, width))
            infinite_pairs.append((dim, ((birth_y, birth_x), None)))

    # Combine all pairs
    all_pairs = regular_pairs + infinite_pairs

    # Match with gudhi persistence
    for gudhi_item in gudhi_persistence:
        gudhi_dim, gudhi_persistence_value = gudhi_item

        # Find corresponding coface
        if gudhi_persistence_value[1] == float('inf'):
            l = list(gudhi_persistence_value)
            l[1] = 1
            gudhi_persistence_value = tuple(l)
            # Look for infinite pairs
            for coface_item in infinite_pairs:
                coface_dim, coface_coordinates = coface_item
                if coface_dim == gudhi_dim:
                    result.append((gudhi_dim, gudhi_persistence_value, coface_coordinates))
                    infinite_pairs.remove(coface_item)  # Remove matched item
                    break
        else:
            # Look for regular pairs
            for coface_item in regular_pairs:
                coface_dim, coface_coordinates = coface_item
                if coface_dim == gudhi_dim:
                    result.append((gudhi_dim, gudhi_persistence_value, coface_coordinates))
                    regular_pairs.remove(coface_item)  # Remove matched item
                    break

    return result

def load_image(file_path):
    img = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
    img = img / 255.0
    print(img.max(), img.min())
    return img

if __name__ == '__main__':
    img = load_image("img/31000_train.mhd")
    print(img.shape)
    lr_img = img[:, 0:64]
    sr_img = img[:, 64:128]
    hr_img = img[:, 128:192]

    img_list = [lr_img, hr_img, sr_img]
    list_name =["LR_img", "HR_img", "SR_img"]
    img_rgb_list = []

    # concat_img = np.concatenate(img_list, axis=1)
    # plt.axis("off")
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.imshow(concat_img, cmap="gray")
    # plt.savefig("img/save_image.png")

    plt.clf()
    fig, axs = plt.subplots(1, len(img_list), figsize=(8*len(img_list), 6))

    for i, img in enumerate(img_list):
        img_scaled = (img * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2RGB)

        cc = gd.CubicalComplex(
            dimensions=img.shape, top_dimensional_cells=1.0-img.flatten()
        )
        persistence = cc.persistence()
        cofaces = cc.cofaces_of_persistence_pairs()
        result = match_cofaces_with_gudhi(img, cofaces, persistence)
        distance_list = []
        for dim, (birth, death) , coordinates in result:
            # print(f"Image {i} Dimension {dim} Birth {birth} Death {death} Coordinates {coordinates}")
            if dim == 0:
                color = 'red'
            else:
                color = 'blue'

            distance = np.abs(birth - death) / np.sqrt(2)

            distance_list.append([distance, birth, death, coordinates])

            axs[i].scatter(birth, death, c=color, s=20)
            lims = [-0.1,1.1]
            axs[i].plot(lims, lims, 'k-', alpha=0.3, zorder=0)
            axs[i].set_xlim(-0.1, 1.1)
            axs[i].set_ylim(-0.1, 1.1)
            axs[i].set_title(f'Persistence Diagram {list_name[i]}')
            axs[i].set_xlabel('Birth')
            axs[i].set_ylabel('Death')

        # distance_listをdistanceで降順にソート
        distance_list = sorted(distance_list, key=lambda x: x[0], reverse=True)
        # print(f"Image {i} distance_list {distance_list}")
        for j, (distance, birth, death, coordinates) in enumerate(distance_list):
            if i == 1 and (j == 0 or j == 7):
                axs[i].scatter(birth, death, c="green", s=50)
                coord_text = f"({coordinates[0][0]}, {coordinates[0][1]})"
                axs[i].annotate(coord_text, (birth, death), xytext=(3, -8), textcoords='offset points', fontsize=17)
                img_rgb[coordinates[0][0], coordinates[0][1]] = [0, 255, 0]
            elif i == 1 and j == distance_list.__len__() - 1:
                axs[i].scatter(birth, death, c="yellow", s=50)
                coord_text = f"({coordinates[0][0]}, {coordinates[0][1]})"
                axs[i].annotate(coord_text, (birth, death), xytext=(4, -8), textcoords='offset points', fontsize=17)
                img_rgb[coordinates[0][0], coordinates[0][1]] = [255, 255, 0]
        img_rgb_list.append(img_rgb)
    plt.savefig("img/save_image_with_persistence.png")

    plt.clf()
    concat_img = np.concatenate(img_rgb_list, axis=1)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(concat_img)
    plt.savefig("img/save_image.png")
