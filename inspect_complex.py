import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
# import SimpleITK as sitk
import torch
import cv2
from torch import nn
import SimpleITK as sitk
import time
from torch_topological.nn import CubicalComplex, WassersteinDistance

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def plot_persistence_diagram_with_coordinates(ax, cofaces_output):
    # 点をプロットし、座標情報を追加
    for dim, (birth, death), coordinates in cofaces_output:
        if dim == 0:
            color = 'red'
        else:
            color = 'blue'

        # deathがinfの場合は1に設定
        death = 1 if np.isinf(death) else death

        ax.scatter(birth, death, c=color, s=10)

        # 座標情報のテキストを作成
        if coordinates[1] is None:
            coord_text = f"({coordinates[0][0]}, {coordinates[0][1]})"
        else:
            coord_text = f"({coordinates[0][0]}, {coordinates[0][1]}) -> ({coordinates[1][0]}, {coordinates[1][1]})"

        # 点の横にテキストを追加
        ax.annotate(coord_text, (birth, death), xytext=(5, 5), textcoords='offset points', fontsize=8)

    # 対角線を描画
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.3, zorder=0)

    ax.set_title('Persistence Diagram with Coordinates')
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
# image = np.array([
#     [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
#     [0.3, 0.7, 0.7, 0.5, 0.5, 0.6, 0.6, 0.3],
#     [0.3, 0.7, 0.5, 0.5, 0.5, 0.6, 0.6, 0.3],
#     [0.3, 0.7, 0.7, 0.5, 0.5, 0.6, 0.6, 0.3],
#     [0.3, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.3],
#     [0.3, 0.9, 0.9, 0.7, 0.7, 0.6, 0.6, 0.3],
#     [0.3, 0.9, 0.7, 0.7, 0.7, 0.6, 0.6, 0.3],
#     [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
# ])

# image_data = np.array(
#     [
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 1, 1, 1, 1, 0.8, 0],
#         [0, 0.4, 0, 0, 0, 0, 0.4, 0],
#         [0, 1, 0, 0, 0, 0, 1, 0],
#         [0, 1, 0, 0, 0, 0, 1, 0],
#         [0, 1, 0, 0, 0, 0, 1, 0],
#         [0, 1, 1, 1, 1, 1, 1, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#     ]
# )

# image_data_2 = np.array(
#     [
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 1, 0.1, 0.1, 0.2, 0.2, 0],
#         [0, 1, 1, 0.1, 0.1, 0.2, 0.2, 0],
#         [0, 0.3, 0.3, 0, 0, 1, 1, 0],
#         [0, 0.3, 0.3, 0, 0, 1, 1, 0],
#         [0, 0.8, 0.8, 0.5, 0.5, 0.6, 0.6, 0],
#         [0, 0.8, 0.8, 0.5, 0.5, 0.6, 0.6, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#     ]
# )

# image_data_3 = np.array(
#     [
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 1, 1, 1, 1, 0, 0],
#         [0, 1, 0, 0, 0, 0, 1, 0],
#         [0, 0.8, 0.3, 0.5, 0.5, 0, 1, 0],
#         [0, 1, 0, 0.5, 0.5, 0, 1, 0],
#         [0, 1, 0, 0, 0, 0, 1, 0],
#         [0, 1, 1, 1, 1, 1, 1, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#     ]
# )

image_list = []



# image = sitk.GetArrayFromImage(sitk.ReadImage("200.mhd"))

# for i in range(2):
#     # ランダムな場所から64*64に切り取る
#     x = np.random.randint(0, image.shape[0] - 64)
#     y = np.random.randint(0, image.shape[1] - 64)

#     while True:
#         x = np.random.randint(0, image.shape[0] - 64)
#         y = np.random.randint(0, image.shape[1] - 64)
#         org_image = image[x:x+64, y:y+64]
#         if (np.count_nonzero(org_image==0)/np.count_nonzero(org_image>=0) <= 0.6):#黒の割合
#             break

#     org_image = (org_image - org_image.min()) / (org_image.max() - org_image.min())
#     image_list.append(org_image)

image = sitk.GetArrayFromImage(sitk.ReadImage("img/1.mhd"))
image = (image - image.min()) / (image.max() - image.min())
image_list.append(image)
image = sitk.GetArrayFromImage(sitk.ReadImage("img/40.mhd"))
image = (image - image.min()) / (image.max() - image.min())
image_list.append(image)
image = sitk.GetArrayFromImage(sitk.ReadImage("img/100.mhd"))
image = (image - image.min()) / (image.max() - image.min())
image_list.append(image)

# image_list = [image_data, image_data_2, image_data_3]

# image_list = [image_data]

ori_image_list = []
ori1_image = sitk.GetArrayFromImage(sitk.ReadImage("img/1_ori.mhd"))
ori40_image = sitk.GetArrayFromImage(sitk.ReadImage("img/40_ori.mhd"))
ori100_image = sitk.GetArrayFromImage(sitk.ReadImage("img/100_ori.mhd"))
ori_image_list.append(ori1_image)
ori_image_list.append(ori40_image)
ori_image_list.append(ori100_image)

# from soft_frangi.soft_frangi_filter2d import SoftFrangiFilter2D
# soft_frangi_filter = SoftFrangiFilter2D(channels=1, kernel_size=7, sigmas=range(1, 10, 2), beta=0.5, c=0.5, device='cpu')
from skimage.filters import frangi, sato


plt.clf()
fig, axs = plt.subplots(3, len(image_list), figsize=(7*len(image_list), 10))

for i, (img, ori_img) in enumerate(zip(image_list, ori_image_list)):
    # 画像の表示
    axs[0, i].imshow(img, cmap='gray')
    axs[0, i].axis('off')
    axs[0, i].set_title("Image " + str(i))

    # axs[2, i].imshow(ori_img, cmap='gray')
    # axs[2, i].axis('off')
    # axs[2, i].set_title("Ori Image " + str(i))

    # ori_imgの192の画素の数をカウント
    # print(f"ori_imgの192の画素の数: {np.count_nonzero(ori_img==192)}")

    # Persistent Homologyの計算
    cc = gd.CubicalComplex(
        dimensions=img.shape, top_dimensional_cells=1.0-img.flatten()
    )
    persistence = cc.persistence()
    cofaces = cc.cofaces_of_persistence_pairs()

    result = match_cofaces_with_gudhi(img, cofaces, persistence)

    # print(f"persistence for Image {i}:", persistence)
    # print(f"cofaces for Image {i}:", cofaces)
    # print(f"result for Image {i}:", result)

    # Persistence Diagramの表示
    ax = axs[1, i]
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    bx = axs[0, i]

    cx = axs[2, i]

    filtered_diagram = [(dim, (birth, 1 if np.isinf(death) else death)) for dim, (birth, death), _ in result]

    frangi_img = frangi(1-img)
    print("frangi max min",frangi_img.max(), frangi_img.min())
    cx.imshow(frangi_img, cmap='gray')

    img_scaled = (img * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2RGB)
    point_num = 0
    # 点をプロットし、座標情報を追加
    for (dim, (birth, death)), (_, _, coordinates) in zip(filtered_diagram, result):
        if dim == 0:
            color = 'red'
        else:
            color = 'blue'
            continue

        # カラーマップを適用
        # ori image の(coordinates[0][0], coordinates[0][1])の座標の画素値が192の場合点を描画
        if frangi_img[coordinates[0][0], coordinates[0][1]] > 0.45:
            if coordinates[1] is not None:
                    img_rgb[coordinates[0][0], coordinates[0][1]] = [255, 0, 0]
                    # img_rgb[coordinates[1][0], coordinates[1][1]] = [0, 0, 255]
                    point_num += 1
            ax.scatter(birth, death, c=color, s=10)
        else:
            continue
        # ax.scatter(birth, death, c=color, s=10)

        # 座標情報のテキストを作成
        if coordinates[1] is None:
            coord_text = f"({coordinates[0][0]}, {coordinates[0][1]})"
        else:
            coord_text = f"({coordinates[0][0]}, {coordinates[0][1]}) -> ({coordinates[1][0]}, {coordinates[1][1]})"

        # 点の横にテキストを追加
        ax.annotate(coord_text, (birth, death), xytext=(5, 5), textcoords='offset points', fontsize=12)

    # 対角線を描画
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.3, zorder=0)

    print(f"点の数: {point_num}")

    ax.set_title(f'Persistence Diagram {i}')
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')

    bx.imshow(img_rgb)

plt.tight_layout()
plt.savefig("images_with_diagrams_and_coordinates.png")

# for i, img in enumerate(image_list):
#     # gudhiでの表示を行う
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#     cc = gd.CubicalComplex(
#         dimensions=img.shape, top_dimensional_cells=1.0-img.flatten()
#     )
#     persistence = cc.persistence()

#     gd.plot_persistence_diagram(persistence)
#     plt.title(f"Image {i} Persistence Diagram")
#     plt.savefig(f"image_{i}_persistence_diagram.png")

# print("gudhi:", persistences)

# cofaces = cc.cofaces_of_persistence_pairs()

# for idx, (birth, death) in enumerate(persistence):
#     if death[1] == float("inf"):
#         persistence[idx] = (birth, (death[0], image_data.max()))

# print("gudhi:", persistence)
# print("cofaces:", cofaces)

combined_images = np.array(image_list)
img_batch = torch.tensor(combined_images, device=device, dtype=torch.float32).unsqueeze(1)
cubical = CubicalComplex()
wd_loss = WassersteinDistance(q=2)
per_cc = cubical(img_batch)
# print(per_cc)

# loss = wd_loss(per_cc, per_cc)