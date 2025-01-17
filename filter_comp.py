import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import frangi, sato, hessian, meijering
from skimage.color import rgb2gray
import SimpleITK as sitk
import torch
import gudhi as gd

# サンプル画像の読み込み
image = sitk.GetArrayFromImage(sitk.ReadImage("40.mhd"))

# ランダムな場所から64*64に切り取る
# x = np.random.randint(0, image.shape[0] - 64)
# y = np.random.randint(0, image.shape[1] - 64)

# while True:
#     x = np.random.randint(0, image.shape[0] - 64)
#     y = np.random.randint(0, image.shape[1] - 64)
#     org_image = image[x:x+64, y:y+64]
#     if (np.count_nonzero(org_image==0)/np.count_nonzero(org_image>=0) <= 0.6):#黒の割合
        # break

# org_image = (org_image - org_image.min()) / (org_image.max() - org_image.min())
org_image = (image - image.min()) / (image.max() - image.min())
print(org_image.max(), org_image.min())
# 反転
image = 1 - org_image

# Frangiフィルターの適用
frangi_result = frangi(image)
print("frangi_result", frangi_result.max(), frangi_result.min())

# Satoフィルターの適用
sato_result = sato(image)

hessian_result = hessian(image)
meijering_result = meijering(image)

# パーシステントダイアグラムの生成
def compute_persistence_diagram(image):
    cubical_complex = gd.CubicalComplex(dimensions=image.shape, top_dimensional_cells=image.flatten())
    persistence = cubical_complex.persistence()
    return persistence

# 各フィルターのパーシステントダイアグラムを計算
diagrams = {
    "Original Image": compute_persistence_diagram(org_image),
    "Frangi Filter": compute_persistence_diagram(frangi_result),
    "Sato Filter": compute_persistence_diagram(sato_result),
    "Hessian Filter": compute_persistence_diagram(hessian_result),
    "Meijering Filter": compute_persistence_diagram(meijering_result),
    "Soft Frangi Filter": compute_persistence_diagram(np_image)
}

# 結果の表示
fig, axes = plt.subplots(2, 3)

# 画像の表示
images = [org_image, frangi_result, sato_result, hessian_result, meijering_result, np_image]
titles = ['Original Image', 'Frangi Filter', 'Sato Filter', 'Hessian Filter', 'Meijering Filter', 'Soft Frangi Filter']

for i, (img, title) in enumerate(zip(images, titles)):
    ax = axes[i // 3, i % 3]
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

# # パーシステントダイアグラムの表示
# for i, (title, diagram) in enumerate(diagrams.items()):
#     ax = axes[i // 3 + 2, i % 3]

#     # フィルタリングして無限大の値を除外
#     filtered_diagram = [(dim, (birth, death)) for dim, (birth, death) in diagram if not np.isinf(death)]

#     # 点をプロット
#     for dim, (birth, death) in filtered_diagram:
#         if dim == 0:
#             ax.scatter(birth, death, c='red', s=10)
#         else:
#             ax.scatter(birth, death, c='blue', s=10)

#     # 対角線を描画
#     lims = [
#         np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#         np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
#     ]
#     ax.plot(lims, lims, 'k-', alpha=0.3, zorder=0)

#     ax.set_title(f'{title} Persistence Diagram')
#     ax.set_xlabel('Birth')
#     ax.set_ylabel('Death')

plt.tight_layout()
plt.savefig('filter_comp_with_persistence.png')