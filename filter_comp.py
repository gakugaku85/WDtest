import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import frangi, sato, hessian, meijering
from skimage.color import rgb2gray
import SimpleITK as sitk
from soft_frangi.soft_frangi_filter2d import SoftFrangiFilter2D
import torch

# サンプル画像の読み込み
image = sitk.GetArrayFromImage(sitk.ReadImage("200.mhd"))

# ランダムな場所から64*64に切り取る
x = np.random.randint(0, image.shape[0] - 64)
y = np.random.randint(0, image.shape[1] - 64)

org_image = image[x:x+64, y:y+64]

# org_image = sitk.GetArrayFromImage(sitk.ReadImage("1.mhd"))
org_image = (org_image - org_image.min()) / (org_image.max() - org_image.min())

print(org_image.max(), org_image.min())
# 反転
image = 1 - org_image

# Frangiフィルターの適用
frangi_result = frangi(image)

# Satoフィルターの適用
sato_result = sato(image)

hessian_result = hessian(org_image)
meijering_result = meijering(image)

# ライブラリのSoft Frangiフィルターの適用
tensor_image = torch.tensor(org_image).unsqueeze(0).unsqueeze(0).float()
soft_frangi_filter = SoftFrangiFilter2D(1, 7, [2,4,8], 0.5, 0.5, 'cpu')
soft_frangi_response = soft_frangi_filter(tensor_image)
np_image = soft_frangi_response.squeeze().detach().numpy()

# 結果の表示
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(org_image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(frangi_result, cmap='gray')
axes[0, 1].set_title('Frangi Filter')
axes[0, 1].axis('off')

axes[0, 2].imshow(sato_result, cmap='gray')
axes[0, 2].set_title('Sato Filter')
axes[0, 2].axis('off')

axes[1, 0].imshow(hessian_result, cmap='gray')
axes[1, 0].set_title('Hessian Filter')
axes[1, 0].axis('off')

axes[1, 1].imshow(meijering_result, cmap='gray')
axes[1, 1].set_title('Meijering Filter')
axes[1, 1].axis('off')

axes[1, 2].imshow(np_image, cmap='gray')
axes[1, 2].set_title('Soft Frangi Filter')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('filter_comp.png')
