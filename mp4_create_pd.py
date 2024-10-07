import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import cv2
import os
import SimpleITK as sitk

def match_cofaces_with_gudhi(image_data, cofaces):
    result = []

    for dim, pairs in enumerate(cofaces[0]):
        for birth, death in pairs:
            birth_y, birth_x = np.unravel_index(birth, image_data.shape)
            death_y, death_x = np.unravel_index(death, image_data.shape)
            pers = (1.00-image_data.ravel()[birth], 1.00-image_data.ravel()[death])
            result.append((dim, pers,((birth_y, birth_x), (death_y, death_x))))

    for dim, births in enumerate(cofaces[1]):
        for birth in births:
            birth_y, birth_x = np.unravel_index(birth, image_data.shape)
            pers = (1.00-image_data.ravel()[birth], 1.0)
            result.append((dim, pers, ((birth_y, birth_x), None)))

    return result

# 64x64 Gaussian noise image
image_list = []
image = sitk.GetArrayFromImage(sitk.ReadImage("img/200.mhd"))

np.random.seed(11)
for i in range(1):
    # ランダムな場所から64*64に切り取る
    x = np.random.randint(0, image.shape[0] - 64)
    y = np.random.randint(0, image.shape[1] - 64)

    while True:
        x = np.random.randint(0, image.shape[0] - 64)
        y = np.random.randint(0, image.shape[1] - 64)
        org_image = image[x:x+64, y:y+64]
        if (np.count_nonzero(org_image==0)/np.count_nonzero(org_image>=0) <= 0.6):#黒の割合
            break

    org_image = (org_image - org_image.min()) / (org_image.max() - org_image.min())
    image_list.append(org_image)


image_size = (64, 64)
# mean = 0
# std_dev = 1
# gaussian_noise_image = np.random.normal(mean, std_dev, image_size)
# gaussian_noise_image = (gaussian_noise_image - gaussian_noise_image.min()) / (gaussian_noise_image.max() - gaussian_noise_image.min())

gaussian_noise_image = org_image

# Save the Gaussian noise image
plt.imsave('img/gaussian_noise_image.png', gaussian_noise_image, cmap='gray')

# Plot the Gaussian noise image
plt.imshow(gaussian_noise_image, cmap='gray')
plt.title('Gaussian Noise Image')
plt.colorbar()

# Create a Cubical Complex from the image
cubical_complex = gd.CubicalComplex(dimensions=image_size, top_dimensional_cells=1.0-gaussian_noise_image.flatten())
cubical_complex.persistence()
cofaces = cubical_complex.cofaces_of_persistence_pairs()
result = match_cofaces_with_gudhi(gaussian_noise_image, cofaces)

plt.clf()
fig, axs = plt.subplots(1, 1, figsize=(8, 6))

point_num = 0
for dim, (birth, death) , coordinates in result:
    # print(f"Image {i} Dimension {dim} Birth {birth} Death {death} Coordinates {coordinates}")
    if dim == 0:
        color = 'red'
    else:
        color = 'blue'

    distance = np.abs(birth - death) / np.sqrt(2)
    point_num += 1

    if birth < 0.08:
        if coordinates[1] is None:
            coord_text = f"({coordinates[0][0]}, {coordinates[0][1]})"
        else:
            coord_text = f"({coordinates[0][0]}, {coordinates[0][1]}), ({coordinates[1][0]}, {coordinates[1][1]})"
        axs.annotate(coord_text, (birth, death), xytext=(3, -8), textcoords='offset points', fontsize=12)

    axs.scatter(birth, death, c=color, s=20)
    lims = [-0.1,1.1]
    axs.plot(lims, lims, 'k-', alpha=0.3, zorder=0)
    axs.set_xlim(-0.1, 1.1)
    axs.set_ylim(-0.1, 1.1)
    axs.set_title(f'Persistence Diagram')
    axs.set_xlabel('Birth')
    axs.set_ylabel('Death')

print(f"点の数: {point_num}")
axs.annotate(f"point num: {point_num}", (0.7, 0.5), xytext=(3, 3), textcoords='offset points', fontsize=12)
plt.savefig("img/save_image_with_persistence.png")

# 出力フォルダの作成
output_dir = 'img/threshold_images'
os.makedirs(output_dir, exist_ok=True)

# 動画ファイルの設定
video_path = 'img/output_video.mp4'
frame_height, frame_width = gaussian_noise_image.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_path, fourcc, 10, (frame_width, frame_height), False)

threshold_step = 0.01
font = cv2.FONT_HERSHEY_SIMPLEX  # フォントの設定
font_scale = 0.2  # フォントサイズ
threshold_value = 1.0
color = (255)  # 文字の色（白）
thickness = 1  # 文字の太さ
position = (1, 5)
while threshold_value >= 0:
    # 二値化
    _, binary_image = cv2.threshold(gaussian_noise_image, threshold_value, 1, cv2.THRESH_BINARY)
    binary_image = (binary_image * 255).astype(np.uint8)  # 画像を0-255の範囲に戻す

    # 現在の閾値を表示
    threshold_value_dis = 1.00 - threshold_value
    text = f'{threshold_value_dis:.2f}'
    if threshold_value_dis > 0.6:
        color = (0)
    cv2.putText(binary_image, text, position, font, font_scale, color, thickness)

    # 画像の保存
    output_image_path = os.path.join(output_dir, f'threshold_{threshold_value:.2f}.png')
    cv2.imwrite(output_image_path, binary_image)

    # 動画にフレームを追加
    video.write(binary_image)

    # 次の閾値に進む
    threshold_value -= threshold_step

# 動画を保存
video.release()

print("動画作成が完了しました。")
