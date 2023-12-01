import os
import SimpleITK as sitk
import numpy as np

def normalize_image(image_path):
    image = sitk.ReadImage(image_path)

    # SimpleITKのImageをNumPyの配列に変換
    image_array = sitk.GetArrayFromImage(image)
    upper_percentile = np.percentile(image_array, 99.95)
    lower_percentile = np.percentile(image_array, 0.05)
    normalized_image_array = np.clip((image_array - lower_percentile) / (upper_percentile - lower_percentile), 0, 1)
    normalized_image_array = (normalized_image_array * 255).astype(np.uint8)
    normalized_image = sitk.GetImageFromArray(normalized_image_array)
    normalized_image.SetOrigin(image.GetOrigin())
    normalized_image.SetSpacing(image.GetSpacing())
    normalized_image.SetDirection(image.GetDirection())

    return normalized_image

def process_folder(input_folder, output_folder):
    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mhd"):
            input_path = os.path.join(input_folder, filename)
            normalized_image = normalize_image(input_path)
            output_path = os.path.join(output_folder, filename.replace(".mhd", "_normalized.mhd"))
            sitk.WriteImage(normalized_image, output_path)

# 使用例: フォルダのパスを指定
input_folder = "images/sigma32/sr_16_64_mhd"
output_folder = "images/sigma32/sr_16_64_mhd2"

input_folder = "images/sigma32/hr_64_mhd"
output_folder = "images/sigma32/hr_64_mhd2"

process_folder(input_folder, output_folder)
