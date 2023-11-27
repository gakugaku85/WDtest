import os
import SimpleITK as sitk
import numpy as np

def normalize_image(image_path):
    # MHDファイルを読み込む
    image = sitk.ReadImage(image_path)

    # SimpleITKのImageをNumPyの配列に変換
    image_array = sitk.GetArrayFromImage(image)

    # 上下99.95パーセンタイルを計算
    upper_percentile = np.percentile(image_array, 99.95)
    lower_percentile = np.percentile(image_array, 0.05)

    # 上下99.95パーセンタイルを使って正規化
    normalized_image_array = np.clip((image_array - lower_percentile) / (upper_percentile - lower_percentile), 0, 1)

    # [0, 255]の範囲にスケーリング
    normalized_image_array = (normalized_image_array * 255).astype(np.uint8)

    # 正規化された配列をSimpleITKのImageに変換
    normalized_image = sitk.GetImageFromArray(normalized_image_array)

    # 元の画像と同じメタデータを保持
    normalized_image.SetOrigin(image.GetOrigin())
    normalized_image.SetSpacing(image.GetSpacing())
    normalized_image.SetDirection(image.GetDirection())

    return normalized_image

def process_folder(input_folder, output_folder):
    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 入力フォルダ内のすべてのMHDファイルに対して処理
    for filename in os.listdir(input_folder):
        if filename.endswith(".mhd"):
            input_path = os.path.join(input_folder, filename)

            # 正規化を実行
            normalized_image = normalize_image(input_path)

            # 出力ファイルのパスを構築
            output_path = os.path.join(output_folder, filename.replace(".mhd", "_normalized.mhd"))

            # 正規化された画像を保存
            sitk.WriteImage(normalized_image, output_path)

# 使用例: フォルダのパスを指定
input_folder = "images/sigma32/sr_16_64_mhd"
output_folder = "images/sigma32/sr_16_64_mhd2"

input_folder = "images/sigma32/hr_64_mhd"
output_folder = "images/sigma32/hr_64_mhd2"

process_folder(input_folder, output_folder)
