import numpy as np
import SimpleITK as sitk
from skimage.filters import frangi
from skimage.measure import label, regionprops
from icecream import ic
from skimage import morphology
import os
import os.path as osp
from glob import glob
from natsort import natsorted

# MHDファイルを読み込む関数
def load_mhd(file_path):
    image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(image)  # SimpleITKイメージをNumPy配列に変換
    return array, image

# フィルタ適用後、二値化しMHD形式で保存する関数
def apply_frangi_filter_and_save(input_path, output_path, threshold):
    # MHDファイルを読み込み
    array, image_info = load_mhd(input_path)

    ic(input_path ,array.max(), array.min())

    # arrayを0-255に正規化
    array = (array - array.min()) / (array.max() - array.min()) * 255
    ic(array.max(), array.min())

    # Frangiフィルターの適用
    filtered_array = frangi(array, black_ridges=False)

    # フィルター結果を0-255に変換
    normalized_filtered = (filtered_array * 255).astype(np.uint8)

    # SimpleITKで大津のしきい値を計算
    sitk_image = sitk.GetImageFromArray(normalized_filtered)
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_image = otsu_filter.Execute(sitk_image)
    threshold = otsu_filter.GetThreshold()  # しきい値を取得
    ic(threshold)

    # 大津のしきい値を用いて二値化
    binary_array = (normalized_filtered >= threshold).astype(np.uint8)

    # 面積画素数が100以下の成分を削除
    labeled_array = label(binary_array, connectivity=1)  # ラベリング
    for region in regionprops(labeled_array):
        if region.area <= 10:  # 面積が100以下の成分を削除
            labeled_array[labeled_array == region.label] = 0

    # 二値化画像をリセット（削除された領域を反映）
    binary_array = (labeled_array > 0).astype(np.uint8)

    # NumPy配列をSimpleITKイメージに変換
    binary_image = sitk.GetImageFromArray(binary_array)

    # オリジナルのメタデータ（位置情報や解像度など）を保持
    binary_image.CopyInformation(image_info)

    # MHDファイルとして保存
    sitk.WriteImage(binary_image, output_path)

threshold = "otsu"
# 入力と出力のファイルパス
# name_list = ["denoise_2dslices_1794233", "denoise_2dslices_1794312", "denoise_2dslices_1794464", "denoise_2dslices_1794561", "denoise_2dslices_1794700"]
# name_list = ["100", "200", "300", "400", "500", "600", "700", "800", "900", "1000"]

input_path = "/take/dataset/microCT_slices_1792_2/patches"

input_mhd_files = natsorted(glob(osp.join('{}/patch_hr*.mhd'.format(input_path))))

for input_mhd_file in input_mhd_files:
    output_path = '/take/dataset/microCT_slices_1792_2/patches_otsu_label/'
    os.makedirs(output_path, exist_ok=True)
    output_mhd_file = osp.join(output_path, osp.basename(input_mhd_file.replace('hr', 'label')))
    apply_frangi_filter_and_save(input_mhd_file, output_mhd_file, threshold)
