import SimpleITK as sitk
import numpy as np
from skimage.filters import frangi

# MHDファイルを読み込む関数
def load_mhd(file_path):
    image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(image)  # SimpleITKイメージをNumPy配列に変換
    return array, image

# フィルタ適用後、二値化しMHD形式で保存する関数
def apply_frangi_filter_and_save(input_path, output_path, threshold):
    # MHDファイルを読み込み
    array, image_info = load_mhd(input_path)

    # Frangiフィルターの適用
    filtered_array = frangi(256-array)

    # 二値化 (フィルター結果が0.5以上なら1、以下なら0)
    filtered_array = (filtered_array >= threshold).astype(np.uint8)

    # NumPy配列をSimpleITKイメージに変換
    binary_image = sitk.GetImageFromArray(filtered_array)

    # オリジナルのメタデータ（位置情報や解像度など）を保持
    binary_image.CopyInformation(image_info)

    # MHDファイルとして保存
    sitk.WriteImage(binary_image, output_path)

threshold = 0.03
# 入力と出力のファイルパス
# name_list = ["denoise_2dslices_1794233", "denoise_2dslices_1794312", "denoise_2dslices_1794464", "denoise_2dslices_1794561", "denoise_2dslices_1794700"]
name_list = ["100", "200", "300", "400", "500", "600", "700", "800", "900", "1000"]

for name in name_list:
    input_mhd_file = "/take/dataset/microCT_slices_1792_2/hr/"+name+".mhd"  # 読み込むMHDファイルのパス
    output_mhd_file = '/take/dataset/microCT_slices_1792_2/interlobular_septum_mask/'+name+'_thre_'+str(threshold)+'.mhd'  # 保存するMHDファイルのパス

    # 関数を実行
    apply_frangi_filter_and_save(input_mhd_file, output_mhd_file, threshold)

# 関数を実行
apply_frangi_filter_and_save(input_mhd_file, output_mhd_file, threshold)
