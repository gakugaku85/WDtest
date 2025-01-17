import SimpleITK as sitk
import numpy as np
from skimage.filters import frangi
from skimage.measure import label, regionprops
import os
from icecream import ic
from multiprocessing import Pool

# MHDファイルを読み込む関数
def load_mhd(file_path):
    image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(image)  # SimpleITKイメージをNumPy配列に変換
    return array, image

# フィルタ適用後、二値化しMHD形式で保存する関数
def apply_frangi_filter_and_save(input_path, output_path, threshold):
    # MHDファイルを読み込み
    array, image_info = load_mhd(input_path)

    # arrayを0-255に正規化
    array = (array - array.min()) / (array.max() - array.min()) * 255
    ic(array.max(), array.min())

    # Frangiフィルターの適用
    filtered_array = frangi(array, black_ridges=False)


    # 二値化 (フィルター結果が0.5以上なら1、以下なら0)
    filtered_array = (filtered_array >= threshold).astype(np.uint8)

    labeled_array = label(filtered_array, connectivity=2)  # ラベリング
    for region in regionprops(labeled_array):
        if region.area <= 50:  # 面積が100以下の成分を削除
            labeled_array[labeled_array == region.label] = 0

    # ラベル付けされた画像をリセット（削除された領域を反映）
    labeled_array = (labeled_array > 0).astype(np.uint8)

    # NumPy配列をSimpleITKイメージに変換
    binary_image = sitk.GetImageFromArray(labeled_array)

    # オリジナルのメタデータ（位置情報や解像度など）を保持
    binary_image.CopyInformation(image_info)

    # MHDファイルとして保存
    sitk.WriteImage(binary_image, output_path)

def process_file(name):
    threshold = 0.03

    # 入力と出力のファイルパス
    input_mhd_file = f"/take/dataset/microCT_slices_1792_2/hr/{name}.mhd"  # 読み込むMHDファイルのパス
    output_path = f"/take/dataset/microCT_slices_1792_2/interlobular_septum_mask/{str(threshold)}/"  # 保存するMHDファイルのディレクトリ
    os.makedirs(output_path, exist_ok=True)
    output_mhd_file = f"{output_path}{name}_{str(threshold)}.mhd"  # 保存するMHDファイルのパス

    # フィルター適用関数を呼び出し
    apply_frangi_filter_and_save(input_mhd_file, output_mhd_file, threshold)

# メインスクリプト
if __name__ == "__main__":
    # 並列処理対象の名前リスト
    name_list = ["100", "200", "300", "400", "500", "600", "700", "800", "900", "1000"]

    # 並列処理用プールを作成 (CPUコア数を自動的に検出)
    with Pool() as pool:
        # 並列実行
        pool.map(process_file, name_list)
