import numpy as np
import SimpleITK as sitk
from skimage.filters import frangi
from skimage.morphology import disk
from skimage.measure import label, regionprops
from skimage.morphology import white_tophat
from icecream import ic
import os
from multiprocessing import Pool

def load_mhd(file_path):
    image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(image)  # SimpleITKイメージをNumPy配列に変換
    return array, image

def apply_frangi_filter_and_save(input_path, output_path, threshold):
    # MHDファイルを読み込み
    array, image_info = load_mhd(input_path)

    input_name = input_path.split("/")[-1].split(".")[0]

    # arrayを0-255に正規化
    array = (array - array.min()) / (array.max() - array.min()) * 255
    array = array.astype(np.uint8)  # ホワイトトップハット変換用に型変換
    ic(array.max(), array.min())

    selem_small = disk(4)
    tophat_array_small = white_tophat(array, selem_small)
    ic(input_name ,tophat_array_small.max(), tophat_array_small.min())

    # 半径14のホワイトトップハット変換
    selem_large = disk(14)
    tophat_array_large = white_tophat(array, selem_large)
    ic(input_name ,tophat_array_large.max(), tophat_array_large.min())

    # 2つの結果のORを取る
    combined_tophat = np.maximum(tophat_array_small, tophat_array_large)

    # Frangiフィルターの適用
    filtered_array = frangi(combined_tophat, black_ridges=False)

    # 二値化 (フィルター結果がしきい値以上なら1、以下なら0)
    binary_array = (filtered_array >= threshold).astype(np.uint8)

    # 小さい領域の削除 (面積が10以下の成分を削除)
    labeled_array = label(binary_array, connectivity=1)  # ラベリング
    for region in regionprops(labeled_array):
        if region.area <= 100:  # 面積が10以下の成分を削除
            labeled_array[labeled_array == region.label] = 0

    # 二値化画像をリセット（削除された領域を反映）
    binary_array = (labeled_array > 0).astype(np.uint8)

    # NumPy配列をSimpleITKイメージに変換
    binary_image = sitk.GetImageFromArray(binary_array)

    # オリジナルのメタデータ（位置情報や解像度など）を保持
    binary_image.CopyInformation(image_info)

    # MHDファイルとして保存
    sitk.WriteImage(binary_image, output_path)

def process_file(name):
    threshold = 0.03

    # 入力と出力のファイルパス
    input_mhd_file = f"/take/dataset/microCT_slices_1792_2/hr/{name}.mhd"  # 読み込むMHDファイルのパス
    output_path = f"/take/dataset/microCT_slices_1792_2/interlobular_septum_mask/tophat_5/"  # 保存するMHDファイルのディレクトリ
    os.makedirs(output_path, exist_ok=True)
    output_mhd_file = f"{output_path}{name}_{threshold}.mhd"  # 保存するMHDファイルのパス

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