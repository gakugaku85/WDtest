import sys

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

# Dicomの読み込み
# imgdir = sys.argv[1]
# reader = sitk.ImageSeriesReader()

# dicom_names = reader.GetGDCMSeriesFileNames(imgdir)
# reader.SetFileNames(dicom_names)
# image = reader.Execute()  # ITK形式で読み込まれた画像

input_path = "/take/dataset/microCT_slices_1792_2/interlobular_septum_mask/Segmentation_gaku.nrrd"

# NIFTIやNRRDを読み込む場合
image = sitk.ReadImage(input_path)
# 画像のサイズ取得
size = image.GetSize()
print("Image size:", size[0], size[1], size[2]) #1948 1713 10

# ITK形式をndarray形式に変換
ndImage = sitk.GetArrayFromImage(image)
# 画像を表示
# nは任意のスライスについて表示

# 5スライス目を2Dでnrrdで保存
n = 5
slice = ndImage[n, :, :]
print(slice.shape)
slice = sitk.GetImageFromArray(slice)
sitk.WriteImage(slice, "/take/dataset/microCT_slices_1792_2/interlobular_septum_mask/600_label_gaku.nrrd".format(n))


