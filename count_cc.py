import numpy as np
from skimage import measure
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import argparse
import cv2
from natsort import natsorted
from skimage.color import label2rgb
import pandas as pd


def binary_threshold(image, threshold):
    binary_image = (image > threshold).astype(np.uint8)
    return binary_image

def create_connected_components_image(image, original_image, fname, dir_path):
    os.makedirs(dir_path + "/images/", exist_ok=True)
    cv2.imwrite(dir_path + "/images/" + fname.split(".")[0] + "_original" + ".png", original_image * 255)
    cv2.imwrite(dir_path + "/images/" + fname.split(".")[0] + "_binary" + ".png", image * 255)

    labeled_image = measure.label(image, connectivity=1)
    # output color labels
    colored_labels = label2rgb(labeled_image, bg_label=0)
    cv2.imwrite(dir_path + "/images/" + fname.split(".")[0] + "color_labeled" + ".png", colored_labels * 255)
    mix_labeled_image = labeled_image * original_image

    # 残ったラベルのみを抽出
    labels = np.unique(mix_labeled_image)[1:]
    num_components = len(labels)
    label_image = np.zeros_like(original_image)
    for label in labels:
        label_image[labeled_image == label] = label
    colored_labels = label2rgb(label_image, bg_label=0)
    cv2.imwrite(dir_path + "/images/" + fname.split(".")[0] + "color_label" + ".png", colored_labels * 255)
    cv2.imwrite(dir_path + "/images/" + fname.split(".")[0] + "res_label" + ".png", label_image * 255)
    return num_components

def main(path, sigma):
    threshold_value = 128

    original_val1_path = "images/original_val1/"
    assert os.path.isdir(original_val1_path), '{:s} is not a valid directory'.format(original_val1_path)
    original_val1_images = []
    for dir_path, _, fnames in natsorted(os.walk(original_val1_path)):
        for fname in natsorted(fnames):
            if fname.endswith(".mhd"):
                mhd_file_path = os.path.join(dir_path, fname)
                original_image = sitk.GetArrayFromImage(sitk.ReadImage(mhd_file_path))
                original_val1_images.append(original_image)

    original_val2_path = "images/original_val2/"
    assert os.path.isdir(original_val2_path), '{:s} is not a valid directory'.format(original_val2_path)
    original_val2_images = []
    for dir_path, _, fnames in natsorted(os.walk(original_val2_path)):
        for fname in natsorted(fnames):
            if fname.endswith(".mhd"):
                mhd_file_path = os.path.join(dir_path, fname)
                original_image = sitk.GetArrayFromImage(sitk.ReadImage(mhd_file_path))
                original_val2_images.append(original_image)

    result_path = "../SR3/SR3_wdTest/experiments/"
    result_path = result_path + path
    result_path = result_path + "/results/"
    assert os.path.isdir(result_path), '{:s} is not a valid directory'.format(result_path)

    df2 = pd.DataFrame(columns=["iteration_num", "val1_num_components", "val2_num_components"])

    for dir_path, _, fnames in natsorted(os.walk(result_path)):
        if dir_path.split("/")[-1] in ["val1", "val2"]:
            val1_num_components = []
            val2_num_components = []
            validation_type = dir_path.split("/")[-1]
            iteration_num = dir_path.split("/")[-2]
            i = 0
            for fname in natsorted(fnames):
                if fname.endswith(".mhd"):
                    mhd_file_path = os.path.join(dir_path, fname)
                    print("mhd_file_path: ", mhd_file_path)
                    image = sitk.GetArrayFromImage(sitk.ReadImage(mhd_file_path))
                    cv2.imwrite(dir_path + "/images/" + fname.split(".")[0] + "_image" + ".png", image)
                    image = image[:, 64:128]

                    if validation_type == "val1":
                        cc = create_connected_components_image(binary_threshold(image, threshold_value), binary_threshold(original_val1_images[i], threshold_value), fname, dir_path)
                        val1_num_components.append(cc)
                    elif validation_type == "val2":
                        cc = create_connected_components_image(binary_threshold(image, threshold_value), binary_threshold(original_val2_images[i], threshold_value), fname, dir_path)
                        val2_num_components.append(cc)
                    i += 1

            avg_cc_val1 = np.mean(val1_num_components)
            avg_cc_val2 = np.mean(val2_num_components)

            # validation_num = len(df2)
            # if validation_type == "val1":
            #     if validation_num >= len(df2):
            #         df2.loc[validation_num] = [validation_num, avg_cc_val1, avg_cc_val2]
            #     else:
            #         df2.loc[validation_num, "val1_num_components"] = avg_cc
            # elif validation_type == "val2":
            #     if validation_num >= len(df2):
            #         df2.loc[validation_num] = [validation_num, None, avg_cc]
            #     else:
            #         df2.loc[validation_num, "val2_num_components"] = avg_cc
            if validation_type == "val1":
                df2.loc[len(df2)] = [iteration_num, avg_cc_val1, None]
            if validation_type == "val2":
                df2.loc[len(df2)-1, "val2_num_components"] = avg_cc_val2

            # print(f"{validation_type}平均連結成分数: {avg_cc}")

    df2.to_csv(result_path + "val_cc_combined.csv", index=False)
    plt.plot(df2["val1_num_components"], label="val1")
    plt.plot(df2["val2_num_components"], label="val2")
    plt.title("sigma={} TPCC".format(sigma))
    plt.legend()
    plt.xlabel("validation num")
    plt.ylabel("TPCC num")
    plt.savefig(result_path + "val_cc_combined.png")
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to result folder")
    parser.add_argument("-s", "--sigma", type=int, default=8 ,help="sigma value")
    args = parser.parse_args()
    main(args.path, args.sigma)
