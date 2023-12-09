import numpy as np
from skimage import measure
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import argparse
import cv2
from natsort import natsorted


def binary_threshold(image, threshold):
    binary_image = (image > threshold).astype(np.uint8)
    return binary_image

def create_connected_components_image(image, original_image, fname, dir_path):
    os.makedirs(dir_path + "/images/", exist_ok=True)
    cv2.imwrite(dir_path + "/images/" + fname.split(".")[0] + "_original" + ".png", original_image * 255)
    cv2.imwrite(dir_path + "/images/" + fname.split(".")[0] + "_binary" + ".png", image * 255)

    labeled_image = measure.label(image, connectivity=2)
    mix_labeled_image = labeled_image * original_image
    cv2.imwrite(dir_path + "/images/" + fname.split(".")[0] + "_labeled" + ".png", labeled_image * 255)

    # 残ったラベルのみを抽出
    labels = np.unique(mix_labeled_image)[1:]
    num_components = len(labels)
    label_image = np.zeros_like(original_image)
    for label in labels:
        label_image[labeled_image == label] = 1
    cv2.imwrite(dir_path + "/images/" + fname.split(".")[0] + "_label" + ".png", label_image * 255)
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

    avg_val1_components = []
    avg_val2_components = []

    for dir_path, _, fnames in natsorted(os.walk(result_path)):
        if dir_path.split("/")[-1] == "val1":
            i = 0
            num_components = []
            for fname in natsorted(fnames):
                if fname.endswith(".mhd"):
                    mhd_file_path = os.path.join(dir_path, fname)
                    print("mhd_file_path: ", mhd_file_path)
                    image = sitk.GetArrayFromImage(sitk.ReadImage(mhd_file_path))
                    image = image[:, 64:128]
                    binary_image = binary_threshold(image, threshold_value)
                    num_components.append(create_connected_components_image(binary_image, binary_threshold(original_val1_images[i], threshold_value), fname, dir_path))
                    i += 1
            avg_val1_components.append(np.mean(num_components))
            print(f"val1平均連結成分数: {np.mean(num_components)}")

        elif dir_path.split("/")[-1] == "val2":
            i = 0
            num_components = []
            for fname in natsorted(fnames):
                if fname.endswith(".mhd"):
                    mhd_file_path = os.path.join(dir_path, fname)
                    print("mhd_file_path: ", mhd_file_path)
                    image = sitk.GetArrayFromImage(sitk.ReadImage(mhd_file_path))
                    image = image[:, 64:128]
                    binary_image = binary_threshold(image, threshold_value)
                    num_components.append(create_connected_components_image(binary_image, binary_threshold(original_val2_images[i], threshold_value), fname, dir_path))
                    i += 1
            avg_val2_components.append(np.mean(num_components))
            print(f"val2平均連結成分数: {np.mean(num_components)}")
    plt.plot(avg_val1_components, label="val1")
    plt.plot(avg_val2_components, label="val2")
    plt.title("sigma={} TPCC".format(sigma))
    plt.legend()
    plt.xlabel("validation num")
    plt.ylabel("number of mean connected components")
    plt.savefig(result_path + "val_cc.png")
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to result folder")
    parser.add_argument("-s", "--sigma", type=int, default=64 ,help="sigma value")
    args = parser.parse_args()
    main(args.path, args.sigma)
