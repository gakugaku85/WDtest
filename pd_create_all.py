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
import gudhi as gd


def create_persistent_diagram(ori_image_data, fname, dir_path, sigma):

    image_list = ["LR", "SR", "HR"]
    # Create output directory
    output_dir = dir_path + "/pd/"
    os.makedirs(output_dir, exist_ok=True)

    # Create output file name
    output_file_name = fname[:-4] + "_pd"

    # Create a list to store the image paths
    image_paths = []

    for i, image_name in enumerate(image_list):
        image_data = ori_image_data[:, i*64:(i+1)*64]
        cc = gd.CubicalComplex(
            dimensions=image_data.shape, top_dimensional_cells=255 - image_data.flatten()
        )
        persistence = cc.persistence()

        for idx, (birth, death) in enumerate(persistence):
            if death[1] == float("inf"):
                persistence[idx] = (birth, (death[0], 255))

        plt.clf()
        gd.plot_persistence_diagram(persistence)
        plt.title("PD" + output_file_name + "_" + image_name)
        plt.xlim(-3, 260)
        plt.ylim(0, 260)
        # Create output file path
        output_file_path = output_dir + output_file_name + "_" + image_name + ".png"
        print(output_file_path)
        plt.savefig(output_file_path)

        #concat image
        image_paths.append(output_file_path)

    # Read images using OpenCV
    images = [cv2.imread(path) for path in image_paths]

    # Concatenate images horizontally using OpenCV
    concatenated_image = cv2.hconcat(images)

    # Save the concatenated image
    concatenated_image_path = output_dir + output_file_name + "_concatenated.png"
    cv2.imwrite(concatenated_image_path, concatenated_image)


def main(path, sigma):

    result_path = "../SR3/SR3_wdTest/experiments/"
    result_path = result_path + path
    result_path = result_path + "/results/"
    assert os.path.isdir(result_path), '{:s} is not a valid directory'.format(result_path)

    for dir_path, _, fnames in natsorted(os.walk(result_path)):
        if dir_path.split("/")[-1] == "val1" or dir_path.split("/")[-1] == "val2":
            for fname in natsorted(fnames):
                if fname.endswith(".mhd"):
                    mhd_file_path = os.path.join(dir_path, fname)
                    original_image = sitk.GetArrayFromImage(sitk.ReadImage(mhd_file_path))
                    create_persistent_diagram(original_image, fname, dir_path, sigma)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to result folder")
    parser.add_argument("-s", "--sigma", type=int, default=64 ,help="sigma value")
    args = parser.parse_args()
    main(args.path, args.sigma)