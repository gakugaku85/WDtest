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
import imageio
from PIL import ImageDraw
from PIL import Image


def make_mp4(image, fname, dir_path):
    print(dir_path, fname)

    output_dir = dir_path + "/mp4/"
    os.makedirs(output_dir, exist_ok=True)

    frames = []
    for threshold in range(256):
        image[image <= threshold] = 0
        frames.append(Image.fromarray(image))
    # 各フレームに閾値をテキストとして追加
    frames_with_text = []

    for i, frame in reversed(list(enumerate(frames))):
        img_copy = frame.copy()
        draw = ImageDraw.Draw(img_copy)
        text = f"{255-i}"
        draw.text((3, 3), text, fill=255)
        frames_with_text.append(img_copy)

    # 画像リストをMP4に変換
    mp4_with_text_output_path = os.path.join(
        output_dir, fname.split(".")[0] + ".mp4"
    )
    with imageio.get_writer(mp4_with_text_output_path, mode="I", fps=10) as writer:
        for frame in frames_with_text:
            writer.append_data(np.array(frame))

def create_persistent_diagram(ori_image_data, fname, dir_path):

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
                    create_persistent_diagram(original_image, fname, dir_path)
                    make_mp4(original_image, fname, dir_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to result folder")
    parser.add_argument("-s", "--sigma", type=int, default=64 ,help="sigma value")
    args = parser.parse_args()
    main(args.path, args.sigma)