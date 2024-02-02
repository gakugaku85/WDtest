import argparse
import os

import cv2
import gudhi as gd
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from natsort import natsorted
from PIL import Image, ImageDraw
from skimage import measure
from skimage.color import label2rgb


def make_mp4(image, fname, dir_path):
    print(dir_path, fname)

    output_dir = dir_path + "/mp4/"
    os.makedirs(output_dir, exist_ok=True)

    frames = []

    for threshold in range(256):
        image[image <= threshold] = 0
        frames.append(Image.fromarray(image))
    frames_with_text = []

    for i, frame in reversed(list(enumerate(frames))):
        img_copy = frame.copy()
        draw = ImageDraw.Draw(img_copy)
        text = f"{255-i}"
        draw.text((3, 3), text, fill=255)
        frames_with_text.append(img_copy)

    mp4_with_text_output_path = os.path.join(
        output_dir, fname.split(".")[0] + ".mp4"
    )
    with imageio.get_writer(mp4_with_text_output_path, mode="I", fps=10) as writer:
        for frame in frames_with_text:
            writer.append_data(np.array(frame))

def binary_threshold(image, threshold):
    binary_image = (image > threshold).astype(np.uint8)
    return binary_image

def save_persistent_diagram(persistence, output_dir, output_file_name):
    plt.clf()
    gd.plot_persistence_diagram(persistence)
    plt.title("PD" + output_file_name)
    plt.xlim(-3, 260)
    plt.ylim(0, 260)
    output_file_path = output_dir + output_file_name + ".png"
    plt.savefig(output_file_path)
    plt.close()
    return output_file_path

def homologize_persistence(image_data, persistence, output_dir, output_file_name, each_save=False, image_name="HR"):
    os.makedirs(output_dir+"/point/", exist_ok=True)
    for i, (betch, (birth, death)) in enumerate(persistence):
        if betch == 0:#連結成分のみ
            if death - birth < 200:
                continue

            save_persistent_diagram([(betch, (birth, death))], output_dir+"/point/", output_file_name+str(i)+"_pd_"+image_name)
            threshold = (death + birth) // 2
            thre_bin_image = binary_threshold(image_data, threshold)

            #deathの半分で二値化したものを元に、ラベリング
            label = measure.label(thre_bin_image, connectivity=2)

            #ラベリングしたものを元に、ラベルごとに色を付ける
            image_label_overlay = label2rgb(label, bg_label=0)

            #deathの直前の閾値で二値化
            threshold = death - birth - 1
            print(image_data.max())
            death_before_image = binary_threshold(image_data, threshold)
            death_image = binary_threshold(image_data, death)

            sub_image = death_before_image - death_image

            #deathの直前の閾値で二値化した画像の255の部分を、ラベリングした画像のラベルに置き換える
            mix_labeled_image = label * death_before_image
            labels = np.unique(mix_labeled_image)[1:]
            label_image = np.zeros_like(thre_bin_image)
            for lab in labels:
                label_image[label == lab] = lab
            print(image_name, "label:", len(labels), "death:", death, "birth:", birth, output_file_name+str(i)+"_pd_"+image_name)
            colored_labels = label2rgb(label_image, bg_label=0)

            #最も面積の大きいラベルのみを抽出
            # labels = np.unique(label_image)[1:]
            # most_large_label = labels.count_max()
            # print("most_large_label:", most_large_label)

            # label_image_max = np.zeros_like(thre_bin_image)
            # for lab in labels:
            #     label_image_max[label == lab] = lab
            # colored_labels_max = label2rgb(label_image_max, bg_label=0)

            #画像にタイトルを付ける
            if each_save:
                cv2.imwrite(output_dir+"/point/"+output_file_name+str(i)+"_point_color.png", image_label_overlay*255)
                cv2.imwrite(output_dir+"/point/"+output_file_name+str(i)+"_point.png", thre_bin_image*255)
                cv2.imwrite(output_dir+"/point/"+output_file_name+str(i)+"_death.png", death_image*255)
                cv2.imwrite(output_dir+"/point/"+output_file_name+str(i)+"_death_before.png", death_before_image*255)
                cv2.imwrite(output_dir+"/point/"+output_file_name+str(i)+"_extract_color.png", colored_labels*255)
                cv2.imwrite(output_dir+"/point/"+output_file_name+str(i)+"_extract.png", label_image*255)
                cv2.imwrite(output_dir+"/point/"+output_file_name+str(i)+"_death_sub.png", sub_image*255)

            #白黒画像を拡張
            thre_bin_image = cv2.cvtColor(thre_bin_image, cv2.COLOR_GRAY2BGR)
            death_before_image = cv2.cvtColor(death_before_image, cv2.COLOR_GRAY2BGR)
            image_label_overlay = image_label_overlay.astype(np.uint8)
            colored_labels = colored_labels.astype(np.uint8)
            # colored_labels_max = colored_labels_max.astype(np.uint8)

            #画像を横に並べる
            concat_image = cv2.hconcat([thre_bin_image*255, death_before_image*255, image_label_overlay*255, colored_labels*255])
            cv2.imwrite(output_dir+"/point/"+output_file_name+str(i)+"_concat_"+image_name+".png", concat_image)


def create_persistent_diagram(ori_image_data, fname, dir_path):

    image_list = ["LR", "SR", "HR"]
    output_dir = dir_path + "/pd"
    os.makedirs(output_dir, exist_ok=True)
    ori_image_data = ori_image_data.astype(np.uint8)

    output_file_name = fname[:-4] + "_pd"
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

        homologize_persistence(image_data, persistence, output_dir, output_file_name, image_name=image_name)

        output_file_path = save_persistent_diagram(persistence, output_dir, output_file_name + "_" + image_name)
        image_paths.append(output_file_path)

    images = [cv2.imread(path) for path in image_paths]
    concatenated_image = cv2.hconcat(images)
    concatenated_image_path = output_dir + output_file_name + "_concatenated.png"
    cv2.imwrite(concatenated_image_path, concatenated_image)


def main(path):
    result_path = "../SR3/SR3_wdTest/experiments/"
    result_path = result_path + path
    result_path = result_path + "/results/"
    assert os.path.isdir(result_path), '{:s} is not a valid directory'.format(result_path)

    for dir_path, _, fnames in natsorted(os.walk(result_path)):
        if (dir_path.split("/")[-1] == "val1" or dir_path.split("/")[-1] == "val2") and dir_path.split("/")[-2] == "10000":
            for fname in natsorted(fnames):
                if fname.endswith(".mhd"):
                    mhd_file_path = os.path.join(dir_path, fname)
                    original_image = sitk.GetArrayFromImage(sitk.ReadImage(mhd_file_path))
                    create_persistent_diagram(original_image, fname, dir_path)
                    # make_mp4(original_image, fname, dir_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to result folder", default="wdtest_16_64_231209_095537")
    args = parser.parse_args()
    main(args.path)