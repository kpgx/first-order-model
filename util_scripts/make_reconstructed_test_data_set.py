import os
import cv2
import imageio
import numpy as np
from imageio import mimread
from skimage import io, img_as_float32


# reconstruct_files = "/Users/larcuser/Data/bair-eval-for-object-detection/reconstructed/"
reconstruct_files = "/Users/larcuser/pc_folder/data/bair-train-test-eval/eval_h265_diff_crf_diff_presets/h265_crf17_medium/png/"

src_test = "/Users/larcuser/pc_folder/data/yolo_data/images/src_test/"
reconstructed_test = "/Users/larcuser/pc_folder/data/yolo_data/images/recon_crf17_medium_test/"


def get_png_files_in_dir(a_dir):
    return [name for name in os.listdir(a_dir)
            if name.lower().endswith('jpg')]


src_test_file_list = get_png_files_in_dir(src_test)

for file_name in src_test_file_list:
    folder_name = file_name.split("_")[0]
    new_file_name = file_name.split("_")[1].split('.')[0]
    # reconstructed_file_path = "{}{}.mp4_h265.mp4/{:04d}.png".format(reconstruct_files, folder_name, int(new_file_name) + 1)
    reconstructed_file_path = "{}{}.mp4_h265.mp4/{:04d}.png".format(reconstruct_files, folder_name, int(new_file_name) + 1)

    print(reconstructed_file_path)
    try:
        reconstructed_img = img_as_float32((io.imread(reconstructed_file_path)))
    except FileNotFoundError as e:
        print(str(e))
    res_frame = cv2.resize(reconstructed_img, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
    resized_reconstructed_frame_path="{}{}_{}.jpg".format(reconstructed_test, folder_name, new_file_name)
    imageio.imwrite(resized_reconstructed_frame_path, res_frame)