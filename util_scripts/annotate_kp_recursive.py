import subprocess

from tqdm import tqdm

png_folder = "/Users/larcuser/Data/bair-eval-for-object-detection/src/"
cropped_png_folder = "/Users/larcuser/Data/bair-eval-for-object-detection/png/cropped"
kp_folder = "/Users/larcuser/Data/bair-eval-for-object-detection/kp/"


import os

def get_sub_folder_list(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def get_png_files_in_dir(a_dir):
    return [name for name in os.listdir(a_dir)
            if name.lower().endswith('png')]


for sub_dir in tqdm(get_sub_folder_list(png_folder)):
    kp_file = os.path.join(kp_folder, sub_dir + '.kp.npy')

    # kp_file = os.path.join(kp_folder,sub_dir.split('_')[0]+'.kp.npy')
    current_png_folder = os.path.join(png_folder, sub_dir)
    current_cropped_png_folder = os.path.join(cropped_png_folder, sub_dir)
    command = ['python3', 'annotate_kp.py', '--in_frame_dir', current_png_folder, '--out_frame_dir', current_cropped_png_folder, '--kp_file', kp_file]
    # print(command)
    subprocess.run(command)