import subprocess

from tqdm import tqdm

src_root = "../test_data_subset/vox"
target_root = "../out_data/vox/10_6"
master_q_log_file = src_root+"/vox_10_6_fom_full_q_matrics.csv"

import os

def get_sub_folder_list(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def get_png_files_in_dir(a_dir):
    return [name for name in os.listdir(a_dir)
            if name.lower().endswith('png')]


for src_sub_dir in tqdm(get_sub_folder_list(src_root)):
    src_sub_dir_path = os.path.join(src_root, src_sub_dir)
    target_sub_dir_path = os.path.join(target_root, src_sub_dir+"")
    command = ["python3", "calculate_quality_matrics.py", "--src_dir", src_sub_dir_path, "--target_dir", target_sub_dir_path, "--master_q_file", master_q_log_file]
    # print(command)
    subprocess.run(command)