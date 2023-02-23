import subprocess

from tqdm import tqdm

src_root = "/Users/larcuser/tmp/test_subset/png"
# target_root = src_root

target_root = "/Users/larcuser/tmp/test_subset_reconstructed/taichi-256-10-6"
master_q_log_file = "/Users/larcuser/Projects/first-order-model/working/src_vs_taichi_10-6.csv"

import os

def get_sub_folder_list(a_dir):
    return sorted([name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))])


def get_png_files_in_dir(a_dir):
    return sorted([name for name in os.listdir(a_dir)
            if name.lower().endswith('png')])


for src_sub_dir in tqdm(get_sub_folder_list(src_root)):
    src_sub_dir_path = os.path.join(src_root, src_sub_dir)
    target_sub_dir_path = os.path.join(target_root, src_sub_dir+"")
    command = ["python3", "/Users/larcuser/Projects/first-order-model/calculate_quality/calculate_quality_matrics.py", "--src_dir", src_sub_dir_path, "--target_dir", target_sub_dir_path, "--master_q_file", master_q_log_file]
    # print(command)
    subprocess.run(command)