import os
from os.path import isfile, join
from os import listdir

folder_path = "/Users/larcuser/tmp/test_subset_reconstructed/taichi-256-10-6"
kp_file_ext = "kp_compressed.gz"


def get_file_size_in_KB(file_name):
    return os.path.getsize(file_name)/1000


def get_src_file_list_from_dir(mypath, ext):
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    only_src_files = [f for f in onlyfiles if f.lower().endswith(ext)]
    return only_src_files


file_list = get_src_file_list_from_dir(folder_path, kp_file_ext)
for file in file_list:
    print(f"{file}, {get_file_size_in_KB(file)}")