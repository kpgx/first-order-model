import subprocess
import time
from os import listdir
from os.path import isfile, join

COMMAND = ["python3", "reciever.py",]
DIR = "working"


def get_src_file_list_from_dir(mypath):
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    only_src_files = [f for f in onlyfiles if f.lower().endswith('.mp4')]
    return only_src_files


file_list = get_src_file_list_from_dir(DIR)

for file_name in file_list:
    result_video = file_name+"_fom"
    in_kp_file = file_name+".kp"
    in_img_file = file_name+".jpeg"
    current_command = COMMAND + ["--result_video", result_video, '--kp_file', in_kp_file, '--src_img', in_img_file]
    print(current_command)
    # subprocess.run(current_command)

