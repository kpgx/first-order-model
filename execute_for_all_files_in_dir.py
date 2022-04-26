import subprocess
import time
from os import listdir
from os.path import isfile, join

COMMAND = ["python3", "onnx_sender.py",]
DIR = "working"


def get_file_list_from_dir(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles


file_list = get_file_list_from_dir(DIR)

for file_name in file_list:
    driving_video = file_name
    out_kp_file = file_name+".kp"
    out_img_file = file_name+".jpeg"
    current_command = COMMAND + ["--driving_video", driving_video, '--out_kp_file', out_kp_file, '--out_img_file', out_img_file]
    print(current_command)
    subprocess.run(current_command)

