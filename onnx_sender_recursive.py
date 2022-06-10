import subprocess
import time
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

CHECKPOINT = "pc_folder/checkpoints/bair_new_conv2/checkpoint.onnx"
RUN_TIME = "gpu"
FP = "32"
COMMAND = ["python3", "onnx_sender.py",]
DIR = "pc_folder/data/bair-train-test-eval/eval"


def get_file_list_from_dir(mypath):
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles


file_list = get_file_list_from_dir(DIR)

for file_name in tqdm(file_list):
    driving_video = file_name
    out_kp_file = file_name+".kp"
    out_img_file = file_name+".jpeg"
    current_command = COMMAND + ["--checkpoint",CHECKPOINT, "--driving_video", driving_video, '--out_kp_file', out_kp_file, '--out_img_file', out_img_file, "--run_time", RUN_TIME, "--fp", FP]
    #print(current_command)
    subprocess.run(current_command)

