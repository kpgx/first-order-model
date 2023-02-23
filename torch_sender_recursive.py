import subprocess
import time
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

LOG_FOLDER = "models_to_eval/mgif/5_6"
CHECKPOINT = LOG_FOLDER + "/00000099-checkpoint.pth.tar"
CONFIG = LOG_FOLDER + "/mgif-256.yaml"
COMMAND = ["python3", "torch_sender.py",]
IN_DIR = "data/moving-gif/test"
OUT_DIR = "out_data/mgif/5_6"
NUM_OF_FILES = 10



def get_file_list_from_dir(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles


file_list = get_file_list_from_dir(IN_DIR)[:NUM_OF_FILES]

for file_name in tqdm(file_list):
    driving_video = join(IN_DIR, file_name)
    out_kp_file = join(OUT_DIR, file_name+".kp")
    out_img_file = join(OUT_DIR, file_name+".jpeg")
    current_command = COMMAND + ["--checkpoint",CHECKPOINT, "--config", CONFIG, "--driving_video", driving_video, '--out_kp_file', out_kp_file, '--out_img_file', out_img_file]
    #print(" ".join(current_command))
    subprocess.run(current_command)

