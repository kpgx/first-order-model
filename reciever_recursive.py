import subprocess
import time
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

# LOG_FOLDER = "log/taichi-256/taichi-256-10-6"
# CHECKPOINT = LOG_FOLDER + "/00000099-checkpoint.pth.tar"
LOG_FOLDER = "models_to_eval/vox/10_6"
CONFIG = LOG_FOLDER + "/vox-256.yaml"
COMMAND = ["python3", "reciever.py",]
FOM_DIR = "out_data/vox/10_6"
# FOM_DIR = "reconstructed_sub/mgif/5_6"
src_file_ext = 'jpeg'


CHECKPOINT = LOG_FOLDER + "/00000099-checkpoint.pth.tar"
# CONFIG = LOG_FOLDER + "/mgif-256.yaml"
# COMMAND = ["python3", "torch_sender.py",]
# IN_DIR = "data/moving-gif/test"
# OUT_DIR = "out_data/mgif/5_6"
# NUM_OF_FILES = 10


def get_src_file_list_from_dir(mypath, ext):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    only_src_files = [f for f in onlyfiles if f.lower().endswith(ext)]
    return only_src_files


file_list = get_src_file_list_from_dir(FOM_DIR, src_file_ext)

for file_name in tqdm(file_list):
    file_name = file_name[:-5]
    print(f"processing :{file_name}")
    result_video = join(FOM_DIR, file_name)
    in_kp_file = join(FOM_DIR, file_name+".kp.npy")
    in_img_file = join(FOM_DIR, file_name+".jpeg")
    current_command = COMMAND + ["--config", CONFIG,"--checkpoint", CHECKPOINT, "--result_dir", result_video, '--kp_file', in_kp_file, '--src_img', in_img_file]
    #print(" ".join(current_command))
    subprocess.run(current_command)

