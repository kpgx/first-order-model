import subprocess
import time
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

COMMAND = ["python3", "reciever.py",]
DIR = "test-data/vox-eval-100"
FOM_DIR = "test-data/vox-eval-100/fom/png"
src_file_ext = 'mp4'


def get_src_file_list_from_dir(mypath, ext):
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    only_src_files = [f for f in onlyfiles if f.lower().endswith(ext)]
    return only_src_files


file_list = get_src_file_list_from_dir(DIR, src_file_ext)

for file_name in tqdm(file_list):
    result_video = join(FOM_DIR, file_name)
    in_kp_file = file_name+".kp.npy"
    in_img_file = file_name+".jpeg"
    current_command = COMMAND + ["--result_dir", result_video, '--kp_file', in_kp_file, '--src_img', in_img_file]
    #print(current_command)
    subprocess.run(current_command)

