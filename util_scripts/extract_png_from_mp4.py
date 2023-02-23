import os
import subprocess
from os import listdir
from os.path import isfile, join

<<<<<<< HEAD

DIR = "../data/moving-gif/test"
PNG_DIR = "../test_data_subset/mgif"
=======
DIR = "/Users/larcuser/tmp/test_subset"
PNG_DIR = DIR + "/png"
>>>>>>> c234fa855e6918563ad7498fab9230090138f0dc
COMMAND = ["ffmpeg", "-i", "placeholder1", "placeholder2"]
NUMBER_OF_FILES = 10


def create_dir_if_not_exists(name):
    if not os.path.exists(name):
        os.makedirs(name)


def get_file_list_from_dir(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles


video_list = get_file_list_from_dir(DIR)[:NUMBER_OF_FILES]

create_dir_if_not_exists(PNG_DIR)
for file_name in video_list:
    out_folder = os.path.join(PNG_DIR, file_name)
    create_dir_if_not_exists(out_folder)
    placeholder2 = "{}/{}/%04d.png".format(PNG_DIR,file_name)
    current_command = COMMAND
    current_command[2]=os.path.join(DIR, file_name)
    current_command[3] = placeholder2
    print(current_command)
    subprocess.run(current_command)
