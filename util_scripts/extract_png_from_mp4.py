import os
import subprocess

DIR = "/Users/larcuser/pc_folder/data/vox-test"
PNG_DIR = "/Users/larcuser/pc_folder/data/vox-test/png"
COMMAND = ["ffmpeg", "-i", "placeholder1", "placeholder2"]


def create_dir_if_not_exists(name):
    if not os.path.exists(name):
        os.makedirs(name)


def get_video_list_from_dir(name):
    vidoes = sorted([x for x in os.listdir(name) if x.lower().endswith('.mp4')])
    return vidoes

create_dir_if_not_exists(PNG_DIR)
video_list = get_video_list_from_dir(DIR)
for file_name in video_list:
    out_folder = os.path.join(PNG_DIR, file_name)
    create_dir_if_not_exists(out_folder)
    placeholder2 = "{}/{}/%04d.png".format(PNG_DIR,file_name)
    current_command = COMMAND
    current_command[2]=os.path.join(DIR, file_name)
    current_command[3] = placeholder2
    print(current_command)
    subprocess.run(current_command)
