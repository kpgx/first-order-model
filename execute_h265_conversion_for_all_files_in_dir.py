import subprocess
import time
from os import listdir
from os.path import isfile, join
import time
import os

COMMAND = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", "placeholder.mp4", "-c:v", "hevc", "-preset", "medium","-crf", "28", "-x265-params", "bframes=0", "placeholder.mp4"]
DIR = "mgif-test"
OUT_DIR = join(DIR, "h265")
SINGLE_LOG_FILE_NAME = "h265_dir.csv"
WAIT = 10
TIMES = 100
SRC_EXT = '.gif'


def write_log_entry(file_name, line):
    with open(file_name, 'a+') as f:
        f.write(line)


def get_file_size_in_KB(file_name):
    return os.path.getsize(file_name)/1000


def get_src_file_list_from_dir(mypath):
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    only_src_files = [f for f in onlyfiles if f.lower().endswith(SRC_EXT)]
    return only_src_files


file_list = get_src_file_list_from_dir(DIR)

for file_name in file_list:
    src_video = file_name
    out_video = join(OUT_DIR, file_name+"_h265.mp4")
    
    current_command = COMMAND
    current_command[6] = src_video
    current_command[-1] = out_video

    write_log_entry(SINGLE_LOG_FILE_NAME, "{}, ".format(src_video))
    wait_st=time.time()
    time.sleep(WAIT)
    wait_end = time.time()
    write_log_entry(SINGLE_LOG_FILE_NAME, "{},{}, ".format(wait_st, wait_end))
    print(current_command)
    st_time = time.time()
    for i in range(TIMES):
        subprocess.run(current_command)
    end_time = time.time()
    write_log_entry(SINGLE_LOG_FILE_NAME, "{}, {}, ".format(st_time, end_time))
    write_log_entry(SINGLE_LOG_FILE_NAME, "{}, ".format(get_file_size_in_KB(src_video)))
    write_log_entry(SINGLE_LOG_FILE_NAME, "{}\n".format(get_file_size_in_KB(out_video)))


