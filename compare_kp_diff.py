from os import listdir
from os.path import isfile, join
import numpy

SRC_DIR = "/Users/larcuser/pc_folder/data/bair-train-test-eval/kp_detector_stats/conv5/npy"
TARGET_DIR = "/Users/larcuser/pc_folder/data/bair-train-test-eval/kp_detector_stats/conv3/npy"


def get_src_file_list(dir):
    onlyfiles = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    return onlyfiles


def get_file_content(file_name):
    content = numpy.load(file_name, allow_pickle=True)
    return content


src = (get_file_content(get_src_file_list(SRC_DIR)[0]))
target = (get_file_content(get_src_file_list(TARGET_DIR)[0]))

print("that's all folks")