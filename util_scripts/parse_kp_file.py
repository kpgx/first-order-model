import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--kp_file", required=True, help="kp file path")

opt = parser.parse_args()

kp_file_name = opt.kp_file

key_points = np.load(kp_file_name, allow_pickle=True)

for kp in key_points:
    print(kp)
