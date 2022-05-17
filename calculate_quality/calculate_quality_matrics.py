import os
from argparse import ArgumentParser
import cv2 as cv
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from skimage import io, img_as_float32
from skimage.color import gray2rgb

LOG_FILE_NAME = "q_matrices.csv"


def getPSNR(ref, comp):
    psnr = cv.PSNR(ref, comp)
    return psnr


def getSSIM(ref, comp):
    ssim = compare_ssim(ref, comp, multichannel=True)
    return ssim


def get_frames_from_dir(name):
    frames = sorted([x for x in os.listdir(name) if x.lower().endswith('.png')])  # only consider png files in DIR
    num_frames = len(frames)
    video_array = np.array(
        [img_as_float32(gray2rgb(io.imread(os.path.join(name, frames[idx])))) for idx in range(num_frames)])

    return video_array


def write_log_entry(file_name, line):
    with open(file_name, 'a+') as f:
        f.write(line)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_dir", default='working/test_out/src', help="path to src img folder")
    parser.add_argument("--target_dir", default='working/test_out/fom', help="path to target img folder")
    parser.add_argument("--master_q_file", default='q_matrices.csv',help='master q matrices file')

    opt = parser.parse_args()

    log_file_name = os.path.join(opt.target_dir, LOG_FILE_NAME)
    src_frames = get_frames_from_dir(opt.src_dir)
    target_frames = get_frames_from_dir(opt.target_dir)

    if len(src_frames) != len(target_frames):
        print("length mismatch srt[{}]->[{}] and target[{}]->[{}]".format(opt.src_dir, len(src_frames), opt.target_dir, len(target_frames)))
        exit(1)
    # write header for individual logs
    write_log_entry(log_file_name, "frame_no, psnr, ssim\n")
    psnr_list = []
    ssim_list = []
    for idx in range(len(src_frames)):
        src_frame = src_frames[idx]
        target_frame = target_frames[idx]
        psnr = getPSNR(src_frame, target_frame)
        psnr_list.append(psnr)
        ssim = getSSIM(src_frame, target_frame)
        ssim_list.append(ssim)
        write_log_entry(log_file_name, "{}, {}, {}\n".format(idx, psnr, ssim))
    avg_psnr = sum(psnr_list)/len(psnr_list)
    avg_ssim = sum(ssim_list)/len(ssim_list)

    write_log_entry(opt.master_q_file, "{}, {}, {}, {}, {}\n".format(opt.src_dir, opt.target_dir, len(src_frames), avg_psnr, avg_ssim))

