import gzip
import pickle

import matplotlib
matplotlib.use('Agg')
import sys
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
import torch
from imageio import mimread
from skimage import io, img_as_float32


from animate import normalize_kp
import time
import onnxruntime as rt
import os

WAIT = 10
TIMES = 10
LOG_FILE_SUFFIX = "_fom.log"
LOG_FILE_NAME = "" # this get populated once video file is read
SINGLE_LOG_FILE_NAME = "main.log"
#WAIT = 60


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def onnx_extract_keypoints(video, kp_detector_file_name,fp, rtime='gpu'):
    kp = []
    EP_list = []
    if rtime=='cpu':
        EP_list.append('CPUExecutionProvider')
    if rtime=='gpu':
        EP_list.append('CUDAExecutionProvider')
    if rtime=='trt':
        EP_list.append('TensorrtExecutionProvider')

    write_log_entry(LOG_FILE_NAME, "loading_model, {}\n".format(time.time()))
    # print("loading_model,", time.time())
    sess = rt.InferenceSession(kp_detector_file_name, providers=EP_list)
    if fp == '16':
        driving = torch.tensor(np.array(video)[np.newaxis].astype(np.float16)).permute(0, 4, 1, 2, 3)
    elif fp == '32':
        driving = torch.tensor(np.array(video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
    else:
        print("unhandled floating point prec")
        return
    init_frame = {sess.get_inputs()[0].name: to_numpy(driving[:,:, 0])}
    init_frame_kp = sess.run(None, init_frame)
    init_frame_kp = {"value":torch.from_numpy(init_frame_kp[0]), "jacobian":torch.from_numpy(init_frame_kp[1])}
    # loop the video
    write_log_entry(LOG_FILE_NAME, "extracting_key_points_{}_times, {}\n".format(TIMES, time.time()))
    st_time = time.time()
    # print("extracting_key_points ", TIMES, ',', time.time())
    for i in range(TIMES):
        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            current_frame = {sess.get_inputs()[0].name: to_numpy(driving_frame)}
            current_frame_kp = sess.run(None, current_frame)
            current_frame_kp = {"value":torch.from_numpy(current_frame_kp[0]), "jacobian":torch.from_numpy(current_frame_kp[1])}
            current_frame_kp = normalize_kp(kp_source=init_frame_kp, kp_driving=current_frame_kp,
                                   kp_driving_initial=init_frame_kp, use_relative_movement=False,
                                   use_relative_jacobian=False, adapt_movement_scale=False)
            kp.append(current_frame_kp)
    end_time = time.time()
    write_log_entry(SINGLE_LOG_FILE_NAME, "{}, {}, ".format(st_time,end_time))
    return kp


def write_compressed_keypoint_file(file_name, file):
    with gzip.open(file_name+".gz", "wb") as f:
        pickle.dump(np.array(file), f)


def write_compressed_keypoint_file2(file_name, file):
    np.savez_compressed(file_name+".npz", file)


def get_video_array(file_name):
    video = np.array(mimread(file_name))
    if len(video.shape) == 3:
        video = np.array([gray2rgb(frame) for frame in video])
    if video.shape[-1] == 4:
        video = video[..., :3]
    video_array = img_as_float32(video)

    return video_array


def write_log_entry(file_name, line):
    with open(file_name, 'a+') as f:
        f.write(line)


def get_file_size_in_KB(file_name):
    return os.path.getsize(file_name)/1000


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default='checkpoints/onnx_models/conv3_fp32_kpd.onnx', help="path to onnx checkpoint to restore")
    parser.add_argument("--driving_video", default='working/in_video/h264_long.mp4', help="path to driving video")
    parser.add_argument("--out_kp_file", default='working/onnx_fp32_f2_t10.kp', help="path to output keypoints file")
    parser.add_argument("--out_img_file", default='working/src_image.jpeg', help="path to output image file")
    parser.add_argument("--run_time", default='gpu', help="choose between cpu, gpu or trt")
    parser.add_argument("--fp", default='32', help="precision of the model weights")

    opt = parser.parse_args()

    LOG_FILE_NAME = opt.driving_video+LOG_FILE_SUFFIX
    driving_video = get_video_array(opt.driving_video)
    # write_log_entry(SINGLE_LOG_FILE_NAME, "file_name, wait_start, wait_end, kp_extract_start, kp_extract_end, src_size, kp_size, compressed_kp_size\n")
    write_log_entry(SINGLE_LOG_FILE_NAME, "{}, ".format(opt.driving_video))

    write_log_entry(LOG_FILE_NAME, "begin_wait, {}\n".format(time.time()))
    # print("begin_wait,", time.time())
    wait_st=time.time()
    time.sleep(WAIT)
    wait_end = time.time()
    write_log_entry(SINGLE_LOG_FILE_NAME, "{},{}, ".format(wait_st, wait_end))

    import copy
    source_image = copy.deepcopy(driving_video[0])

    src_img_file_name = opt.out_img_file
    imageio.imwrite(src_img_file_name, source_image)

    key_points = onnx_extract_keypoints(driving_video, opt.checkpoint, opt.fp, opt.run_time)

    write_log_entry(LOG_FILE_NAME, "save_key_points, {}\n".format(time.time()))
    # print("save_key_points,", time.time())
    np.save(opt.out_kp_file, np.array(key_points), allow_pickle=True)
    write_compressed_keypoint_file2(opt.out_kp_file+"_compressed", key_points)
    
    write_log_entry(SINGLE_LOG_FILE_NAME, "{}, ".format(get_file_size_in_KB(opt.driving_video)))
    write_log_entry(SINGLE_LOG_FILE_NAME, "{}, ".format(get_file_size_in_KB(opt.out_kp_file+".npy")))
    write_log_entry(SINGLE_LOG_FILE_NAME, "{}\n".format(get_file_size_in_KB(opt.out_kp_file+"_compressed.npz")))


    write_log_entry(LOG_FILE_NAME, "end, {}\n".format(time.time()))
    # print("end,", time.time())