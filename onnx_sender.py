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

from animate import normalize_kp
import time
import onnxruntime as rt

WAIT = 30
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

    print("loading_model,", time.time())
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
    print("extracting_key_points,", time.time())
    for frame_idx in tqdm(range(driving.shape[2])):
        driving_frame = driving[:, :, frame_idx]
        current_frame = {sess.get_inputs()[0].name: to_numpy(driving_frame)}
        current_frame_kp = sess.run(None, current_frame)
        current_frame_kp = {"value":torch.from_numpy(current_frame_kp[0]), "jacobian":torch.from_numpy(current_frame_kp[1])}
        current_frame_kp = normalize_kp(kp_source=init_frame_kp, kp_driving=current_frame_kp,
                               kp_driving_initial=init_frame_kp, use_relative_movement=False,
                               use_relative_jacobian=False, adapt_movement_scale=False)
        kp.append(current_frame_kp)
    return kp


def write_compressed_keypoint_file(file_name, file):
    with gzip.open(file_name+".gz", "wb") as f:
        pickle.dump(np.array(file), f)


def write_compressed_keypoint_file2(file_name, file):
    np.savez_compressed(file_name+".npz", file)



if __name__ == "__main__":
    print("begin_wait,", time.time())
    time.sleep(WAIT)
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default='conv3_fp32_kpd.onnx', help="path to onnx checkpoint to restore")
    parser.add_argument("--driving_video", default='in_video/h264_long_t10_2fps.mp4', help="path to driving video")
    parser.add_argument("--out_kp_file", default='working/onnx_fp32_f2_t10.kp', help="path to output keypoints file")
    parser.add_argument("--out_img_file", default='working/src_image.jpeg', help="path to output image file")
    parser.add_argument("--run_time", default='gpu', help="choose between cpu, gpu or trt")
    parser.add_argument("--fp", default='32', help="precision of the model weights")

    opt = parser.parse_args()

    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    print("reading_video,", time.time())
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()
    print("resizing_video,", time.time())
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    import copy
    source_image = copy.deepcopy(driving_video[0])

    src_img_file_name = opt.out_img_file
    imageio.imwrite(src_img_file_name, source_image)

    key_points = onnx_extract_keypoints(driving_video, opt.checkpoint, opt.fp, opt.run_time)

    #print("process_key_points,", time.time())
    processed_keypoints = key_points

    print("save_key_points,", time.time())
    np.save(opt.out_kp_file, np.array(processed_keypoints), allow_pickle=True)
    write_compressed_keypoint_file2(opt.out_kp_file+"_compressed", processed_keypoints)

#
    print("end,", time.time())

    time.sleep(WAIT)
