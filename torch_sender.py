import matplotlib

matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

# from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull
import gzip
import pickle
import time

WAIT = 0

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    print("loading_model,", time.time())
    with open(config_path) as f:
        config = yaml.load(f)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        kp_detector = DataParallelWithCallback(kp_detector)

    kp_detector.eval()

    return None, kp_detector


def extract_keypoints(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True,
                      cpu=False):
    kp = []
    with torch.no_grad():
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)

        if not cpu:
            source = source.cuda()
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        print("extracting_key_points,", time.time())
        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            # extract the keypoints from current video frame
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            kp.append(kp_norm)
    return kp


def write_compressed_keypoint_file(file_name, file):
    with gzip.open(file_name+".gz", "wb") as f:
        pickle.dump(np.array(file), f)


if __name__ == "__main__":
    print("begin_wait,", time.time())
    time.sleep(WAIT)
    parser = ArgumentParser()
    parser.add_argument("--config", default='config/00/bair-256.yaml', help="path to config")
    parser.add_argument("--checkpoint",
                        default='checkpoints/00/bair-cpk.pth.tar',
                        help="path to checkpoint to restore")

    parser.add_argument("--driving_video", default='h264_long.mp4', help="path to driving video")
    parser.add_argument("--cpu", default=False, dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument("--out_kp_file", default='working/torch32.kp', help="path to output keypoints file")
    parser.add_argument("--out_img_file", default='working/src_image.jpeg', help="path to output image file")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

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
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    import copy

    source_image = copy.deepcopy(driving_video[0])

    imageio.imwrite(opt.out_img_file, source_image)

    _, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    key_points = extract_keypoints(source_image, driving_video, None, kp_detector, relative=opt.relative,
                                   adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
    print("process_key_points,", time.time())
    print("save_key_points,", time.time())
    kp_save_start = time.time()
    np.save(opt.out_kp_file, np.array(key_points), allow_pickle=True)

    # save compressed file(for file size comparison)
    write_compressed_keypoint_file(opt.out_kp_file+"_compressed", key_points)

    print("end,", time.time())
    time.sleep(WAIT)
