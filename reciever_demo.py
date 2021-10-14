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

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull
import bz2
import pickle
import _pickle as cPickle


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_generator(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)

    generator.eval()
    
    return generator


def make_animation(source_image, key_points, generator, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        #driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = key_points[0]
        #kp_driving_initial = kp_detector(driving[:, :, 0])

        for kp_norm in key_points:
        #for frame_idx in tqdm(range(driving.shape[2])):
            #driving_frame = driving[:, :, frame_idx]
            #if not cpu:
            #    driving_frame = driving_frame.cuda()
            #kp_driving = kp_detector(driving_frame)
            #kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
            #                       kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
            #                       use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


def load_compressed_pickle(file_name):
    data = bz2.BZ2File(file_name, 'rb')
    data = cPickle.load(data)
    return data

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--result_video", default='result.mp4', help="path to output")
 
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument("--in_file_name", default="video.pkl.pbz2", help="sender output file")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    generator = load_generator(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

    in_file = load_compressed_pickle(opt.in_file_name)

    source_image = in_file["src_img"]

    fps = in_file["fps"]

    key_points = in_file["key_points"]

    predictions = make_animation(source_image, key_points, generator, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)

