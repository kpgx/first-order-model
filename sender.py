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

#from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull
import gzip
import pickle
import time


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    #generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
    #                                    **config['model_params']['common_params'])
    #if not cpu:
    #    generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    #generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
    #    generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    #generator.eval()
    kp_detector.eval()
    
    return None, kp_detector


def extract_keypoints(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    kp=[]
    with torch.no_grad():
        #predictions = []
	# store the source image in a tensor/
	# no idea what's the permute do
        start = time.time()
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        end = time.time()
        print(" loading data in to tensors", end - start)
        if not cpu:
            source = source.cuda()
	# store the video in a tensor
	# find the key points in the source image(in our case frame 1
        start = time.time()
        kp_source = kp_detector(source)
	# find the key points in the source image(in our case frame 1
        kp_driving_initial = kp_detector(driving[:, :, 0])
        end = time.time()
        print(" extracting kp from static images", end - start)
	# loop the video
        start = time.time()
        for frame_idx in tqdm(range(driving.shape[2])):
            #print("frame id", frame_idx)
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
	    # extract the keypoints from current video frame
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            kp.append(kp_norm)
            #out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            #predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        end = time.time()
        print(" extract key points from video", end - start)
    return kp

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

if __name__ == "__main__":
    total_start = time.time()
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

#    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/source.png', help="path to driving video")
#    parser.add_argument("--result_video", default='result.mp4', help="path to output")
 
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
                        help="Set frame to start from.")
 
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
 

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    print("OPT", opt)

#    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    print("FPS", fps)
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    import copy
    source_image = copy.deepcopy(driving_video[0])

    source_image = resize(source_image, (256, 256))[..., :3]

    src_img_file_name = "working/src_image.jpeg"
    imageio.imwrite(src_img_file_name, source_image)

    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    load_start = time.time()
    _, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    load_end = time.time()
    print("loading kp detector", load_end-load_start)
    kp_extract_start = time.time()
    key_points = extract_keypoints(source_image, driving_video, None, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
    #print("KP TYPE", type(key_points))
    kp_extract_end = time.time()
    print("extracting key points", kp_extract_end-kp_extract_start)
    kp_save_start = time.time()
    np.save("working/keypoints", np.array(key_points), allow_pickle=True)

    #save compressed file(for file size comparison)
    with gzip.open("gzip_kp.gz", "wb") as f:
        pickle.dump(np.array(key_points), f)
    kp_save_end = time.time()
    print("kp save", kp_save_end - kp_save_start)
    total_end = time.time()
    print("total time", total_end - total_start)
    #imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)

