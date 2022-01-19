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
import onnx
import onnxruntime as rt


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


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def onnx_extract_keypoints(video, kp_detector_file_name):
    kp = []
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = rt.InferenceSession(kp_detector_file_name, providers=EP_list)
    driving = torch.tensor(np.array(video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
    #print(sess.get_inputs()[0].name, type(to_numpy(kp_driving_initial)))
    init_frame = {sess.get_inputs()[0].name: to_numpy(driving[:,:, 0])}
    init_frame_kp = sess.run(None, init_frame)
    # loop the video                                                                                                                                                                                                                                                      
    for frame_idx in tqdm(range(driving.shape[2])):
        driving_frame = driving[:, :, frame_idx]

        # extract the keypoints from current video frame
        current_frame = {sess.get_inputs()[0].name: to_numpy(driving_frame)}
        current_frame_kp = sess.run(None, current_frame)
#        print(current_frame_kp)
#        kp_driving = kp_detector(driving_frame)
#        kp_norm = normalize_kp(kp_source=init_frame_kp, kp_driving=current_frame_kp,
#                               kp_driving_initial=init_frame_kp, use_relative_movement=False,
#                               use_relative_jacobian=False, adapt_movement_scale=False)
        kp.append(current_frame_kp)
    return kp

def extract_keypoints(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    kp=[]
    with torch.no_grad():
        #predictions = []
	# store the source image in a tensor/
	# no idea what's the permute do
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        if not cpu:
            source = source.cuda()
	# store the video in a tensor
	# find the key points in the source image(in our case frame 1
        kp_source = kp_detector(source)
	# find the key points in the source image(in our case frame 1
        kp_driving_initial = kp_detector(driving[:, :, 0])
	# loop the video
        for frame_idx in tqdm(range(driving.shape[2])):
            #print("frame id", frame_idx)
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
	    # extract the keypoints from current video frame
            kp_driving = kp_detector(driving_frame)
            print(kp_driving)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            kp.append(kp_norm)
            #out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            #predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
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

    key_points = onnx_extract_keypoints(driving_video, "first-order-model-kp_detector.onnx")
    
#    _, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
#    key_points = extract_keypoints(source_image, driving_video, None, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
#    #print("KP TYPE", type(key_points))

#    print("type", type(key_points), "shape", key_points.shape)
    processed_keypoints = []
    for kp in key_points:
        val = torch.from_numpy(kp[0])
        jacobian = torch.from_numpy(kp[1])
        dic = {'value':val, 'jacobian':jacobian}
        processed_keypoints.append(dic)
#    print(type(key_points[0]))
#    print(type(key_points[0][0]))
#    print(type(key_points[0][0][0]))
#    print(type(key_points[0][0][0][0]))
#    print(torch.from_numpy(key_points[0]))
    np.save("working/keypoints", np.array(processed_keypoints), allow_pickle=True)
#
    #save compressed file(for file size comparison)
#    with gzip.open("gzip_kp.gz", "wb") as f:
#        pickle.dump(np.array(processed_keypoints), f)
#
