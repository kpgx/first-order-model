import os
from argparse import ArgumentParser

import cv2.cv2
import imageio
import numpy as np
from imageio import mimread
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from skimage import img_as_ubyte


important_keypoints = [0,6]


def get_video_array(file_name):
    video = np.array(mimread(file_name))
    if len(video.shape) == 3:
        video = np.array([gray2rgb(frame) for frame in video])
    if video.shape[-1] == 4:
        video = video[..., :3]
    video_array = img_as_float32(video)

    return video_array


def get_frames_from_dir(name):
    frames = sorted([x for x in os.listdir(name) if x.lower().endswith('.png')])  # only consider png files in DIR
    num_frames = len(frames)
    video_array = np.array(
        [img_as_float32(gray2rgb(io.imread(os.path.join(name, frames[idx])))) for idx in range(num_frames)])

    return video_array


def get_keypoints(file_name):
    key_points = np.load(file_name, allow_pickle=True)
    return key_points


def get_raw_coordinates(cordinates, shape):
    width, height = shape[:2]
    x_min = int((cordinates['x_min'] + 1) * width/2)
    x_max = int((cordinates['x_max'] + 1) * width/2)
    y_min = int((cordinates['y_min'] + 1) * height/2)
    y_max = int((cordinates['y_max'] + 1) * height/2)

    return {'x_min':x_min, 'y_min':y_min, 'x_max':x_max, 'y_max':y_max}


def get_min_max(int_list):
    return min(int_list), max(int_list)


def get_xy_min_max(key_points):
    x_list = []
    y_list = []
    for index, kp in enumerate(key_points.squeeze().tolist()):
        if index not in important_keypoints:
            continue
        x, y = kp
        x_list.append(x)
        y_list.append(y)
    x_min, x_max = get_min_max(x_list)
    y_min, y_max = get_min_max(y_list)
    return {'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}


def get_blured_cropped_frame(frame, coord):
    box_height = coord['y_max'] - coord['y_min']
    box_width = box_height/2

    x_min = coord['x_min']-int(box_width/1.5)
    x_max = coord['x_max']+int(box_width/2)
    y_min = coord['y_min']
    y_max = coord['y_max']

    print("inside blured image", x_min, x_max, y_min, y_max)

    mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)

    reversed_mask = cv2.bitwise_not(mask)

    blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)
    frame[reversed_mask > 0] = blurred_frame[reversed_mask > 0]
    return frame


def get_black_cropped_frame(frame, coord):
    box_width = coord['x_max']-coord['x_min']
    box_height = coord['y_max'] - coord['y_min']

    x_min = coord['x_min']-int(box_width/2)
    x_max = coord['x_max']+int(box_width/2)
    y_min = coord['y_min']-int(box_height/2)
    y_max = coord['y_max']+int(box_height/2)

    mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    return frame


def get_kp_marked_frame(frame, kp_list):
    abs_kp_list=[]
    width, height = frame.shape[:2]
    kp_mark_color = (255,255,255)
    kp_number_color = kp_mark_color
    for kp in kp_list.squeeze().tolist():
        abs_kp_list.append([int((kp[0]+1)*width/2),int((kp[1]+1)*height/2)])
    for idx, kp in enumerate(abs_kp_list):
        cv2.circle(frame, (kp[0], kp[1]), 0, kp_mark_color, 3)
        cv2.putText(frame, str(idx), (kp[0], kp[1]), cv2.FONT_HERSHEY_SIMPLEX, .4, kp_number_color, 1, cv2.LINE_AA)
    return frame


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in_frame_dir", default='working/bair/h265/PNG/000000.mp4_h265.mp4', help="path to src video")
    parser.add_argument("--out_frame_dir", default='working/bair/h265/cropped_png/000000.mp4_h265.mp4/', help="path to output video")
    parser.add_argument("--kp_file", default='working/bair/kp/000000.mp4.kp.npy', help="path to key-points file")
    parser.add_argument("--crop", default="blured", choices=["black", "blured", "mark_kp"])
    sampled_img_folder = "/Users/larcuser/Data/bair-eval-for-object-detection/images/"
    cropped_img_folder = "/Users/larcuser/Data/bair-eval-for-object-detection/marked_for_test/"
    annotations_folder = "/Users/larcuser/Data/bair-eval-for-object-detection/labels/"

    opt = parser.parse_args()
    keypoints = get_keypoints(opt.kp_file)
    frames = get_frames_from_dir(opt.in_frame_dir)

    if not os.path.exists(sampled_img_folder):
        os.makedirs(sampled_img_folder)

    if not os.path.exists(annotations_folder):
        os.makedirs(annotations_folder)
    if not os.path.exists(cropped_img_folder):
        os.makedirs(cropped_img_folder)

    for idx in range(len(frames)):
        current_frame = frames[idx]
        current_kp_set = keypoints[idx]['value']
        xy_min_max = get_xy_min_max(current_kp_set)
        xy_min_max_raw = get_raw_coordinates(xy_min_max, current_frame.shape)
        # if opt.crop == "blured":
        #     cropped_frame = get_blured_cropped_frame(current_frame, xy_min_max_raw)
        # elif opt.crop == "mark_kp":
        #     cropped_frame = get_kp_marked_frame(current_frame, current_kp_set)
        # else:
        #     cropped_frame = get_black_cropped_frame(current_frame, xy_min_max_raw)

        res_frame = cv2.resize(current_frame, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
        img_name = opt.in_frame_dir.split('/')[-1].split('.')[0]+ "_{0:04d}.jpg".format(idx)
        txt_name = opt.in_frame_dir.split('/')[-1].split('.')[0] + "_{0:04d}.txt".format(idx)

        # imageio.imwrite(os.path.join(cropped_img_folder, img_name), img_as_ubyte(cropped_frame))

        # box_height = coord['y_max'] - coord['y_min']
        # box_width = box_height / 2
        #
        # x_min = coord['x_min'] - int(box_width / 1.5)
        # x_max = coord['x_max'] + int(box_width / 2)
        # y_min = coord['y_min']
        # y_max = coord['y_max']

        height = xy_min_max_raw['y_max'] - xy_min_max_raw['y_min']
        width = height / 2

        x_min = xy_min_max_raw['x_min'] - int(width / 1.5)
        x_max = xy_min_max_raw['x_max'] + int(width / 2)
        y_min = xy_min_max_raw['y_min']
        y_max = xy_min_max_raw['y_max']

        print("outside", x_min, x_max, y_min, y_max)

        # height = y_max-y_min
        # width = x_max - x_min
        x_center = (x_max + x_min)/2
        y_center = (y_max + y_min) / 2

        annotation = "{} {} {} {} {}".format(0, x_center / 256, y_center / 256, width/256, height/256)
        imageio.imwrite(os.path.join(sampled_img_folder, img_name), res_frame)

        with open(os.path.join(annotations_folder, txt_name), 'w') as f:
            f.write(annotation)
        # imageio.imwrite(os.path.join(opt.out_frame_dir, "{0:04d}.png".format(idx)), img_as_ubyte(current_frame))
        # cv2.imshow("img", current_frame)
        # cv2.waitKey(0)