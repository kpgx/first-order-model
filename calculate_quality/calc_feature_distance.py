import itertools

import cv2
import mediapipe as mp
import os
from scipy.spatial import distance
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

out_file = "/Users/larcuser/Projects/first-order-model/working/feature_distance_left_eye.csv"
src_img_dir = "/Users/larcuser/pc_folder/data/subset_of_data_for_feature_distance_study/src"
recon_img_dir = "/Users/larcuser/pc_folder/data/subset_of_data_for_feature_distance_study/fom_recon"
# recon_img_dir = src_img_dir
# src_out_dir = "/Users/larcuser/pc_folder/data/subset_of_data_for_feature_distance_study/src_annotated"
# recon_out_dir = "/Users/larcuser/pc_folder/data/subset_of_data_for_feature_distance_study/recon_annotated"


LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
LIPS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
# RIGHT_IRIS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
# LEFT_IRIS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
RIGHT_EYEBROW_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW)))
LEFT_EYEBROW_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW)))
# FACEOVAL_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))


def get_sub_folder_list(a_dir):
    return sorted([os.path.join(a_dir,name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))])


def get_png_files_in_dir(a_dir):
    return sorted([name for name in os.listdir(a_dir)
            if name.lower().endswith('png')])


def get_filtered_features(feature_lists_per_image, filter_indices):
    selected_feature_lists_per_img = []
    for current_img_feature_list in feature_lists_per_image:
        selected_feature_list = [current_img_feature_list[i] for i in filter_indices]
        selected_feature_lists_per_img.append(selected_feature_list)
    return selected_feature_lists_per_img


def get_deep_features_per_image(img_files):
    feature_list = []
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        for idx, file in enumerate(tqdm(img_files)):
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            landmark_list = list(results.multi_face_landmarks[0].landmark)
            feature_list.append(landmark_list)
    return feature_list


def get_xy_list_from_landmark_list(land_mark_list):
    return_list = []
    for landmark in land_mark_list:
        x = landmark.x
        y = landmark.y
        return_list.append((x, y))
    return return_list


def get_euclidean_distance_between_two_lists(al, bl):
    dist_list = []
    for i in range(len(al)):
        dst = distance.euclidean(al[i], bl[i])
        dist_list.append(dst)
    return dist_list


def get_the_feature_distance(src_feature_list, target_feature_list):
    avg_distance_list=[]
    if len(src_feature_list) != len(target_feature_list):
        print("src len[{}] not equal to target len [{}]".format(len(src_feature_list), len(target_feature_list)))
        return [-1,]
    for idx in range(len(src_feature_list)):
        current_src_feature = src_feature_list[idx]
        current_src_feature_xy = get_xy_list_from_landmark_list(current_src_feature)

        current_target_feature = target_feature_list[idx]
        current_target_feature_xy = get_xy_list_from_landmark_list(current_target_feature)

        current_distance_list = get_euclidean_distance_between_two_lists(current_src_feature_xy, current_target_feature_xy)
        # avg_distance = sum(current_distance_list)/len(current_distance_list)
        avg_distance_list.append(current_distance_list)
    return avg_distance_list


def write_deep_features_to_file(out_dir, img_name, features, image):
    annotated_image = image.copy()
    for face_landmarks in features.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
    cv2.imwrite("{}{}".format(out_dir, img_name), annotated_image)


src_img_files = []
recon_img_files = []

for sub_dir in get_sub_folder_list(src_img_dir):
    cur_src_img_files = [os.path.join(sub_dir, f) for f in get_png_files_in_dir(sub_dir)]
    src_img_files += cur_src_img_files

for sub_dir in get_sub_folder_list(recon_img_dir):
    cur_recon_img_files = [os.path.join(sub_dir, f) for f in get_png_files_in_dir(sub_dir)]
    recon_img_files += cur_recon_img_files

src_features = get_deep_features_per_image(src_img_files)
recon_features = get_deep_features_per_image(recon_img_files)

#### left eye
left_eye_src_features = get_filtered_features(src_features, LEFT_EYE_INDEXES)
left_eye_recon_features = get_filtered_features(recon_features, LEFT_EYE_INDEXES)

dist_list_per_image = get_the_feature_distance(left_eye_src_features, left_eye_recon_features)

out_file = "/Users/larcuser/Projects/first-order-model/working/feature_distance_left_eye.csv"


with open(out_file, 'w') as f:
    f.write("src_img, recon_img, {}\n".format(", ".join([str(idx) for idx in range(len(dist_list_per_image[0]))])))
    for i in range(len(src_img_files)):
        f.write("{}, {}, {}\n".format(src_img_files[i], recon_img_files[i], ", ".join([str(dist) for dist in dist_list_per_image[i]])))


#### right eye
right_eye_src_features = get_filtered_features(src_features, RIGHT_EYE_INDEXES)
right_eye_recon_features = get_filtered_features(recon_features, RIGHT_EYE_INDEXES)

dist_list_per_image = get_the_feature_distance(right_eye_src_features, right_eye_recon_features)

out_file = "/Users/larcuser/Projects/first-order-model/working/feature_distance_right_eye.csv"


with open(out_file, 'w') as f:
    f.write("src_img, recon_img, {}\n".format(", ".join([str(idx) for idx in range(len(dist_list_per_image[0]))])))
    for i in range(len(src_img_files)):
        f.write("{}, {}, {}\n".format(src_img_files[i], recon_img_files[i], ", ".join([str(dist) for dist in dist_list_per_image[i]])))


####left eyebrow
left_eyebrow_src_features = get_filtered_features(src_features, LEFT_EYEBROW_INDEXES)
left_eyebrow_recon_features = get_filtered_features(recon_features, LEFT_EYEBROW_INDEXES)

dist_list_per_image = get_the_feature_distance(left_eyebrow_src_features, left_eyebrow_recon_features)

out_file = "/Users/larcuser/Projects/first-order-model/working/feature_distance_left_eyebow.csv"


with open(out_file, 'w') as f:
    f.write("src_img, recon_img, {}\n".format(", ".join([str(idx) for idx in range(len(dist_list_per_image[0]))])))
    for i in range(len(src_img_files)):
        f.write("{}, {}, {}\n".format(src_img_files[i], recon_img_files[i], ", ".join([str(dist) for dist in dist_list_per_image[i]])))

#### right eyebrow
right_eyebrow_src_features = get_filtered_features(src_features, RIGHT_EYEBROW_INDEXES)
right_eyebrow_recon_features = get_filtered_features(recon_features, RIGHT_EYEBROW_INDEXES)

dist_list_per_image = get_the_feature_distance(right_eyebrow_src_features, right_eyebrow_recon_features)

out_file = "/Users/larcuser/Projects/first-order-model/working/feature_distance_right_eyebrow.csv"


with open(out_file, 'w') as f:
    f.write("src_img, recon_img, {}\n".format(", ".join([str(idx) for idx in range(len(dist_list_per_image[0]))])))
    for i in range(len(src_img_files)):
        f.write("{}, {}, {}\n".format(src_img_files[i], recon_img_files[i], ", ".join([str(dist) for dist in dist_list_per_image[i]])))

### lips
lips_src_features = get_filtered_features(src_features, LIPS_INDEXES)
lips_recon_features = get_filtered_features(recon_features, LIPS_INDEXES)

dist_list_per_image = get_the_feature_distance(lips_src_features, lips_recon_features)

out_file = "/Users/larcuser/Projects/first-order-model/working/feature_distance_lips.csv"


with open(out_file, 'w') as f:
    f.write("src_img, recon_img, {}\n".format(", ".join([str(idx) for idx in range(len(dist_list_per_image[0]))])))
    for i in range(len(src_img_files)):
        f.write("{}, {}, {}\n".format(src_img_files[i], recon_img_files[i], ", ".join([str(dist) for dist in dist_list_per_image[i]])))

print("that's all folks")