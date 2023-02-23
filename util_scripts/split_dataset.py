import os
import random
import shutil

img_dir ="/Users/larcuser/Data/bair-eval-for-object-detection/images/"
txt_dir ="/Users/larcuser/Data/bair-eval-for-object-detection/labels/"



train_img_dir =img_dir+"train/"
test_img_dir =img_dir+"test/"
val_img_dir =img_dir+"val/"
train_txt_dir =txt_dir +"train/"
test_txt_dir =txt_dir +"test/"
val_txt_dir =txt_dir +"val/"


def get_files_in_dir(a_dir, d_type):
    return [name for name in os.listdir(a_dir)
            if name.lower().endswith(d_type)]


img_list = get_files_in_dir(img_dir, "jpg")
txt_lit = get_files_in_dir(txt_dir, 'txt')

random.shuffle(img_list)

train_list = img_list[:2000]
test_list = img_list[2000:2600]
val_list = img_list[2600:]

def copy_files(root_file_list, src_img_dir, dst_img_dir, src_txt_dir, dst_txt_dir):
    for file_name in root_file_list:
        prefix = file_name.split(".")[0]
        img_name = prefix + ".jpg"
        txt_name = prefix + '.txt'
        shutil.copyfile(src_img_dir + img_name, dst_img_dir + img_name)
        shutil.copyfile(src_txt_dir + txt_name, dst_txt_dir + txt_name)

copy_files(train_list, img_dir, train_img_dir, txt_dir, train_txt_dir)
copy_files(val_list, img_dir, val_img_dir, txt_dir, val_txt_dir)
copy_files(test_list, img_dir, test_img_dir, txt_dir, test_txt_dir)

print("that's all folks")

