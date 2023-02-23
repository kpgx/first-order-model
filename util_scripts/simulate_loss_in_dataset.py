import os
import shutil
import numpy as np
from tqdm import tqdm


state_def = {'0': 'good', '1': 'poor'}
loss_def = {'1': 'loss', '0': 'no_loss'}

# channel state transition probabilities
packet_loss_prob_at_chanel_good = 0.01
packet_loss_prob_at_chanel_poor = 0.4
packet_loss_prob_at_chanel_random = 0.05
# channel state transition probabilities
good_to_good = 0.9
good_to_poor = 1-good_to_good
poor_to_poor = 0.7
poor_to_good = 1 - poor_to_poor

start_state = 0  # good = 0 or poor = 1

packet_loss_prob = {'good': packet_loss_prob_at_chanel_good,
                    'poor': packet_loss_prob_at_chanel_poor,
                    'random': packet_loss_prob_at_chanel_random}


# img_dir = "/Users/larcuser/pc_folder/data/packet_loss_study_data_set/src"
# lossy_img_dir = "/Users/larcuser/pc_folder/data/packet_loss_study_data_set/lossy_src"


img_dir = "/Users/larcuser/Data/packet_loss_study_data_set/src"
lossy_img_dir = "/Users/larcuser/Data/packet_loss_study_data_set/lossy_src"
stat_file = "/Users/larcuser/Projects/first-order-model/working/packet_loss_simulation_src.csv"


def get_channel_state_via_2_state_markov_model(seq_len, start_state=0):
    trans_matrix = [[good_to_good, good_to_poor], [poor_to_good, poor_to_poor]]
    prev_state = start_state
    ret_state_seq = [start_state, ]
    for i in range((seq_len - 1)):
        curr_state = np.random.choice([0, 1], p=trans_matrix[prev_state])
        ret_state_seq.append(curr_state)
        prev_state = curr_state
    return ret_state_seq


def is_loss_given_channel_state(channel_state):
    ret_val = False
    loss = np.random.choice([1, 0], p=[packet_loss_prob[channel_state], 1 - packet_loss_prob[channel_state]])
    if loss:
        ret_val = True
    return ret_val


def get_bursty_packet_loss_img_seq(img_name_seq, start_state=0):
    loss_img_seq = []
    is_loss_seq = []
    channel_state_seq = get_channel_state_via_2_state_markov_model(len(img_name_seq), start_state)
    for idx, channel_state in enumerate(channel_state_seq):
        is_loss = is_loss_given_channel_state(state_def[str(channel_state)])
        is_loss_seq.append(is_loss)
        if is_loss:
            loss_img_seq.append(loss_img_seq[idx-1])
        else:
            loss_img_seq.append(img_name_seq[idx])
    return is_loss_seq, loss_img_seq


def get_sub_folder_list(a_dir):
    return sorted([os.path.join(a_dir,name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))])


def get_png_files_in_dir(a_dir):
    return sorted([os.path.join(a_dir,name) for name in os.listdir(a_dir)
            if name.lower().endswith('png')])


def get_all_png_from_sub_dirs(a_dir):
    img_list=[]
    sub_dir_list = get_sub_folder_list(a_dir)
    for sub_dir in sub_dir_list:
        img_list += get_png_files_in_dir(sub_dir)
    return img_list


def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def write_stat(stat_line):
    with open(stat_file, 'a+') as f:
        f.write(stat_line+'\n')


def create_lossy_dataset(original_seq, lossy_seq, is_loss_seq):
    for idx, lossy_img_name in enumerate(tqdm(lossy_seq)):
        src_file_name = lossy_img_name
        dest_file_name = original_seq[idx].replace(img_dir, lossy_img_dir)
        stat_line = "{}, {}, {}".format(is_loss_seq[idx], src_file_name, dest_file_name)
        # print(stat_line)
        write_stat(stat_line)
        copy_file(src_file_name, dest_file_name)
# print("----random packet loss----")
# for _ in range(seq_length):
#     print("packet loss [{}]".format(is_loss_given_channel_state('random')))
#
# print("----bursty packet loss----")
# state_seq = get_channel_state_via_2_state_markov_model(seq_length, start_state)
# for state in state_seq:
#     print("channel_state :[{}] -> packet loss [{}]".format(state_def[str(state)], is_loss_given_channel_state(state_def[str(state)])))

write_stat("is_loss, src, dest\n")

recon_img_list = get_all_png_from_sub_dirs(img_dir)
is_loss_seq, lossy_recon_img_seq = get_bursty_packet_loss_img_seq(recon_img_list)

create_lossy_dataset(recon_img_list, lossy_recon_img_seq, is_loss_seq)

print("num of packet loss :{}".format(sum(is_loss_seq)))
print("that's all folks")
