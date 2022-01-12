import numpy as np
import pickle

pickle_file_name = "working/keypoints.pkl"
key_points = np.load("working/keypoints.npy", allow_pickle=True)
cpu_kp_list = []

for kp in key_points:
	cpu_kp = (kp["value"].cpu().numpy()[0])
	cpu_kp_list.append(cpu_kp)
#	np_kp = kp["value"].cpu().numpy()
#	print(np_kp)
#	print(np_kp[0], np.kp[1])
with open(pickle_file_name, 'wb') as pf:
	pickle.dump(cpu_kp_list, pf)

print("that's all folks")

