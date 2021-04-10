import h5py
from whacc import utils

import matplotlib.pyplot as plt
import numpy as np

H5_list = utils.get_h5s('/Users/phil/Dropbox/Autocurator/testing_data/MP4s/')

H5_FILE = H5_list[0]
with h5py.File(H5_FILE, 'r') as hf:
    for k in hf.keys():
        print(k)
    print(hf['trial_nums_and_frame_nums'][:])


# with h5py.File(H5_FILE, 'r') as hf:
#     plt.plot(hf['in_range'][:])
#     print(hf['in_range'][:])


with h5py.File(H5_FILE, 'r') as hf:
    # for k in hf['trial_nums_and_frame_nums'][1, :]:
    cumsum_frames = np.concatenate((np.asarray([0]), np.cumsum(hf['trial_nums_and_frame_nums'][1, :])))
    tot_frames = np.sum(hf['trial_nums_and_frame_nums'][1, :])
start_pole = 2000
stop_pole = 3000

b = np.vstack((start_pole+cumsum_frames[:-1], cumsum_frames[1:]-1)).astype('int')
b = np.min(b, axis = 0)
a = np.vstack((stop_pole+cumsum_frames[:-1], cumsum_frames[1:])).astype('int')
a = np.min(a, axis = 0)

keep_mask = np.zeros(tot_frames.astype('int'))
for k1, k2 in zip(b, a):
    keep_mask[k1:k2] = 1
plt.plot(keep_mask)
