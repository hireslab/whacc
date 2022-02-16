# from whacc import image_tools
#
# image_tools.h5_iterative_creator()


from whacc import utils
from whacc import image_tools
import time
from keras.preprocessing.image import ImageDataGenerator
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

#
# # # $$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # plt.figure(figsize=[5, 10])
# # with h5py.File('/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/AH0667x170317_subset_AUG.h5', 'r') as hf:
# #     for k in range(20):
# #         print(k)
# #         plt.imshow(image_tools.img_unstacker(hf['images'][30*k: 30*(k+1)], 5))
# #         plt.pause(.5)
# #         plt.show()
#
#
#
# # $$$$$$$$$$$$$$$$$$$$$$$$$$$$
# image_tools.split_h5('/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/AH0667x170317.h5', [9, 1],
#                        '/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/AH0667x170317_temp_')
# # $$$$$$$$$$$$$$$$$$$$$$$$$$$$
from whacc import image_tools
# help(image_tools.predict_multiple_H5_files)

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$
h5_name = '/Users/phil/Dropbox/Autocurator/moving_pole_video/AH0001x999999.h5'
h5_name = '/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/AH0667x170317.h5'
plt.figure(figsize= [6, 6])
with h5py.File(h5_name, 'r') as hf:
    print(hf.keys())
    tmp1 = hf['max_val_stack'][:].copy()
    plt.plot(tmp1[2])

    for i, k in enumerate(hf['images'][:]):
        if i>500:
            plt.imshow(k)
            plt.pause(1/120)

            print(i)
    plt.show()


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$
import os
from whacc import utils
import pyment
a = utils.get_files('/Users/phil/Dropbox/HIRES_LAB/GitHub/whacc/whacc/', '*.py')
for k in a:
    k
    os.system("pyment -w -o numpydoc " + k)

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$

import matplotlib.pyplot as plt
import h5py
import numpy as np
h5_file = '/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/AH0667x170317.h5'
h5_file = '/Users/phil/Downloads/AH1157X22012021xS408 (1).h5'
h5_file = "/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AAA_test_tracking/AH1157X22012021xS408.h5"

with h5py.File(h5_file, 'r') as hf:
    print(hf.keys())
    last_range = 0
    for i, k in enumerate(hf['max_val_stack'][:]):
        y = k+last_range-min(k)
        plt.plot(y)
        plt.annotate(str(i), (-100,y[0]))
        # plt.plot(k, np.zeros_like(k)+last_range)
        last_range += np.ptp(k)*1.2
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$
import matplotlib.pyplot as plt
import h5py
import numpy as np
h5_file = "/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AAA_test_tracking/AH1157X22012021xS408.h5"
with h5py.File(h5_file, 'r') as hf:
    for i, k in enumerate(hf['images'][:]):
        if i%30 == 0:
            plt.imshow(k)
            a = plt.text(0, 0, str(i), bbox=dict(fill=True, edgecolor='white', linewidth=2))
            plt.pause(.1)
    plt.show()
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$
from whacc import utils

h5_file_list = utils.get_h5s('/Users/phil/Dropbox/Autocurator/testing_data/MP4s/')

h5_file_list2 = utils.lister_it(h5_file_list, remove_string=['subset', 'temp', 'aug'])
print(h5_file_list2)
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$
import matplotlib.pyplot as plt
import h5py
import copy
import numpy as np
h5_file = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/master_train_test1.h5"
with h5py.File(h5_file, 'r') as hf:
    # plt.plot(hf['labels'][:8000])
    # print(len(hf['labels'][:]))
    #
    # print(hf['images'][0].shape)

    labels2insert = copy.deepcopy(hf['labels'][:])

    for k in range(0, 1163998, 40000):
        plt.figure()
        plt.imshow(hf['images'][k])


h5_file = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000.h5"
with h5py.File(h5_file, 'r') as hf:
    print(len(hf['labels'][:]))
    hf['labels'][:] = labels2inser


all_onsets = np.where(np.diff(labels2insert) == 1)[0]
add2 = 1
h5_file = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000.h5"
########### h5_file = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/master_train_test1.h5"

with h5py.File(h5_file, 'r') as hf:
    for k in np.random.choice(all_onsets, 20):
        plt.figure()
        plt.imshow(image_tools.img_unstacker(hf['images'][k-add2 : k+add2+1], num_frames_wide=3))
##
##
##

##

f = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/py_touch_inds.mat"
a = utils.loadmat(f)
all_inds = a['all_inds']
all_vid_inds = a['all_vid_inds']


h5_file = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000.h5"
with h5py.File(h5_file, 'r') as hf:
    frame_nums = copy.deepcopy(hf['trial_nums_and_frame_nums'][1, :])

new_labels = np.zeros(int(np.sum(frame_nums))).astype(int)
for ii, (i1, i2) in enumerate(utils.loop_segments(frame_nums)):
    a = np.where(all_vid_inds==ii)[0]
    print(ii)
    for k in all_inds[a[0]:a[-1]]:
        new_labels[k] = 1





###
