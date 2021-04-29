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

