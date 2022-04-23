from whacc import image_tools
from datetime import datetime

h5 = [
    '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/small_h5s/3lag/small_test_3lag.h5']
h5 = [
    '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/regular/train_regular.h5']

h5 = [
    '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/3lag/train_3lag.h5']

lstm_len = 5
batch_size = 100
h5_file_list = h5
G = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, h5_file_list, label_key='labels', IMG_SIZE=96)
start = datetime.now()
for k in range(G.__len__()):
    x, y = G.__getitem__(k)
print(datetime.now() - start)

lstm_len = 5
batch_size = 400
h5_file_list = h5
G = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, h5_file_list, label_key='labels', IMG_SIZE=96)
start = datetime.now()
for k in range(G.__len__()):
    x, y = G.__getitem__(k)
print(datetime.now() - start)

lstm_len = 5
batch_size = 800
h5_file_list = h5
G = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, h5_file_list, label_key='labels', IMG_SIZE=96)
start = datetime.now()
for k in range(G.__len__()):
    x, y = G.__getitem__(k)
print(datetime.now() - start)

lstm_len = 5
batch_size = 1200
h5_file_list = h5
G = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, h5_file_list, label_key='labels', IMG_SIZE=96)
start = datetime.now()
for k in range(G.__len__()):
    x, y = G.__getitem__(k)
print(datetime.now() - start)

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

multiplier_1 = ((8.33 + 3.59) / 8.33)
multiplier_1 * 8 * 200 / 60

from whacc import utils, image_tools
from datetime import datetime
import numpy as np
import h5py

# righthere
h5_file_list = [
    "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/small_h5s/data/3lag/small_train_3lag.h5"]
h5_file_list = ['/Users/phil/Dropbox/Colab data/H5_data/regular/AH0698_170601_PM0121_AAAA_regular.h5']
lstm_len = 7
b = lstm_len // 2
batch_size = 4000
G = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, h5_file_list, label_key='labels', IMG_SIZE=96)
start = datetime.now()
new_h5 = '/Users/phil/Desktop/LSTM_small_train.h5'
h5creator = image_tools.h5_iterative_creator(new_h5)
for k in range(3):  # the issue here is that I created teh H5 file with the 3 lag with the noise image background to
    # that I didnt have to do it in the generator, but for this I dont do that so I create and image and it overlaps
    # making it so that the end of one trial merges with the next trial, this isnt too hard to fix
    # best to fix this after the file has been generated
    print(k)
    x, y = G.__getitem__(k)
    x = ((x + 1) / 2) * 255  # I trasnform it here no worries
    print(np.min(x), np.max(x))

    h5creator.add_to_h5(x, y)
h5creator.close_h5()
print(datetime.now() - start)
utils.copy_h5_key_to_another_h5(h5_file_list[0], new_h5, 'frame_nums')
# h5 = '/Users/phil/Desktop/LSTM_small_train.h5'
# with h5py.File(h5, 'r+') as h:
#     h['frame_nums'] = h['frame_nums'][:3]
h5 = '/Users/phil/Desktop/LSTM_small_train.h5'
"""%$%$%$%$%$%$%$%___ black out edges of the each trials start and end ___%$%$%$%$%$%$%$%"""

with h5py.File(h5, 'r+') as h:
    for i1, i2 in utils.loop_segments(h['frame_nums']):
        x = h['images'][i1:i2]
        print(x.shape)
        edge_ind = np.flip(np.arange(1, b + 1))
        for i in np.arange(1, b + 1):
            x[i - 1, :edge_ind[i - 1], ...] = np.zeros_like(x[i - 1, :edge_ind[i - 1], ...])
            x[-i, -edge_ind[i - 1]:, ...] = np.zeros_like(x[-i, -edge_ind[i - 1]:, ...])
        h['images'][i1:i2] = x
"""%$%$%$%$%$%$%$%___ black out edges of the each trails start and end ___%$%$%$%$%$%$%$%"""
""" I could also make this directly work in the generator by brabing the inds of these black ones and making them
black based on that when they come up"""
lstm_len = 5
b = lstm_len // 2
with h5py.File(h5,
               'r') as h:  # 0(0, 1) 1(0) 3998(-1) 3999(-2, -1) ...  4000(0, 1) 4001(0) 7998(-1) 7999(-2, -1) #0,0     0,1     1,0     3998,-1     3999,-2     3999,-1
    full_edges_mask = np.ones_like(h['images'][:b * 2])
    edge_ind = np.flip(np.arange(1, b + 1))
    for i in np.arange(1, b + 1):
        full_edges_mask[i - 1, :edge_ind[i - 1], ...] = np.zeros_like(full_edges_mask[i - 1, :edge_ind[i - 1], ...])
        full_edges_mask[-i, -edge_ind[i - 1]:, ...] = np.zeros_like(full_edges_mask[-i, -edge_ind[i - 1]:, ...])
    all_edges = []
    for i1, i2 in utils.loop_segments(h['frame_nums']):  # 0, 1, 3998, 3999 ; 4000, 4001, 7998, 7999; ...
        edges = (np.asarray([[i1], [i2 - b]]) + np.arange(0, b).T).flatten()
        all_edges.append(edges)
    all_edges = np.asarray(all_edges)

adjust_these_edge_frames = np.intersect1d(all_edges.flatten(), np.arange(i1, i2 + 1))
for atef in adjust_these_edge_frames:
    mask_ind = np.where(atef == all_edges)[1][0]
    out[mask_ind] = out[mask_ind] * full_edges_mask[mask_ind]

np.asarray([[i1], [i2]]) + np.arange(0, b)

print(start_mask.shape, end_mask.shape)
# h['images'][i1:i2] = x
"""%$%$%$%$%$%$%$%___ black out edges of the each trails start and end ___%$%$%$%$%$%$%$%"""

"""%$%$%$%$%$%$%$%"""
h5 = '/Users/phil/Desktop/LSTM_small_train.h5'
import h5py
import matplotlib.pyplot as plt

utils.print_h5_keys(h5)
with h5py.File(h5, 'r') as h:
    print(h['images'].shape)
    print(h['frame_nums'][:])
    x = h['images'][0:8000]

# x = (x+1)/2
for ind1 in [3996, 3997, 3998, 3999, 4000, 4001, 4002]:
    y = x[ind1]
    plt.figure()
    for i, k in enumerate(y):
        print(all(k.flatten() == 0))
        plt.subplot(3, 3, i + 1)

        # plt.figure()
        plt.imshow(k)

h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/small_h5s/data/3lag/small_train_3lag.h5'
utils.print_h5_keys(h5)
with h5py.File(h5, 'r') as h:
    print(h['frame_nums'][:])

from whacc import image_tools

h5_file_list = [
    '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/small_h5s/data/3lag/small_train_3lag.h5']
lstm_len = 5
batch_size = 100
G = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, h5_file_list, label_key='labels', IMG_SIZE=96)
for k in range(G.__len__()):
    x, y = G.__getitem__(k)
    print(x.shape, y.shape)

"""asdkfja;lskdjfkl;asjdf;lkjas;ldkfjlaksdjfkl"""

from whacc import utils, image_tools
from datetime import datetime
import numpy as np
import h5py

h5_file_list = [
    "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/small_h5s/data/3lag/small_train_3lag.h5"]
# h5_file_list = ['/Users/phil/Dropbox/Colab data/H5_data/regular/AH0698_170601_PM0121_AAAA_regular.h5']
lstm_len = 5
batch_size = 125
G = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, h5_file_list, label_key='labels', IMG_SIZE=96)
start = datetime.now()
new_h5 = '/Users/phil/Desktop/LSTM_small_train.h5'
h5creator = image_tools.h5_iterative_creator(new_h5)
for k in range(
        G.__len__()):  # the issue here is that I created teh H5 file with the 3 lag with the noise image background to
    # that I didnt have to do it in the generator, but for this I dont do that so I create and image and it overlaps
    # making it so that the end of one trial merges with the next trial, this isnt too hard to fix
    # best to fix this after the file has been generated
    print(k)
    x, y = G.__getitem__(k)
    x = ((x + 1) / 2) * 255
    print(np.min(x), np.max(x))

"""apsdjf;lajsdlfk;jals;kdfj;laksdjflkasjdflkajsdlkfjalksdjfl;kajsdfkl;jakls;dfjlkasdjfkl"""
from whacc import image_tools

h5 = '/Users/phil/Desktop/LSTM_small_train.h5'
G = image_tools.ImageBatchGenerator_simple(1000, h5)
for k in range(G.__len__()):
    x, y = G.__getitem__(k)
    print(x.shape, y.shape)

import os


def copy_folder_structure(src, dst):
    def ignore_files(dir, files):
        new_dir = '/Users/phil/Desktop/aasdf/'
        for f in files:
            if os.path.isfile(os.path.join(dir, f)):
                if f[0] != '.':
                    # Path(new_dir).mkdir(parents=True, exist_ok=True)
                    print(f)
                    print(new_dir + f.split('.')[0])
                    # f = open(new_dir + f.split('.')[0]+'.txt', "w+")
                return f
        return
        # return [f for f in files if os.path.isfile(os.path.join(dir, f))]

        shutil.copytree(src, dst, ignore=ignore_files)


from pathlib import Path
import shutil
import os
from whacc import utils

from tqdm import tqdm


def copy_file_filter(src, dst, keep_strings='', remove_string=None, overwrite=False,
                     just_print_what_will_be_copied=False, disable_tqdm=False):
    src = src.rstrip(os.sep) + os.sep
    dst = dst.rstrip(os.sep) + os.sep

    all_files_and_dirs = utils.get_files(src, search_term='*')
    to_copy = utils.lister_it(all_files_and_dirs, keep_strings=keep_strings, remove_string=remove_string)

    if just_print_what_will_be_copied:
        _ = [print(str(i) + ' ' + k) for i, k in enumerate(to_copy)]
        return
    to_copy2 = []  # this is so I can tqdm the files and not the folders which would screw with the average copy time.
    for k in to_copy:
        k2 = dst.join(k.split(src))
        if os.path.isdir(k):
            Path(k2).mkdir(parents=True, exist_ok=True)
        else:
            to_copy2.append(k)

    for k in tqdm(to_copy2, disable=disable_tqdm):
        k2 = dst.join(k.split(src))
        if overwrite or not os.path.isfile(k2):
            Path(os.path.dirname(k2)).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(k, k2)
        elif not overwrite:
            print('overwrite = False: file exists, skipping--> ' + k2)


copy_file_filter('/Users/phil/Desktop/FAKE_full_data', '/Users/phil/Desktop/aaaaaaaaaa', keep_strings='/3lag/',
                 remove_string=None, overwrite=True, just_print_what_will_be_copied=False)


from whacc import utils
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
all_template_images = utils.get_files('/Users/phil/Desktop/trashTMP/', '*.png')

for i, k in enumerate(all_template_images):
    img = mpimg.imread(k)
    if i%3**2 == 0:
        fig, axs = plt.subplots(3, 3, figsize=[10, 10])
        axs = axs.flatten()
    axs[i%3**2].imshow(img)


np.arange(20)**2
