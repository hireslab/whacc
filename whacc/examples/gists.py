## copy the labels to a single array
from whacc import utils
import numpy as np
import h5py
from natsort import os_sorted
import copy

label_files = os_sorted(utils.get_files("/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/", "*_labels.npy"))
label_files = utils.lister_it(label_files, keep_strings=None, remove_string='curated')

all_labels = []
for k in label_files:
    all_labels.append(np.load(k)[0])

h5_file = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000.h5"
with h5py.File(h5_file, 'r') as hf:
    frame_nums = copy.deepcopy(hf['trial_nums_and_frame_nums'][1, :])

new_labels = np.zeros(int(np.sum(frame_nums))).astype(int)
for ii, (i1, i2) in enumerate(utils.loop_segments(frame_nums)):
    print(ii)
    for k in all_labels[ii]:
        new_labels[i1:i2][int(k-1)] = 1


h5_file = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000.h5"
with h5py.File(h5_file, 'r+') as hf:
    hf['labels'][:] = new_labels

##################
# plot random touch onsets
import matplotlib.pyplot as plt
from whacc import image_tools

all_onsets = np.where(np.diff(new_labels) == 1)[0]+1
add2 = 2
h5_file = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000.h5"
num_plots = 10
with h5py.File(h5_file, 'r') as hf:
    for k in np.random.choice(all_onsets, num_plots):
        plt.figure()
        plt.imshow(image_tools.img_unstacker(hf['images'][k-add2 : k+add2+1], num_frames_wide=1+add2*2))
        print(k-add2,  k+add2+1, k)


#######
# make a new reduced sized only near touches file with only one image deep
from whacc import utils
from whacc import image_tools
import copy
import h5py
f = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000.h5'
f2 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_80border_single.h5'

h5c = image_tools.h5_iterative_creator(f2, overwrite_if_file_exists=True, color_channel=False)

with h5py.File(f, 'r') as h:
    print(len(h['labels'][:]))
    h_cont = copy.deepcopy(h['labels'][:])
utils.create_master_dataset(h5c, [f], [h_cont], borders=80, max_pack_val=100)






# everything you need to make the below plot is here
# plot the histogram for the comparing the error types
a = []
for k in list(itertools.permutations(range(4), 2)):
  a.append(analysis.error_analysis(h_cont_1[k[1], :], h_cont_1[k[0], :], frame_num_array=frame_num_array.astype(int)))

def get_error_str_list(a):
  all_errors = []
  for k in a.error_neg:
    all_errors.append(a.type_list[k])
  for k in a.error_pos:
    all_errors.append(a.type_list[k])
  return all_errors

a_dict = dict()
a_dict['Count'] = []
a_dict['Error type'] = []
a_dict['pairs'] = []
# a_dict['count'] = []
for i, k in enumerate(list(itertools.permutations(range(4), 2))):
  if 3 in k:
    out = get_error_str_list(a[i])
    out = out+['append', 'deduct', 'ghost', 'join', 'miss', 'split']
    keys, counts = np.unique(out, return_counts=True)
    counts = counts-1
    a_dict['Count'] += list(counts)
    a_dict['Error type'] += list(keys)
    a_dict['pairs'] += list(np.tile('pairs '+str(k), len(keys)))

for i, k in enumerate(list(itertools.permutations(range(4), 2))):
  if 3 not in k:
    out = get_error_str_list(a[i])
    out = out+['append', 'deduct', 'ghost', 'join', 'miss', 'split']
    keys, counts = np.unique(out, return_counts=True)
    counts = counts-1
    a_dict['Count'] += list(counts)
    a_dict['Error type'] += list(keys)
    a_dict['pairs'] += list(np.tile('pairs '+str(k), len(keys)))


import pandas as pd
import datetime
import seaborn as sns

df = pd.DataFrame.from_dict(a_dict)
plt.figure(figsize=(15, 10))
# plt.ylim([0, 300])
# plt.yscale("log")
sns.barplot(x="Error type", hue="pairs", y="Count", data=df)

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""1$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$make a very small onset and offset extracot!!$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""
som enotes might have to do this for eeach weird extracted lag and lad diff...check this
"""

from whacc import utils, image_tools, analysis
import numpy as np
import h5py


all_h5s_imgs = ['/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_using_pole_tracker.h5']
h_cont = image_tools.get_h5_key_and_concatenate(all_h5s_imgs, 'labels')
h5c = image_tools.h5_iterative_creator('/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_5border_single_frame_onset_offset_for_aug.h5',
                                           overwrite_if_file_exists = True,
                                           color_channel = False)

h_cont_onset_offset = np.append(0, np.abs(np.diff(h_cont))) # to get jsut the onsets and ofsets not necesarilly the middles
utils.create_master_dataset(h5c, all_h5s_imgs, [h_cont_onset_offset], borders=5, max_pack_val=100)

utils.print_h5_keys(h5c.h5_full_file_name)
corrected_labels = []
with h5py.File(h5c.h5_full_file_name, 'r+') as h:
    for k in h['inds_extracted'][:]:
        corrected_labels.append(h_cont[k])
    h['labels'][:] = corrected_labels

tmp1 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_5border_single_frame_onset_offset_for_aug.h5'
a = analysis.pole_plot(tmp1, true_val=image_tools.get_h5_key_and_concatenate([tmp1]))
a.next()

"""2$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$ make the different types of h5 images and labels $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
base_dir_all_h5s = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/single_frame/'
utils.make_all_H5_types(base_dir_all_h5s)  # auto generate all the h5 types using a single set of flat (no color) image H5
utils.make_alt_labels_h5s(base_dir_all_h5s)  # auto generate the different types of labels

"""3$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$trim and correct the edges of the small subset $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
to_convert = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/regular/AH0000x000000_5border_single_frame_onset_offset_for_aug_regular.h5'
to_convert = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/3lag_diff/AH0000x000000_5border_single_frame_onset_offset_for_aug_3lag_diff.h5'
to_convert = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/3lag/AH0000x000000_5border_single_frame_onset_offset_for_aug_3lag.h5'
utils.print_h5_keys(to_convert)
xxx = to_convert.split('.h5')[0] + '_remove_2_edges' +'.h5'
h5c = image_tools.h5_iterative_creator(xxx,
                                       overwrite_if_file_exists = True,
                                       color_channel = True)
from tqdm import tqdm
border_bash = 2
with h5py.File(to_convert, 'r') as h:
    multiplier = h['multiplier'][:]
    frame_nums = []
    for i1, i2 in tqdm(utils.loop_segments(h['frame_nums'][:])):
        i1 += border_bash
        i2 -= border_bash
        h5c.add_to_h5(h['images'][i1:i2], h['labels'][i1:i2])
        frame_nums.append(i2-i1)
with h5py.File(h5c.h5_full_file_name, 'r+') as h:
    h.create_dataset('frame_nums', shape=np.shape(frame_nums), data=frame_nums)
    # print(h['multiplier'][:])
    # h.create_dataset('multiplier', shape=np.shape(multiplier), data=multiplier)

a = image_tools.get_h5_key_and_concatenate([to_convert], key_name='frame_nums')
b = image_tools.get_h5_key_and_concatenate([xxx], key_name='frame_nums')
from whacc import analysis

labels = image_tools.get_h5_key_and_concatenate([xxx], key_name='labels')
c = analysis.pole_plot(xxx, true_val=labels)
c.plot_it()


"""3$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ make all the augmented files  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""


from whacc import utils
from whacc import image_tools
from keras.preprocessing.image import ImageDataGenerator
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from imgaug import augmenters as iaa  # optional program to further augment data
h5_subset_file = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/3lag/AH0000x000000_5border_single_frame_onset_offset_for_aug_3lag_remove_2_edges.h5'
for h5_subset_file in utils.get_files('/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/', '*remove_2_edges.h5'):
    datagen = ImageDataGenerator(rotation_range=360,  #
                                 width_shift_range=.1,  #
                                 height_shift_range=.1,  #
                                 shear_range=.00,  #
                                 zoom_range=.25,
                                 brightness_range=[0.2, 1.2])  #

    with h5py.File(h5_subset_file, 'r') as hf:
        test_img = hf['images'][2]  # grab a single image
    num_aug = 20

    aug_ims, _ = image_tools.augment_helper(datagen, num_aug, 0 ,test_img, -1)  # make 99 of the augmented images and output 1 of th original
    # '-1' is the label associate ith the image, we dont care becasue we just want the iamge so it doenst matter what it is here

    # plt.figure(figsize=[5, 5])
    # plt.imshow(image_tools.img_unstacker(aug_ims, 10), )
    # plt.show()

    gaussian_noise = iaa.AdditiveGaussianNoise(loc = 0, scale=3)
    aug_ims_2 = gaussian_noise.augment_images(aug_ims)
    # plt.figure(figsize=[10, 10])
    # plt.imshow(image_tools.img_unstacker(aug_ims_2, 10))
    # plt.show()

    num_aug = 1
    for ii in range(10):
        # once we are happy with our augmentation process we can make an augmented H5 file using class
        new_H5_file = h5_subset_file.split('.')[0] + '_AUG_' + str(ii) + '.h5' # create new file name based on teh original H5 name
        h5creator = image_tools.h5_iterative_creator(new_H5_file, overwrite_if_file_exists=True,
                                                     close_and_open_on_each_iteration=True, color_channel=True)
        utils.get_class_info(h5creator)
        with h5py.File(h5_subset_file, 'r') as hf:
            for image, label in zip(tqdm(hf['images'][:]), hf['labels'][:]):
                aug_img_stack, labels_stack = image_tools.augment_helper(datagen, num_aug, 0, image, label)
                aug_img_stack = gaussian_noise.augment_images(aug_img_stack) # optional
                h5creator.add_to_h5(aug_img_stack[:, :, :, :], labels_stack)

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  transfer the frame nums to the aug data    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

bd_all = ['/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/regular/',
          '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/3lag_diff/',
          '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/3lag/']
for bd in bd_all:
    h5_to_split_list = utils.get_files(bd, '*edges_AUG_*')
    frame_num_src = utils.get_files(bd, '*2_edges.h5')[0]
    for dest in h5_to_split_list:
        utils.copy_h5_key_to_another_h5(frame_num_src, dest, 'frame_nums', 'frame_nums')


"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  combine them into training and validation    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

from whacc import image_tools
from whacc import utils

bd_all = ['/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/regular/',
          '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/3lag_diff/',
          '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/3lag/']

for bd in bd_all:
    h5_to_split_list = utils.get_files(bd, '*edges_AUG_*')
    image_tools.split_h5_loop_segments(h5_to_split_list, [7, 3], [bd+'train', bd+'val'], chunk_size=1000, add_numbers_to_name=False,
                 disable_TQDM=False, set_seed = 0, color_channel=True)



"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$  make a final combination of the the aug and other files    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

from whacc import image_tools
from whacc import utils
bd_all = ['/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/regular/',
          '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/3lag_diff/',
          '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/3lag/']

to_combine = ['/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/regular',
              '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/3lag_diff/',
              '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/3lag/']

for bd, cmb in zip(bd_all, to_combine):
    l1 = utils.get_files(bd, '*train*')
    l2 = utils.get_files(cmb, '*train*')
    l2 = utils.lister_it(l2, remove_string = 'zip')
    image_tools.split_h5_loop_segments(l1+l2, [1], [bd+'train_all'], chunk_size=1000, add_numbers_to_name=False,
                 disable_TQDM=False, set_seed = 0, color_channel=True)

    l1 = utils.get_files(bd, '*val*')
    l2 = utils.get_files(cmb, '*val*')
    l2 = utils.lister_it(l2, remove_string = 'zip')
    image_tools.split_h5_loop_segments(l1+l2, [1], [bd+'val_all'], chunk_size=1000, add_numbers_to_name=False,
                 disable_TQDM=False, set_seed = 0, color_channel=True)

"""
USE THIS AFTER CREATING AUG IMAGES TO TRANSFER FRAME NUMS
all_aug = utils.get_files('/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/', 'AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_*.h5')
frame_num_src = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug.h5'
for k in all_aug:
     utils.copy_h5_key_to_another_h5(frame_num_src, k, 'frame_nums', 'frame_nums')

"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
from whacc import utils
all_h5s = utils.get_h5s('/Users/phil/Dropbox/HIRES_LAB/curation_for_auto_curator/finished_contacts/')
all_h5s_imgs = utils.get_h5s('/Users/phil/Dropbox/HIRES_LAB/curation_for_auto_curator/H5_data/')
h_cont = utils._get_human_contacts_(all_h5s)
h5c = image_tools.h5_iterative_creator('/Users/phil/Dropbox/HIRES_LAB/curation_for_auto_curator/holy_set_80_border_single_frame.h5',
                                       overwrite_if_file_exists = True,
                                       color_channel = False)
utils.create_master_dataset(h5c, all_h5s_imgs, h_cont, borders = 80, max_pack_val = 100)


"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ split data based on frame num segments!!$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

from whacc import image_tools

bd = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/'
h5_to_split_list = ['/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_80border_single_frame.h5']
image_tools.split_h5_loop_segments(h5_to_split_list, [7, 3], [bd+'train', bd+'val'], chunk_size=1000, add_numbers_to_name=False,
             disable_TQDM=False, set_seed = 0, color_channel=False)



bd = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/'
h5_to_split_list = ['/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_80border_single_frame.h5',
                    '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_9.h5',
                    '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_8.h5',
                    '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_7.h5',
                    '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_6.h5',
                    '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_5.h5',
                    '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_4.h5',
                    '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_3.h5',
                    '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_2.h5',
                    '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_1.h5',
                    '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_0.h5']

image_tools.split_h5_loop_segments(h5_to_split_list, [7, 3], [bd+'train', bd+'val'], chunk_size=1000, add_numbers_to_name=False,
             disable_TQDM=False, set_seed = 0, color_channel=False)

a = bd+'val'
a = a+'.h5'
a = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/regular/train_regular.h5'
a = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/single_frame/train.h5'
# a = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/3lag_diff/val_3lag_diff.h5'
a = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/3lag/val_3lag.h5'
import h5py
with h5py.File(a, 'r') as h:
    print(h['images'][:].shape)
    print(h['labels'][:].shape)
    print(sum(h['frame_nums'][:]))

a = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/ALT_LABELS/val_ALT_LABELS.h5'
with h5py.File(a, 'r') as h:
    print(h['[0, 1, 2, 3, 4, 5]- (no touch, touch, onset, one after onset, offset, one after offset)'][:].shape)
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  make the mini holy test set for the   $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
from whacc import image_tools, utils
bd = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/10_percent_holy_set/data/single_frame/'
h5_to_split_list = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/full_holy_set/single_frame/holy_set_80_border_single_frame.h5'
image_tools.split_h5_loop_segments([h5_to_split_list], [1, 9], [bd+'holy_test_set_10_percent', [None]], chunk_size=1000, add_numbers_to_name=False,
             disable_TQDM=False, set_seed = 0, color_channel=False)
utils.print_h5_keys('/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/10_percent_holy_set/data/single_frame/holy_test_set_10_percent.h5')


"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  plot that shit   $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

import matplotlib.pyplot as plt
a = info_dict['h5_train']
labels = image_tools.get_h5_key_and_concatenate([a])
bplt = analysis.pole_plot(a, true_val = labels)
points_of_interest = np.where(np.diff(labels)==1)[0]
cnt = -1


cnt+=1
bplt.current_frame = points_of_interest[cnt]-4
bplt.plot_it()
plt.ylim([-.5, 6])







from whacc import image_tools
from whacc import analysis

xx = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/regular/train_regular.h5'
a = analysis.pole_plot(xx)
a.plot_it()
fn = image_tools.get_h5_key_and_concatenate([xx], 'frame_nums')