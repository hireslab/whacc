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
# h5_subset_file = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/3lag/AH0000x000000_5border_single_frame_onset_offset_for_aug_3lag_remove_2_edges.h5'
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

def label_naming_shorthand_dict(name_key=None):
    label_naming_shorthand_dict = {
        '[0, 1, 2, 3, 4, 5]- (no touch, touch, onset, one after onset, offset, one after offset)': 'on-off_set_and_one_after',
        '[0, 1, 2, 3]- (no touch, touch, onset, offset': 'on-off_set',
        '[0, 1, 2]- (no event, onset, offset)': 'only_on-off_set',
        '[0, 1]- (no touch, touch)': 'regular',
        '[0, 1]- (not offset, offset)': 'only_offset',
        '[0, 1]- (not onset, onset)': 'only_onset',
        '[0, 1, 3, 4]- (no touch, touch, one after onset, offset)': 'overlap_whisker_on-off'}
    if name_key is None:
        return label_naming_shorthand_dict
    else:
        return label_naming_shorthand_dict[name_key]
def copy_over_new_labels(label_key_name, image_h5_list, label_h5_list):
    label_key_shorthand = label_naming_shorthand_dict(label_key_name)
    for img_src, lab_src in zip(image_h5_list, label_h5_list):
        utils.copy_h5_key_to_another_h5(lab_src, img_src, label_key_name, 'labels')

from whacc import image_tools
from whacc import utils
bd_all = ['/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/regular/',
          '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/3lag_diff/',
          '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/3lag/']

to_combine = ['/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/regular',
              '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/3lag_diff/',
              '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/3lag/']

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
""" make sure the labels are regular before combining them with the aug set"""
k = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/ALT_LABELS'
alt_labels = utils.get_files(k, '*.h5')

for k in to_combine:
    a = utils.get_files(k, '*.h5')
    a = utils.lister_it(a, remove_string = 'zip')
    print(a)
    copy_over_new_labels('[0, 1]- (no touch, touch)', a, alt_labels)

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

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$# finally after copying them to this directory we can make the labels with any of the data directories$$$$$$$"""
# be careful with this cause if the regular frames is not the normal one then your labels will be all screwed up
# need to run below code above first
'''
for k in to_combine:
    a = utils.get_files(k, '*.h5')
    a = utils.lister_it(a, remove_string = 'zip')
    print(a)
    copy_over_new_labels('[0, 1]- (no touch, touch)', a, alt_labels)
'''
utils.make_alt_labels_h5s('/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/regular')

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$# update all the labels in case you create a new label """
utils.make_alt_labels_h5s('/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/small_h5s/data/single_frame')
utils.make_alt_labels_h5s('/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/single_frame')
utils.make_alt_labels_h5s('/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/small_h5s/single_frame')
utils.make_alt_labels_h5s('/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/10_percent_holy_set/single_frame')
utils.make_alt_labels_h5s('/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/full_holy_set/single_frame')

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


"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$  transfer all model and folders form google cloud machine to google bucket  $$$$$$$$$$$$$$$$$$$$$$$$"""

from google.cloud import storage # !pip install --upgrade google-cloud-storage
import os
from whacc import utils
%cd "/home/jupyter"

client = storage.Client()
storage_client = storage.Client.from_service_account_json('phils-whisker-contacts-7eea4efe3802.json')
bucket = client.get_bucket('whacc_multi_model_test_data')

def transfer_data(bucket):
    blobs=list(bucket.list_blobs(prefix='SAVED_MODELS'))
    cloud_folders = list(set([os.path.dirname(str(k.name)) for k in blobs]))
    a = utils.get_files('model_testing', '*info_dict.json')
    for k in a:
        to_copy = os.path.dirname(k)
        if not any([to_copy in kk for kk in cloud_folders]):
            print(to_copy)
            dest = 'gs://whacc_multi_model_test_data/'+'SAVED_MODELS/'+to_copy
            !gsutil -m cp -r $to_copy $dest
transfer_data(bucket)
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$  reduce to single frames  $$$$$$$$$$$$$$$$$$$$$$$$"""

from pathlib import Path
from whacc import utils, image_tools
import os
from tqdm import tqdm

h5files = utils.get_h5s('/Users/phil/Dropbox/Colab data/H5_data')
for k1 in h5files:
    # print(k.split('__')[0])
    k = '-'.join(k1.split('__'))
    k = k.split('-')[0]
    k = os.path.dirname(k) + os.sep + 'single_frame' + os.sep + os.path.basename(k) + '.h5'
    Path(os.path.dirname(k)).mkdir(parents=True, exist_ok=True)
    utils.reduce_to_single_frame_from_color(k1, k)

utils.make_all_H5_types('/Users/phil/Dropbox/Colab data/H5_data/single_frame/')

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$run in google colab$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$  needs search_sequence_numpy from colab, adds all the correct labels to the H5  $$$$$$$$$$$$$$$$$$$$$$$$"""

from google.colab import drive
import numpy as np
import h5py
import os
from whacc import utils

drive.mount('/content/gdrive')

base_dir = '/content/gdrive/My Drive/Colab data/curation_for_auto_curator/H5_data/'
h5_list_to_write = utils.get_h5s(base_dir)
all_h5s = utils.get_h5s('/content/gdrive/My Drive/Colab data/curation_for_auto_curator/finished_contacts/',
                        print_h5_list=False)
h_cont, h_names = utils._get_human_contacts_(all_h5s)
print('--------\npercent fully agree\n--------')
for k in h_cont:
    a = np.mean(np.mean(k, axis=0) == 1)
    b = np.mean(np.mean(k, axis=0) == 0)
    print(np.round(a + b, 4))


def add_to_h5(h5_file, key, values, overwrite_if_exists=False):
    all_keys = utils.print_h5_keys(h5_file, return_list=True, do_print=False)
    with h5py.File(h5_file, 'r+') as h:
        if key in all_keys and overwrite_if_exists:
            print('key already exists, overwriting value...')
            del h[key]
            h.create_dataset(key, data=values)
        elif key in all_keys and not overwrite_if_exists:
            print("""key already exists, NOT overwriting value..., \nset 'overwrite_if_exists' to True to overwrite""")
        else:
            h.create_dataset(key, data=values)


for kk, h5 in zip(h_cont, all_h5s):
    F_NAME = os.path.basename(h5).split('Phil_')[-1][:-3]
    h52write = utils.lister_it(h5_list_to_write, keep_strings=F_NAME)[0]
    print(h52write)

    avg_cont = (np.mean(kk, axis=0) > .5) * 1
    tmp1 = search_sequence_numpy(avg_cont, np.asarray([0, 1, 0]))
    for k in tmp1 + 1:
        avg_cont[k] = 0

    tmp1 = search_sequence_numpy(avg_cont, np.asarray([1, 0, 1]))
    for k in tmp1 + 1:
        avg_cont[k] = 1

    tmp1 = search_sequence_numpy(avg_cont, np.asarray([0, 1, 1, 0]))
    for k in tmp1 + 1:
        avg_cont[k] = 0
    for k in tmp1 + 2:
        avg_cont[k] = 0

    tmp1 = search_sequence_numpy(avg_cont, np.asarray([1, 0, 0, 1]))
    for k in tmp1 + 1:
        avg_cont[k] = 1
    for k in tmp1 + 2:
        avg_cont[k] = 1

    tmp1 = search_sequence_numpy(avg_cont, np.asarray([0, 1, 0]))
    assert tmp1.size == 0
    tmp1 = search_sequence_numpy(avg_cont, np.asarray([1, 0, 1]))
    assert tmp1.size == 0
    tmp1 = search_sequence_numpy(avg_cont, np.asarray([0, 1, 1, 0]))
    assert tmp1.size == 0
    tmp1 = search_sequence_numpy(avg_cont, np.asarray([1, 0, 0, 1]))
    assert tmp1.size == 0

    utils.add_to_h5(h52write, 'labels', avg_cont, overwrite_if_exists=True)

utils.make_alt_labels_h5s(base_dir)
alt_label_dir = '/content/gdrive/My Drive/Colab data/curation_for_auto_curator/ALT_LABELS/'

to_pred_h5s = '/content/gdrive/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED'
to_pred_h5s = utils.get_h5s(to_pred_h5s)

tmp1 = utils.get_h5s(alt_label_dir)
utils.print_h5_keys(tmp1[0])
for h5, h5_dest in zip(tmp1, to_pred_h5s):
    assert os.path.basename(h5) == os.path.basename(h5_dest)
    keys = utils.print_h5_keys(h5, return_list=True, do_print=False)
    for key in keys:
        utils.copy_h5_key_to_another_h5(h5, h5_dest, key)


"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$  use color map to plot the predicitons with the images   $$$$$$$$$$$$$$$$$$$$$$$$"""

color_dict = dict()

cmap = cm.get_cmap('inferno')
color_list = [0, .2, .4, .5, .6, .8]
for i, k1 in enumerate(color_list):
  color_dict[i] = np.asarray(cmap(k1)[:-1])*255

# color_dict[0] =np.asarray(cmap(0)[:-1])*255 #(255, 230, 230)

# color_dict[2] = np.asarray(cmap(0.2)[:-1])*255#(0, 255, 153)
# color_dict[3] = np.asarray(cmap(0.4)[:-1])*255#(0, 255, 0)

# color_dict[1] = np.asarray(cmap(0.5)[:-1])*255#(51, 102, 0)

# color_dict[4] = np.asarray(cmap(0.6)[:-1])*255#(255, 153, 51)
# color_dict[5] = np.asarray(cmap(.8)[:-1])*255#(255, 51, 0)

img_width = 61
height = 20

tmp1, tmp2 = utils.group_consecutives(np.where(real_bool==1)[0])
border = 2
k +=1
pred_m = pred_m.astype(float)
inds = range(tmp1[k][0]-border, tmp1[k][-1]+1+border*2)
tmp1 = []

tmp1 = np.tile(np.repeat(pred_m[inds], img_width, axis=0), (height, 1))
tmp1 = np.vstack((tmp1, np.tile(np.repeat(real[inds], img_width, axis=0), (height, 1))))
tmp1 = np.vstack((tmp1, np.tile(np.repeat(pred_v[inds], img_width, axis=0), (height, 1))))


tmp1 = np.stack((tmp1,)*3, axis=-1)

for kk in np.unique(tmp1.astype(int)):
  tmp3 = np.where(tmp1 == kk)
  for i1, i2 in zip(tmp3[0], tmp3[1]):
    tmp1[i1, i2, :] = color_dict[kk]

tmp1 = tmp1.astype(int)
# tmp1 = np.round((tmp1/5)*255).astype(int)
# tmp1[:, :, 1] = .5
# tmp1[:, :, 0] = .5

plt.figure(figsize=(20, 10))
with h5py.File(actual_h5_img_file, 'r') as h:
  tmp2 = image_tools.img_unstacker(h['images'][inds[0]:inds[-1]+1], num_frames_wide=len(inds))
  tmp2 = np.vstack((tmp1, tmp2))
  plt.imshow(tmp2)



"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$  use color map to plot the predicitons with the images  VVVV222222 $$$$$$$$$$$$$$$$$$$$$$$$"""



def foo2(in_list_of_arrays = [], touch_number = 0 , border = 4, height = 20, img_width = 61,
         color_list = [0, .5, .2, .3, .75, .85], cmap_col = 'inferno'):
  # in_list_of_arrays[0] needs to be the "true"" values
  if in_list_of_arrays == []:
    print('no input arrays, returning...')
    return
  color_dict = dict()
  cmap = cm.get_cmap(cmap_col)

  for i, k1 in enumerate(color_list):
    color_dict[i] = np.asarray(cmap(k1)[:-1])*255


  in_list_of_arrays = copy.deepcopy(in_list_of_arrays)
  tmp1, tmp2 = utils.group_consecutives(np.where(in_list_of_arrays[0]!=0)[0])

  inds = list(range(tmp1[touch_number][0]-border, tmp1[touch_number][-1]+1+border*2))

  tmp1 = np.tile(np.repeat(pred_m[inds], img_width, axis=0), (height, 1))
  for i, k in enumerate(in_list_of_arrays):
    k = k.astype(float)
    if i == 0:
      tmp1 = np.tile(np.repeat(k[inds], img_width, axis=0), (height, 1))
    else:
      tmp1 = np.vstack((tmp1, np.tile(np.repeat(k[inds], img_width, axis=0), (height, 1))))
  tmp1 = np.stack((tmp1,)*3, axis=-1)

  for kk in np.unique(tmp1.astype(int)):
    tmp3 = np.where(tmp1 == kk)
    for i1, i2 in zip(tmp3[0], tmp3[1]):
      tmp1[i1, i2, :] = color_dict[kk]

  tmp1 = tmp1.astype(int)
  with h5py.File(actual_h5_img_file, 'r') as h:
    tmp2 = image_tools.img_unstacker(h['images'][inds[0]:inds[-1]+1], num_frames_wide=len(inds))
    tmp2 = np.vstack((tmp1, tmp2))
    return tmp2

tmp2 = foo2(in_list_of_arrays = [real, pred_m, pred_v], touch_number = 201, border = 4, height = 20, img_width = 61,
        color_list = [0, .5, .2, .3, .85, .95], cmap_col = 'inferno')
plt.figure(figsize=(20, 10))
plt.imshow(tmp2)



"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ get human agreed percentages $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

from whacc import utils
all_h5s = utils.get_h5s('/content/gdrive/My Drive/Colab data/curation_for_auto_curator/finished_contacts/', print_h5_list=False)
h_cont, h_names = utils._get_human_contacts_(all_h5s)
print('--------\npercent fully agree\n--------')
for k in h_cont:
  a = np.mean(np.mean(k, axis=0)==1)
  b = np.mean(np.mean(k, axis=0)==0)
  print(np.round(a+b, 4))


"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ replace h5 labels with average predictions $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
from whacc import utils
import numpy as np

base_dir = '/content/gdrive/My Drive/Colab data/curation_for_auto_curator/H5_data/'
h5_list_to_write = utils.get_h5s(base_dir)
all_h5s = utils.get_h5s('/content/gdrive/My Drive/Colab data/curation_for_auto_curator/finished_contacts/', print_h5_list=False)
h_cont, h_names = utils._get_human_contacts_(all_h5s)

for kk, h5 in zip(h_cont, all_h5s):
  F_NAME = os.path.basename(h5).split('Phil_')[-1][:-3]
  h52write = utils.lister_it(h5_list_to_write, keep_strings=F_NAME)[0]
  print(h52write)

  avg_cont = (np.mean(kk, axis = 0)>.5)*1
  tmp1 = utils.search_sequence_numpy(avg_cont, np.asarray([0, 1, 0]))
  for k in tmp1+1:
    avg_cont[k] = 0

  tmp1 = utils.search_sequence_numpy(avg_cont, np.asarray([1,0,1]))
  for k in tmp1+1:
    avg_cont[k] = 1

  tmp1 = utils.search_sequence_numpy(avg_cont, np.asarray([0,1,1,0]))
  for k in tmp1+1:
    avg_cont[k] = 0
  for k in tmp1+2:
    avg_cont[k] = 0

  tmp1 = utils.search_sequence_numpy(avg_cont, np.asarray([1,0,0,1]))
  for k in tmp1+1:
    avg_cont[k] = 1
  for k in tmp1+2:
    avg_cont[k] = 1

  # assert that these types of errors dont exist in the final version after correcting them above
  tmp1 = utils.search_sequence_numpy(avg_cont, np.asarray([0, 1, 0]))
  assert tmp1.size == 0
  tmp1 = utils.search_sequence_numpy(avg_cont, np.asarray([1,0,1]))
  assert tmp1.size == 0
  tmp1 = utils.search_sequence_numpy(avg_cont, np.asarray([0,1,1,0]))
  assert tmp1.size == 0
  tmp1 = utils.search_sequence_numpy(avg_cont, np.asarray([1,0,0,1]))
  assert tmp1.size == 0

  utils.add_to_h5(h52write, 'labels', avg_cont, overwrite_if_exists=True)



"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$ Fully automate the generation of all types data folder using a single H5 file (normal with "color" channels) $$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$ WITH AUGMENTATION $$$$$$$$$ WITH AUGMENTATION $$$$$$$$$ WITH AUGMENTATION $$$$$$$$$ WITH AUGMENTATION $$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

from whacc import utils
from whacc import image_tools
from keras.preprocessing.image import ImageDataGenerator
import h5py
import shutil
from imgaug import augmenters as iaa  # optional program to further augment data
subset_h5s = utils.get_h5s("/content/gdrive/MyDrive/Colab data/curation_for_auto_curator/RETRAIN_H5_data/")


def foo_get_frame_nums_from_all_inds(h5_subset_file):
  tmp1 = image_tools.get_h5_key_and_concatenate([h5_subset_file], 'all_inds')
  tmp1, tmp2 = utils.group_consecutives(tmp1)
  frame_nums = []
  for k in tmp2:
    frame_nums.append(len(k))
  return frame_nums

base_dir_all = '/content/'
#**********$%$%$%$%$%%$
# these indicate areas where user may want to customize like make a test set for example
#**********$%$%$%$%$%%$

tmp_h5s = base_dir_all+'tmp_h5s/'
for h5_subset_file in tqdm(subset_h5s):
  a = h5_subset_file.split('/')[-1].split('RETRAIN_')[-1].split('.h5')[0]
  base_h5 =  base_dir_all + 'DATA/data_'+a
  single_frame_h5s = base_h5+'/single_frame/'

  Path(tmp_h5s).mkdir(parents = True, exist_ok = True)
  Path(single_frame_h5s).mkdir(parents = True, exist_ok = True)
  #vvvvv add the frame nums to the H5 file
  frame_nums = foo_get_frame_nums_from_all_inds(h5_subset_file)
  utils.add_to_h5(h5_subset_file, 'frame_nums', frame_nums, overwrite_if_exists = True)
  #^^^^^ add the frame nums to the H5 file
  #vvvvvv single frame and then split
  utils.reduce_to_single_frame_from_color(h5_subset_file, tmp_h5s+'temp.h5')

  split_h5s = image_tools.split_h5_loop_segments([tmp_h5s+'temp.h5'],
                                     split_percentages=[.8, .2], #**********$%$%$%$%$%%$
                                     temp_base_name= [single_frame_h5s+'train',single_frame_h5s+'val'], #**********$%$%$%$%$%%$
                                     chunk_size=10000,
                                     add_numbers_to_name=False,
                                     disable_TQDM=True,
                                     set_seed=0,
                                     color_channel=False)
  #^^^^^^ single frame and then split
  #vvvvvvvv convert to all types
  utils.make_all_H5_types(single_frame_h5s)  # auto generate all the h5 types using a single set of flat (no color) image H5
  utils.make_alt_labels_h5s(single_frame_h5s)  # auto generate the different types of labels
  #^^^^^^^^ convert to all types
  #vvvvv AUGMENT #**********$%$%$%$%$%%$
  datagen = ImageDataGenerator(rotation_range=360,  #
                                width_shift_range=.1,  #
                                height_shift_range=.1,  #
                                shear_range=.00,  #
                                zoom_range=.25,
                                brightness_range=[0.2, 1.2])  #
  gaussian_noise = iaa.AdditiveGaussianNoise(loc = 0, scale=3)
  num_aug = 1
  xxx = utils.get_h5s(base_h5)
  xxx = utils.lister_it(xxx, remove_string=['ALT_LABELS', 'single_frame'])
  for each_h5 in xxx:
    if 'images' in utils.print_h5_keys(each_h5, True, False):
      combine_list = [each_h5]

      shutil.rmtree(tmp_h5s)
      Path(tmp_h5s).mkdir(parents = True, exist_ok = True)
      for ii in range(2):
        new_H5_file = each_h5.split('.')[0] + '_AUG_' + str(ii) + '.h5' # create new file name based on teh original H5 name
        new_H5_file = tmp_h5s + os.path.basename(new_H5_file)
        combine_list.append(new_H5_file)
        h5creator = image_tools.h5_iterative_creator(new_H5_file, overwrite_if_file_exists=True,
                                                      close_and_open_on_each_iteration=True, color_channel=True)

        with h5py.File(each_h5, 'r') as hf:
          for image, label in zip(hf['images'][:], hf['labels'][:]):
            aug_img_stack, labels_stack = image_tools.augment_helper(datagen, num_aug, 0, image, label)
            aug_img_stack = gaussian_noise.augment_images(aug_img_stack) # optional
            h5creator.add_to_h5(aug_img_stack[:, :, :, :], labels_stack)
        utils.copy_h5_key_to_another_h5(each_h5, new_H5_file, 'frame_nums', 'frame_nums') # copy the frame nums to the sug files
      # combine all the
      image_tools.split_h5_loop_segments(combine_list,
                                         [1],
                                         each_h5.split('.h5')[0]+'_AUG.h5',
                                         add_numbers_to_name = False,
                                         set_seed=0,
                                         color_channel=True)


"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$ copy over the paths of the new set into one folder, the new version of above code should make this outdated $$"""

from distutils.dir_util import copy_tree
import glob

a = utils.lister_it(glob("/content/*"), keep_strings='/data_')
x = "xxxxxxxxxxxxx/content/gdrive/MyDrive/Colab data/curation_for_auto_curator/ALL_RETRAIN_H5_data/"
shutil.rmtree(x)
Path(x).mkdir(parents = True, exist_ok = True)
for k in tqdm(a):
  Path(x).mkdir(parents = True, exist_ok = True)
  tmp1 = x+os.path.basename(k)
  Path(tmp1).mkdir(parents = True, exist_ok = True)
  copy_tree(str(k), tmp1)

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$print only the directories in a folder $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# can use this to copy the folder structure if we want to
src = "/content/gdrive/MyDrive/Colab data/curation_for_auto_curator/ALL_RETRAIN_H5_data"
for rootdir, dirs, files in os.walk(src):
    for subdir in dirs:
        print(os.path.join(src, subdir))






"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$ Fully automate the generation of all types data folder using a single H5 file (normal with "color" channels) $$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
from whacc import utils
from whacc import image_tools
from keras.preprocessing.image import ImageDataGenerator
import h5py
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa  # optional program to further augment data
#**********$%$%$%$%$%%$
# v^v^v^v^v^ these indacate areas where user may want to customize like make a test set for example
#**********$%$%$%$%$%%$
base_dir_all = '/content/'


full_session_h5s = "/content/gdrive/My Drive/Colab data/curation_for_auto_curator/H5_data/"
full_session_h5s = utils.get_h5s(full_session_h5s, 0)
b = utils.print_h5_keys(full_session_h5s[0], 1, 0)
c = utils.lister_it(b, remove_string='MODEL')
utils.print_list_with_inds(c)

for h5_subset_file in tqdm(full_session_h5s[:4]):
  a = h5_subset_file.split('/')[-1].split('RETRAIN_')[-1].split('.h5')[0]
  base_h5 =  base_dir_all + 'DATA_FULL/data_'+a
  single_frame_h5s = base_h5+'/single_frame/'

  # Path(tmp_h5s).mkdir(parents = True, exist_ok = True)
  Path(single_frame_h5s).mkdir(parents = True, exist_ok = True)

  #vvvvvv single frame
  single_name = single_frame_h5s+a+'.h5'
  utils.reduce_to_single_frame_from_color(h5_subset_file, single_name)
  #^^^^^^ single frame
  #vvvvvvvv convert to all types
  utils.make_all_H5_types(single_frame_h5s)  # auto generate all the h5 types using a single set of flat (no color) image H5
  utils.make_alt_labels_h5s(single_frame_h5s)  # auto generate the different types of labels
  #^^^^^^^^ convert to all types


"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$ transfer alt labels to the training and val set  $$"""


x = "/content/gdrive/MyDrive/Colab data/curation_for_auto_curator/ALL_RETRAIN_H5_data/"
for k in next(os.walk(x))[1]:
  a = utils.get_h5s(x+os.sep+k)
  a = utils.lister_it(a, keep_strings='train')
  src_h5 = utils.lister_it(a, '/ALT_LABELS/')[0]
  dest_h5s = utils.lister_it(a, remove_string='/ALT_LABELS/')
  for src_label_key in utils.print_h5_keys(src_h5, 1):
    for dest_H5 in dest_h5s:
      # print('from '+ src_h5 + ' to '+ dest_H5 + ' key--> '+ src_label_key)
      utils.copy_h5_key_to_another_h5(src_h5, dest_H5, src_label_key, src_label_key)

for k in next(os.walk(x))[1]:
  a = utils.get_h5s(x+os.sep+k)
  a = utils.lister_it(a, keep_strings='val')
  src_h5 = utils.lister_it(a, '/ALT_LABELS/')[0]
  dest_h5s = utils.lister_it(a, remove_string='/ALT_LABELS/')
  for src_label_key in utils.print_h5_keys(src_h5, 1):
    for dest_H5 in dest_h5s:
      # print('from '+ src_h5 + ' to '+ dest_H5 + ' key--> '+ src_label_key)
      utils.copy_h5_key_to_another_h5(src_h5, dest_H5, src_label_key, src_label_key)
# #testing it
# x = '/content/ALL_RETRAIN_H5_data/data_AH0698_170601_PM0121_AAAA/single_frame/train.h5'
# L1 = get_h5_key_and_concatenate(x, 'labels')
# L2 = get_h5_key_and_concatenate(x, '[0, 1]- (no touch, touch)')
# all(L1==L2)


"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$ transfer alt labels to the 'test' set  $$"""


x = "/content/gdrive/MyDrive/Colab data/curation_for_auto_curator/DATA_FULL/"

for k in next(os.walk(x))[1]:
  a = utils.get_h5s(x+os.sep+k)
  # a = utils.lister_it(a, keep_strings='train')
  src_h5 = utils.lister_it(a, '/ALT_LABELS/')[0]
  dest_h5s = utils.lister_it(a, remove_string='/ALT_LABELS/')
  for src_label_key in utils.print_h5_keys(src_h5, 1):
    for dest_H5 in dest_h5s:
      # print('from \n'+ src_h5 + ' to \n'+ dest_H5 + ' \nkey--> '+ src_label_key)
      utils.copy_h5_key_to_another_h5(src_h5, dest_H5, src_label_key, src_label_key)


"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$ copy over all of one h5 keys to another based on a matching string $$"""

model_3_h5s = "/content/gdrive/MyDrive/Colab data/curation_for_auto_curator/h5_data_withMODEL3/"
model_3_h5s = utils.get_h5s(model_3_h5s)
print('\n')
to_pred_h5s = '/content/gdrive/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED'
to_pred_h5s = utils.get_h5s(to_pred_h5s)

for src, dest in tzip(model_3_h5s, to_pred_h5s):
  assert os.path.basename(src)[:15] == os.path.basename(dest)[:15]
  with h5py.File(src, 'r') as h5src:
    with h5py.File(dest, 'r+') as h5dest:
      for key in tqdm(h5src.keys()):
        if 'MODEL_3_' in key:
          try:
            h5dest.create_dataset(key, shape=np.shape(h5src[key][:]),data=h5src[key][:])
          except:
            del h5dest[key]
            time.sleep(4)
            h5dest.create_dataset(key, shape=np.shape(h5src[key][:]),data=h5src[key][:])
