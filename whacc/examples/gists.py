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
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$make a very small onset and offset extracot!!$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""
som enotes might have to do this for eeach weird extracted lag and lad diff...check this
"""

from whacc import utils, image_tools, analysis
import numpy as np
import h5py


all_h5s_imgs = ['/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_using_pole_tracker.h5']
h_cont = image_tools.get_h5_key_and_concatenate(all_h5s_imgs, 'labels')
h5c = image_tools.h5_iterative_creator('/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug.h5',
                                           overwrite_if_file_exists = True,
                                           color_channel = False)

h_cont_onset_offset = np.append(0, np.abs(np.diff(h_cont)))
utils.create_master_dataset(h5c, all_h5s_imgs, [h_cont_onset_offset], borders=3, max_pack_val=100)

utils.print_h5_keys(h5c.h5_full_file_name)
corrected_labels = []
with h5py.File(h5c.h5_full_file_name, 'r+') as h:
    for k in h['inds_extracted'][:]:
        corrected_labels.append(h_cont[k])
    h['labels'][:] = corrected_labels

tmp1 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug.h5'
a = analysis.pole_plot(tmp1, true_val=image_tools.get_h5_key_and_concatenate([tmp1]))
a.next()

"""
US THIS AFTER CREATING AUG IMAGES TO TRANSFER FRAME NUMS
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
                    '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_0.h5',]
image_tools.split_h5_loop_segments(h5_to_split_list, [7, 3], [bd+'train', bd+'val'], chunk_size=1000, add_numbers_to_name=False,
             disable_TQDM=False, set_seed = 0, color_channel=False)





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




