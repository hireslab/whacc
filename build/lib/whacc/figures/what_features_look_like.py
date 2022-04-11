import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from whacc import utils
import numpy as np
from whacc import image_tools
from natsort import os_sorted
import pickle
import pandas as pd
import os
import copy
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
import h5py
from imgaug import augmenters as iaa  # optional program to further augment data

import lightgbm as lgb

import shap


def load_data(h5_in, feature_list, split_int_n_chunks=1, split_chunk_ind=0, label_key='labels'):
    if isinstance(h5_in, list):
        all_x = [];
        all_y = []
        for k in h5_in:
            tmp_x, tmp_y = load_data(k, feature_list, split_int_n_chunks=split_int_n_chunks,
                                     split_chunk_ind=split_chunk_ind, label_key=label_key)
            all_x.append(tmp_x)
            all_y.append(tmp_y)
            del tmp_x, tmp_y

        all_x = np.vstack(all_x)
        all_y = np.hstack(all_y)
        return all_x, all_y
    all_x = None
    for k in feature_list:
        if all_x is None:
            all_x = image_tools.get_h5_key_and_concatenate(h5_in, k)[split_chunk_ind::split_int_n_chunks].astype(
                'float32')
        else:
            x = image_tools.get_h5_key_and_concatenate(h5_in, k)[split_chunk_ind::split_int_n_chunks].astype('float32')
            if len(x.shape) > 1:
                all_x = np.hstack((all_x, x))
            else:
                all_x = np.hstack((all_x, x[:, None]))

    all_y = image_tools.get_h5_key_and_concatenate(h5_in, label_key)[split_chunk_ind::split_int_n_chunks].astype(
        'float32')
    return all_x, all_y


def rm_nan_rows(a, a2=None):
    indx = ~np.isnan(a).any(axis=1)
    if a2 is None:
        return a[indx, :]
    else:
        return a[indx, :], a2[indx]


def get_feature_data_names(feature_list, n=2048):
    featuredata_names = [k.replace('FD__', '').replace('____', '') for k in feature_list]
    final_feature_names = []
    names = []
    nums = []
    for i, (k1, k2) in enumerate(zip(featuredata_names, feature_list)):
        if '_TOTAL_' in k2:
            names.append(k1)
            nums.append(np.arange(1))
            final_feature_names.append(k1)
        else:

            names.append(np.repeat(k1, n))
            nums.append(np.arange(n))
            final_feature_names.append([i1 + '_' + str(i2) for i1, i2 in zip(names[-1], nums[-1])])
    out = [np.hstack(k) for k in [final_feature_names, names, nums, featuredata_names]]
    out[0] = list(out[0])
    out[-1] = list(out[-1])
    return out


h5_in = '/Users/phil/Desktop/LIGHT_GBM/reg_80_border_and_andrew_both_samson_both__test_num_0_percent_15.h5'
mod_fn = '/Users/phil/Desktop/LIGHT_GBM/model_saves/model_num_0_of_10.pkl'

with open(mod_fn, 'rb') as f:
    model = pickle.load(f)

explainer = shap.TreeExplainer(model)

feature_list = os_sorted(utils.lister_it(utils.print_h5_keys(h5_in, 1, 0), keep_strings='FD__'))
final_feature_names, feature_names, feature_nums, feature_list_short = get_feature_data_names(feature_list)
x_train, y_train = load_data(h5_in, feature_list, split_int_n_chunks=1, split_chunk_ind=0)


cut_index = 1030 + np.asarray([-50, 50])

plt.plot(y_train[cut_index[0]:cut_index[1]])

starti = 88
sns.heatmap(x_train[:-41][cut_index[0]:cut_index[1], starti::2048//4].T)


'''
before these are all randomized i need to reference the final version that is very big 
i want fast touches with unique sidedness on each side 
access to images 
images that are good quality like jons or mine

'''



from whacc import analysis
file_name = 'ANM234232_140120_AH1030_AAAA_a_3lag.h5'
file_name = 'AH0407_160613_JC1003_AAAC_3lag.h5'
file_name = utils.lister_it(utils.get_h5s('/Volumes/GoogleDrive-114825029448473821206/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/'), file_name)


a = analysis.pole_plot(file_name[0])
ind = 14467 # 14467 to 14567
a.current_frame = ind
a.plot_it()

cut_index = [ind-150, ind+150]
plt.plot(np.arange(*cut_index), a.true_val[cut_index[0]:cut_index[1]])


cut_index = [ind+70, ind+140]  # [14537, 14607]
print(np.diff(cut_index))
plt.plot(np.arange(*cut_index), a.true_val[cut_index[0]:cut_index[1]])







def load_data(h5_in, feature_list, start_ind, end_ind, label_key='labels'):
    if isinstance(h5_in, list):
        all_x = [];
        all_y = []
        for k in h5_in:
            tmp_x, tmp_y = load_data(k, feature_list, split_int_n_chunks=start_ind,
                                     split_chunk_ind=end_ind, label_key=label_key)
            all_x.append(tmp_x)
            all_y.append(tmp_y)
            del tmp_x, tmp_y

        all_x = np.vstack(all_x)
        all_y = np.hstack(all_y)
        return all_x, all_y
    all_x = None
    for k in feature_list:
        if all_x is None:
            all_x = image_tools.get_h5_key_and_concatenate(h5_in, k)[start_ind:end_ind].astype('float32')
        else:
            x = image_tools.get_h5_key_and_concatenate(h5_in, k)[start_ind:end_ind].astype('float32')
            if len(x.shape) > 1:
                all_x = np.hstack((all_x, x))
            else:
                all_x = np.hstack((all_x, x[:, None]))

    all_y = image_tools.get_h5_key_and_concatenate(h5_in, label_key)[start_ind:end_ind].astype('float32')
    return all_x, all_y



## a) use this range [14537, 14607]
base_name = 'AH0407_160613_JC1003_AAAC_3lag.h5'
# /Users/phil/Dropbox/HIRES_LAB/curation_for_auto_curator/Data/Jon/AH0407/160613/AH0407x160609-144.mp4
img_h5 = utils.lister_it(utils.get_h5s('/Volumes/GoogleDrive-114825029448473821206/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/'), base_name)
feature_h5 = utils.lister_it(utils.get_h5s('/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/FEATURE_DATA/colab data/curation_for_auto_curator/DATA_FULL_in_range_only/data_AH0407_160613_JC1003_AAAC/3lag/'), base_name)
inds = [14419, 16417]
with h5py.File(img_h5[0], 'r') as h:
    images = h['images'][inds[0]:inds[1]]
    labels = h['labels'][inds[0]:inds[1]]
    frame_nums = np.asarray([len(labels)])
with h5py.File(feature_h5[0], 'r') as h:
    FD__original = h['FD__original'][inds[0]:inds[1]]

with h5py.File('/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/example_feature.h5', 'w') as h:
    h['images'] = images
    h['labels'] = labels
    h['frame_nums'] = frame_nums
    h['FD__original'] = FD__original
    h['base_name'] = base_name.encode("ascii", "ignore")
    h['img_h5'] = img_h5[0].encode("ascii", "ignore")
    h['feature_h5'] = feature_h5[0].encode("ascii", "ignore")


#
# for i, (i1, i2) in enumerate(utils.loop_segments(frame_nums)):
#     if i1<=14587 and i2>14587:
#         print(i)
#         print(i1, i2)
#         break
#
# frame_nums[12]
#
#
#
# feature_list = os_sorted(utils.lister_it(utils.print_h5_keys(h5_in, 1, 0), keep_strings='FD__'))
# final_feature_names, feature_names, feature_nums, feature_list_short = get_feature_data_names(feature_list)
# all_x, all_y = load_data(feature_h5[0], feature_list, inds[0], inds[1], label_key='labels')
#
# all_y[inds[0]:inds[1]]
# utils.print_h5_keys(feature_h5[0])
## b) get full 2048 features from resnet heatmap it

## c) take one example neuron and plot all normal feature transformations, group the transforms base on type (shift,
# SD, diff etc.)

## d) use b) to show how I do the TOTAL transforms
