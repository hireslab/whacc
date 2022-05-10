# from whacc.image_tools import *
# import copy
# import pdb
#
#
# class ImageBatchGenerator_feature_array(keras.utils.Sequence):
#
#     def __init__(self, time_length, batch_size, h5_file_list, label_key='labels', feature_len=None,
#                  label_index_to_lstm_len=None, edge_value=-1, remove_any_time_points_with_edges = True):
#         """
#
#         Parameters
#         ----------
#         time_length : total time points
#         batch_size : batch output for generator
#         h5_file_list : list of h5 strings or single h5 string
#         label_key : where y output comes from
#         feature_len : length of the features per time point
#         label_index_to_lstm_len : determines look back and look forward index refers to where the 'current' time point is
#         within the range of look_back_len; e.g. look_back_len = 7 label_index_to_lstm_len = 3 (middle index of 7) then
#         time point 0 will be at 3 and index 0, 1, 2 will be the past values and index 4, 5, 6 will be the future values.
#         look_back_len = 7 label_index_to_lstm_len = 0 (first index) then current time point will be at index 0 and all
#         other time point (1, 2, 3, 4, 5, 6) will be future values. Default is middle time point
#         edge_value : what to replace the edge values with, when time shifting you will have edges with no value, this
#         will replace those values with this number.
#         remove_any_time_points_with_edges : if true then batch size will not be the actual batch size it will be batch
#         size - the number of time points with edges in them, x and y will still match and this method is preferred for
#         training due to it not including unknown values.
#         """
#         assert time_length % 2 == 1, "number of images must be odd"
#         if label_index_to_lstm_len is None:
#             label_index_to_lstm_len = time_length // 2  # in the middle
#         h5_file_list = utils.make_list(h5_file_list, suppress_warning=True)
#         num_frames_in_all_H5_files = get_total_frame_count(h5_file_list)
#         file_inds_for_H5_extraction = batch_size_file_ind_selector(
#             num_frames_in_all_H5_files, batch_size)
#         subtract_for_index = reset_to_first_frame_for_each_file_ind(
#             file_inds_for_H5_extraction)
#         self.remove_any_time_points_with_edges = remove_any_time_points_with_edges
#         self.label_key = label_key
#         self.batch_size = batch_size
#         self.H5_file_list = h5_file_list
#         self.num_frames_in_all_H5_files = num_frames_in_all_H5_files
#         self.file_inds_for_H5_extraction = file_inds_for_H5_extraction
#         self.subtract_for_index = subtract_for_index
#         self.label_index_to_lstm_len = label_index_to_lstm_len
#         self.lstm_len = time_length
#         self.feature_len = feature_len
#         self.edge_value = edge_value
#         if remove_any_time_points_with_edges:
#             self.edge_value = np.nan
#             print('remove_any_time_points_with_edges == True : forcing edge_value to np.nan to aid in removing these time points')
#
#
#         self.get_frame_edges()
#         # self.full_edges_mask = self.full_edges_mask - (self.lstm_len // 2 - self.label_index_to_lstm_len)
#
#     def __getitem__(self, num_2_extract):
#         h = self.H5_file_list
#         i = self.file_inds_for_H5_extraction
#         all_edges = self.all_edges_list[np.int(i[num_2_extract])]
#         H5_file = h[np.int(i[num_2_extract])]
#         num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
#
#         with h5py.File(H5_file, 'r') as h:
#             b = self.lstm_len // 2
#             tot_len = h['images'].shape[0]
#
#             # assert tot_len - b > self.batch_size, "reduce batch size to be less than total length of images minus floor(lstm_len) - 1, MAX->" + str(
#             #     tot_len - b - 1)
#
#             i1 = num_2_extract_mod * self.batch_size - b
#             i2 = num_2_extract_mod * self.batch_size + self.batch_size + b
#             edge_left_trigger = abs(min(i1, 0))
#             edge_right_trigger = min(abs(min(tot_len - i2, 0)), b)
#             x = h['images'][max(i1, 0):min(i2, tot_len)]
#             if edge_left_trigger + edge_right_trigger > 0:  # in case of edge cases
#                 pad_shape = list(x.shape)
#                 pad_shape[0] = edge_left_trigger + edge_right_trigger
#                 pad = np.zeros(pad_shape).astype('float32')
#                 if edge_left_trigger > edge_right_trigger:
#                     x = np.concatenate((pad, x), axis=0)
#                 else:
#                     x = np.concatenate((x, pad), axis=0)
#
#             s = list(x.shape)
#             s.insert(1, self.lstm_len)
#             out = np.zeros(s).astype('float32')  # before was uint8
#             Z = self.label_index_to_lstm_len - self.lstm_len // 2
#             for i in range(self.lstm_len):
#                 i_temp = i
#                 i = i - Z
#                 i1 = max(0, b - i)
#                 i2 = min(s[0], s[0] + b - i)
#                 i3 = max(0, i - b)
#                 i4 = min(s[0], s[0] + i - b)
#                 # print('take ', i3, ' to ', i4, ' and place in ', i1, ' to ', i2)
#                 out[i1:i2, i_temp, ...] = x[i3:i4, ...]
#
#             out = out[b:s[0] - b, ...]
#             i1 = num_2_extract_mod * self.batch_size
#             i2 = num_2_extract_mod * self.batch_size + self.batch_size
#             raw_Y = h[self.label_key][i1:i2]
#
#             adjust_these_edge_frames = np.intersect1d(all_edges.flatten(), np.arange(i1, i2))
#             b2 = b - self.label_index_to_lstm_len  # used to adjust mask postion based on where the center value is
#             for atef in adjust_these_edge_frames:
#                 mask_ind = np.where(atef == all_edges)[1][0]
#                 mask_ind = mask_ind - b2
#                 mask_ind = mask_ind % self.full_edges_mask.shape[0]  # wrap around index
#
#                 mask_ = self.full_edges_mask[mask_ind]
#                 mask_ = mask_ == 1
#                 out_ind = atef + i1 - b2
#                 out_ind = out_ind % out.shape[0]  # wrap around index
#                 out[out_ind][mask_] = self.edge_value
#
#
#             s = out.shape
#             out = np.reshape(out, (s[0], s[1] * s[2]))
#             if self.remove_any_time_points_with_edges:
#                 keep_inds = ~np.isnan(np.mean(out, axis = 1))
#                 out = out[keep_inds]
#                 raw_Y = raw_Y[keep_inds]
#
#             return out, raw_Y
#
#     def __len__(self):
#         return len(self.file_inds_for_H5_extraction)
#
#     def getXandY(self, num_2_extract):
#         """
#
#         Parameters
#         ----------
#         num_2_extract :
#
#
#         Returns
#         -------
#
#         """
#         rgb_tensor, raw_Y = self.__getitem__(num_2_extract)
#         return rgb_tensor, raw_Y
#
#     def image_transform(self, raw_X):
#         """input num_of_images x H x W, image input must be grayscale
#         MobileNetV2 requires certain image dimensions
#         We use N x 61 x 61 formated images
#         self.IMG_SIZE is a single number to change the images into, images must be square
#
#         Parameters
#         ----------
#         raw_X :
#
#
#         Returns
#         -------
#
#
#         """
#         # kept this cause this is the format of the image generators I know this is redundant
#         rgb_batch = copy.deepcopy(raw_X)
#         rgb_tensor = rgb_batch
#         self.IMG_SHAPE = (self.feature_len)
#         return rgb_tensor
#
#     def get_frame_edges(self):
#         self.all_edges_list = []
#         b = self.lstm_len // 2
#
#         s = [b * 2, self.lstm_len, self.feature_len]
#         for H5_file in self.H5_file_list:
#             with h5py.File(H5_file, 'r') as h:
#                 full_edges_mask = np.ones(s)
#                 tmp1 = np.arange(1, self.lstm_len)
#                 front_edge = tmp1[:self.label_index_to_lstm_len]
#                 back_edge = tmp1[:self.lstm_len - self.label_index_to_lstm_len - 1]
#
#                 edge_ind = np.flip(front_edge)
#                 for i in front_edge:
#                     # print(i - 1, ':', edge_ind[i - 1])
#                     # print(full_edges_mask[i - 1, :edge_ind[i - 1], ...].shape)
#                     # print('\n')
#                     full_edges_mask[i - 1, :edge_ind[i - 1], ...] = np.zeros_like(
#                         full_edges_mask[i - 1, :edge_ind[i - 1], ...])
#
#                 edge_ind = np.flip(back_edge)
#                 for i in back_edge:
#                     # print(-i, -edge_ind[i - 1], ':')
#                     # print(full_edges_mask[-i, -edge_ind[i - 1]:, ...].shape)
#                     # print('\n')
#                     full_edges_mask[-i, -edge_ind[i - 1]:, ...] = np.zeros_like(
#                         full_edges_mask[-i, -edge_ind[i - 1]:, ...])
#
#                 all_edges = []
#                 for i1, i2 in utils.loop_segments(h['frame_nums']):  # 0, 1, 3998, 3999 ; 4000, 4001, 7998, 7999; ...
#                     edges = (np.asarray([[i1], [i2 - b]]) + np.arange(0, b).T).flatten()
#                     all_edges.append(edges)
#
#                 all_edges = np.asarray(all_edges)
#             self.all_edges_list.append(all_edges)
#             # pdb.set_trace()
#             full_edges_mask = full_edges_mask.astype(int)
#             self.full_edges_mask = full_edges_mask == 0
#
#
# h5 = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing_features_data/feature_data/regular_80_border/DATA_FULL/3lag/holy_test_set_10_percent_3lag.h5'
# # h5 = '/Users/phil/Desktop/temp.h5'
# a = ImageBatchGenerator_feature_array(7, 4000, h5, label_key='labels', feature_len=2048,
#                                       label_index_to_lstm_len=3, edge_value=-1, remove_any_time_points_with_edges = False)
# x, y = a.__getitem__(0)
# # # utils.np_stats(x)
# # # s = x.shape;
# # # x2 = np.reshape(x, (s[0], s[1] * s[2]))
#
#
# plt.figure()
# plt.plot(y[2200:2300])
#
# plt.figure()
# plt.imshow(x[2200:2300, ::512].T)
#
#





import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from whacc import utils
import numpy as np
from whacc import image_tools
from natsort import natsorted
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
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$ SET MATPLOTLIB DEFAULTS $$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

import seaborn as sns

sns.set(rc={
 'axes.axisbelow': False,
 'axes.edgecolor': 'lightgrey',
 'axes.facecolor': 'None',
 'axes.grid': False,
 'axes.labelcolor': 'dimgrey',
 'axes.spines.right': False,
 'axes.spines.top': False,
 'figure.facecolor': 'white',
 'lines.solid_capstyle': 'round',
 'patch.edgecolor': 'w',
 'patch.force_edgecolor': True,
 'text.color': 'dimgrey',
 'xtick.bottom': False,
 'xtick.color': 'dimgrey',
 'xtick.direction': 'out',
 'xtick.top': False,
 'ytick.color': 'dimgrey',
 'ytick.direction': 'out',
 'ytick.left': False,
 'ytick.right': False})
sns.set_context("notebook", rc={"font.size":16,
                                "axes.titlesize":20,
                                "axes.labelsize":18})

## font

import matplotlib
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.sans-serif'] = "Arial"

plt.rcParams.update({'font.family':'Arial'})

# plt.rcParams.update({'font.family':'sans-serif', 'fontname':'Arial'})
from matplotlib.font_manager import findfont, FontProperties
font = findfont(FontProperties(family=['sans-serif']))
font
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$ SET MATPLOTLIB DEFAULTS $$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""


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

#
# h5_in = '/Users/phil/Desktop/LIGHT_GBM/reg_80_border_and_andrew_both_samson_both__test_num_0_percent_15.h5'
# mod_fn = '/Users/phil/Desktop/LIGHT_GBM/model_saves/model_num_0_of_10.pkl'
#
# with open(mod_fn, 'rb') as f:
#     model = pickle.load(f)
#
# explainer = shap.TreeExplainer(model)
#
# feature_list = natsorted(utils.lister_it(utils.print_h5_keys(h5_in, 1, 0), keep_strings='FD__'))
# final_feature_names, feature_names, feature_nums, feature_list_short = get_feature_data_names(feature_list)
# x_train, y_train = load_data(h5_in, feature_list, split_int_n_chunks=1, split_chunk_ind=0)
#
#
# cut_index = 1030 + np.asarray([-50, 50])
#
# plt.plot(y_train[cut_index[0]:cut_index[1]])
#
# starti = 88
# sns.heatmap(x_train[:-41][cut_index[0]:cut_index[1], starti::2048//4].T)


'''
before these are all randomized i need to reference the final version that is very big 
i want fast touches with unique sidedness on each side 
access to images 
images that are good quality like jons or mine

'''
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$ SELECTING AN INFORMATIVE TOUCH TRACE $$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

# h5_in = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/3lag/train_3lag.h5'
# # here I am look for a good slice of touches in the orignal H5 file
# # training
# 383810 # aug - 3 border X 10
# 292217 # reg - 80 border (676027 - 383810)
# y = image_tools.get_h5_key_and_concatenate(h5_in)[383810:]
# smooth_by = 151
# ind = 0
#
# y2 = np.convolve(np.diff(y), np.ones(smooth_by)/smooth_by)
# bst_inds = np.flip(np.argsort(y2))
# add_to = smooth_by//2
# fig, ax = plt.subplots(5, 4)
# for ind, ax2 in enumerate(ax.flatten()):
#     ind+=100
#     ax2.plot(y[bst_inds[ind]-add_to:bst_inds[ind]+add_to])
#
# ind = 115
# add_to = 200
# plt.plot(y[bst_inds[ind]-add_to:bst_inds[ind]+add_to])
#
# plt.plot(y[271792:271792+200])
#
# plt.plot(y[271792:271792+200-50])
# 271792+383810
#
# with h5py.File(h5_in, 'r') as h:
#     test_image = h['images'][bst_inds[ind]+383810]
# plt.imshow(test_image)
# # CHECKING THE MAKE SURE THE IMAGES LOOK GOOD
# from whacc import analysis
# pp = analysis.pole_plot(h5_in)
# pp.current_frame = bst_inds[ind]+383810-20
# pp.plot_it()
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$ MAKE AN EXAMPLE H5 FOR SHOWING FEATURE ENGINEERING $$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
h5_in = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/3lag/train_3lag.h5'

h5_new = '/Users/phil/Desktop/feature_example2.h5'
utils.print_h5_keys(h5_in)
with h5py.File(h5_in, 'r') as h:
    with h5py.File(h5_new, 'w') as h2:
        h2['images'] = h['images'][655102:656252]
        h2['labels'] = h['labels'][655102:656252]
        h2['frame_nums'] = [len(h2['labels'])]

with h5py.File(h5_new, 'r') as h:
    y3 = h['labels'][:]
    print(h['frame_nums'][:])


from whacc import model_maker
# convert to feature data
# h5_3lag = '/Users/phil/Desktop/temp_dir/AH0407x160609_3lag.h5'
h5_feature_data = h5_new.replace('.h5', '_feature_data.h5')
in_gen = image_tools.ImageBatchGenerator(500, h5_new, label_key='labels')
RESNET_MODEL = model_maker.load_final_model()
utils.convert_h5_to_feature_h5(RESNET_MODEL, in_gen, h5_feature_data)


# h5_feature_data = '/Users/phil/Desktop/temp_dir/AH0407x160609_3lag_feature_data.h5'
# generate all the modified features (41*2048)+41 = 84,009
utils.standard_feature_generation(h5_feature_data)

#
#
# x = utils.get_whacc_path() + "/whacc_data/feature_data/"
# d = utils.load_obj(x + 'feature_data_dict.pkl')
# # x = d['features_used_of_10'][:]#>=4
# d['final_selected_features'] = utils.get_selected_features(greater_than_or_equal_to=4)
# x = d['final_selected_features'][:]
# d['final_selected_features_bool'] = [True if k in x else False for k in range(2048 * 41 + 41)]
#
# d = load_feature_data()
# def load_selected_features(h5_in, feature_index=None):
#     """
#
#     Parameters
#     ----------
#     h5_in : full directory of the full feature h5 file 84,009
#     feature_index : bool array of len 84,009
#
#     Returns
#     -------
#
#     """
#     d = utils.load_feature_data()
#     feature_list = d['feature_list_unaltered']
#     if feature_index is None:
#         feature_index= d['final_selected_features_bool']
#         # feature_index = utils.get_selected_features(greater_than_or_equal_to=4)
#     feature_index = utils.make_list(feature_index)
#
#     if isinstance(h5_in, list):
#         all_x = []
#         all_y = []
#         for k in h5_in:
#             tmp_x, tmp_y = load_selected_features(k, feature_list)
#             all_x.append(tmp_x)
#             all_y.append(tmp_y)
#             del tmp_x, tmp_y
#
#         all_x = np.vstack(all_x)
#         all_y = np.hstack(all_y)
#         return all_x, all_y
#
#     all_x = None
#     for k in tqdm(feature_list):
#         if all_x is None:
#             all_x = image_tools.get_h5_key_and_concatenate(h5_in, k).astype('float32')
#             inds = feature_index[:all_x.shape[1]]
#             del feature_index[:all_x.shape[1]]
#             all_x = all_x[:, inds]
#         else:
#             x = image_tools.get_h5_key_and_concatenate(h5_in, k).astype('float32')
#             if len(x.shape) > 1:
#                 inds = feature_index[:x.shape[1]]
#                 del feature_index[:x.shape[1]]
#                 x = x[:, inds]
#                 all_x = np.hstack((all_x, x))
#             else:
#                 inds = feature_index.pop(0)  # single true or false to include the "TOTAL" variables
#                 if inds:
#                     all_x = np.hstack((all_x, x[:, None]))
#     return all_x


x = utils.get_whacc_path() + "/whacc_data/feature_data/"
d = utils.load_obj(x + 'feature_data_dict.pkl')
# x = d['features_used_of_10'][:]#>=4
d['final_selected_features'] = np.where(d['features_used_of_10'][:]>=4)[0]
x = d['final_selected_features'][:]
d['final_selected_features_bool'] = [True if k in x else False for k in range(2048 * 41 + 41)]

all_x = utils.load_selected_features(h5_feature_data)

with h5py.File(h5_feature_data, 'r+') as h:
    h['final_3095_features'] = all_x



"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$ NOW PLOT IT TO SHOW THE READER  what the features look like$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""


"""$$$$$$$$$$$$$$$$$$$$$$$$$ grab the correct index $$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
import scipy.cluster.hierarchy as sch
def cluster_corr(corr_array, max_div_by = 2):
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/max_div_by
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx], idx

h5_feature_data = '/Users/phil/Desktop/feature_example2_feature_data.h5'
utils.print_h5_keys(h5_feature_data)

with h5py.File(h5_feature_data, 'r') as h:
    d = dict()
    for k in h.keys():
        d[k] = copy.deepcopy(h[k][500:500+150])

# with h5py.File(h5_feature_data, 'r') as h:
#     d = dict()
#     for k in h.keys():
#         d[k] = copy.deepcopy(h[k][:])

def foo_neuron_heatmap(FD, inds, labels, only_return_data = False):
    # sns.heatmap(cc)
    data = FD[:, inds].T
    data = data-np.min(data)
    data = data/np.max(data)
    if only_return_data:
        return data
    fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [10, 1]})
    plt.sca(ax[0])
    ax_heat = sns.heatmap(data, cbar=False)
    x = np.linspace(0, 2047, 5).astype(int)
    ax_heat.set_yticks(x)
    ax_heat.set_yticklabels(x+1)
    plt.ylabel('2048 CNN output neurons')
    plt.sca(ax[1])
    plt.plot(labels, 'k-', label='touch trace')
    ax[1].set_yticklabels([])
    x = np.arange(0, 151, 50).astype(int)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(x)
    plt.xlabel('time (ms)')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(.8, 1.85), loc=2, borderaxespad=0, fontsize=10)
    plt.sca(ax[0])
    return data
"""$$$$$$$$$$$$$$$$$$$$$$$$$ to plot each $$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

labels = d['labels']
FD = d['FD__original']
FD[0, :] = FD[0, :]+0.00000001 # so that CC works, prevents values form one neuron all being the same
# 1 method use clustering
cc = np.corrcoef(FD, rowvar=False)
cc, inds = cluster_corr(cc, 2)


data = foo_neuron_heatmap(FD, inds, labels)

from matplotlib.patches import Rectangle
plt.gca().add_patch(Rectangle((40 - .5, 755 - .5), 30, 50, facecolor='none'))


ax.vlines(np.arange(0, x.shape[1], 30), *ax.get_xlim(), 'w')

plt.ylim([780+25, 780-25])
plt.xlim([40, 70])
plt.figure()

sns.heatmap(data[780-25:780+25, 40:70])

plt.show(block=False)
plt.close('all')

plt.legend(bbox_to_anchor=(.8, 1.85), loc=2, borderaxespad=0, fontsize=10)


fd = d['FD__FD__original_diff_periods_-50____']
foo_neuron_heatmap(fd, inds, labels)

fd = d['FD__FD__original_diff_periods_-2____']
foo_neuron_heatmap(fd, inds, labels)

from natsort import natsorted
keys = utils.print_h5_keys(h5_feature_data, 1, 0)
keys = natsorted(utils.lister_it(keys, 'FD__', 'TOTAL'))
cut_list = []
for k in keys:
    fd = d[k]
    data = foo_neuron_heatmap(fd, inds, labels, only_return_data = True)
    cut_list.append(data[780-25:780+25, 40:70])

x = np.hstack(cut_list[:16])
ax = sns.heatmap(x)
ax.vlines(np.arange(0, x.shape[1], 30), *ax.get_xlim(), 'w')


x = np.hstack(cut_list[16:23])
ax = sns.heatmap(x)
ax.vlines(np.arange(0, x.shape[1], 30), *ax.get_xlim(), 'w')

x = np.hstack(cut_list[23:30])
ax = sns.heatmap(x)
ax.vlines(np.arange(0, x.shape[1], 30), *ax.get_xlim(), 'w')

x = np.hstack(cut_list[30:40])
ax = sns.heatmap(x)
ax.vlines(np.arange(0, x.shape[1], 30), *ax.get_xlim(), 'w')

keys[:16] # diff
keys[16:23] # rolling mean
keys[23:30] # rolling std
keys[30:40] # shift

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$ PLOT TOTAL STD FEATURES   $$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
keys = utils.print_h5_keys(h5_feature_data, 1, 0)
keys = natsorted(utils.lister_it(keys, 'TOTAL'))
total_type_features = []
for k in keys:
    fd = d[k]
    total_type_features.append(fd)

total_type_features = np.vstack(total_type_features)

key_labels = []
for k in keys:
    k2 = ' '.join(k.split('FD__original_')[-1].split('_')[:4])
    key_labels.append(k2)
    print(k2)
key_labels[-1] = 'Original'

fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [10, 1]})
plt.sca(ax[0])

ax_heat = sns.heatmap(total_type_features, cbar=False, yticklabels=key_labels)
ax_heat.set_yticklabels(key_labels, fontsize= 7)

# x = np.linspace(0, 40, 5).astype(int)
# ax_heat.set_yticks(x)
# ax_heat.set_yticklabels(x)
plt.ylabel('TOTAL SD features (2048 reduced to 1)')
plt.sca(ax[1])
plt.plot(labels, 'k-', label='touch trace')
ax[1].set_yticklabels([])
x = np.arange(0, 151, 50).astype(int)
ax[1].set_xticks(x)
ax[1].set_xticklabels(x)
plt.xlabel('time (ms)')
plt.tight_layout()
plt.legend(bbox_to_anchor=(.8, 1.65), loc=2, borderaxespad=0, fontsize=10)
plt.sca(ax[0])

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$ PLOT 3095 features   $$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
def foo_norm(x):################################################
    x = x-np.min(x)
    x = x/np.max(x)
    return x
# def foo_norm(x): ################################################
#     return x
FD = d['final_3095_features']

FD2 = copy.deepcopy(FD)
for i, k in enumerate(FD.T):
    FD2[:, i] = foo_norm(k)
FD = FD2
FD[0, :] = FD[0, :]+0.00000001
cc = np.corrcoef(FD, rowvar=False)
cc, inds = cluster_corr(cc, 1.8)



# inds = np.arange(FD.shape[1]) ################################################



data = FD[:, inds].T
sns.heatmap(cc)

fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [10, 1]})
plt.sca(ax[0])

ax_heat = sns.heatmap(data, cbar=False)
# ax_heat.set_yticklabels(key_labels, fontsize= 7)

# x = np.linspace(0, 40, 5).astype(int)
# ax_heat.set_yticks(x)
# ax_heat.set_yticklabels(x)
plt.ylabel('______')
plt.sca(ax[1])
plt.plot(d['labels'], 'k-', label='touch trace')


ax[1].set_yticklabels([])
x = np.arange(0, 151, 50).astype(int)
ax[1].set_xticks(x)
ax[1].set_xticklabels(x)
plt.xlabel('time (ms)')
plt.tight_layout()
plt.legend(bbox_to_anchor=(.8, 1.65), loc=2, borderaxespad=0, fontsize=10)
plt.sca(ax[0])



"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$ test loaded model on small set   $$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""



model_save_test = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/models/top_features_greter_than_3_out_of_10_3095_features/model_num_0_of_10.pkl'
bd = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/models/top_features_greter_than_3_out_of_10_3095_features/'
for i, model_save_test in enumerate(utils.get_files(bd, '*of_10.pkl')):
    mod = utils.load_obj(model_save_test)
    # d['final_3095_features']
    pred_y = mod.predict(d['final_3095_features'])
    plt.plot(pred_y+i+1, '-', label='pred trace')



x = utils.get_whacc_path() + "/whacc_data/feature_data/"
d = utils.load_obj(x + 'feature_data_dict.pkl')

x = d['feature_list_unaltered']

from natsort import ns
from natsort import natsorted, humansorted

natsorted(x, alg=ns.INT | ns.LOCALE | ns.PATH)
natsorted(x)
humansorted(x)

natsorted(x, alg=ns.REAL)






def foo(x, remove_inds, axis = 0):
    x = np.swapaxes(x, 0, axis)
    save_x = []
    for k in remove_inds:
        save_x.append(copy.deepcopy(x[k]))
        x = np.delete(x, k)
    return x, save_x

"""
so below works but want to se if it works without giving memory errors 
"""
import os

# Getting all memory using os.popen()
import copy
import pandas as pd
import numpy as np
from psutil import Process, virtual_memory
def mem():
    print('RAM memory % used:', virtual_memory()[2])

del x
mem()
a, b = 6000, 800000
x = pd.DataFrame(np.arange(a*b).reshape(a, b))
mem()
tmp1 = pd.DataFrame([])
pop_inds = [1, 3]
for i, k in enumerate(pop_inds):
    tmp1.insert(i, i, x.pop(k))
mem()
print(tmp1)
print(x)
for i, k in enumerate(pop_inds):
    x.insert(k, k, tmp1.pop(i))
mem()
print(x)



x2, save_x = foo(x, [2, 3], axis = 0)
x, x2, save_x



def foo(x, axis = 0):
    assert axis is in [0, 1], 'I can only deal with 2 dim data'
    out = copy.deepcopy(x[])
    return x, out

all_mean_shap = []
x = np.arange(6*4).reshape(6, 4)
x
rows = [10, 20, 25]
col = [2, 4]
for i in range(5):
    for c in
    tmp_save_x = copy.deepcopy(all_val_sets[k1:k2, :])
    print(tmp_save_x.shape)
    all_val_sets = np.delete(all_val_sets, np.arange(k1, k2), axis = 0) # delete chunk
    print(all_val_sets.shape)
    # PREDICT
    shap_vals = lgbm.predict(all_val_sets, pred_contrib = True)


    np.abs(shap_vals, out = shap_vals) # this prevent mem error, shap_vals is modified here
    mean_shape = np.mean(shap_vals, axis = 0)
    all_mean_shap.append(mean_shape)

    del shap_vals
    gc.collect()

    all_val_sets = np.insert(all_val_sets, k1, tmp_save_x, axis = 0) # insert chunk back in


