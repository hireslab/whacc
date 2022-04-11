from whacc import utils, image_tools, transfer_learning, analysis
from IPython.utils import io
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow import keras
from sklearn.utils import class_weight
import time
from pathlib import Path
import os
import copy
import numpy as np
from tensorflow.keras import applications
from pathlib import Path
import shutil
import zipfile
from datetime import datetime
import pytz
import json
from whacc import model_maker

from whacc.model_maker import *
import itertools

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import h5py
from whacc import image_tools
from whacc import utils
import copy
import time
import os
import pdb
import glob
from tqdm.contrib import tzip
import scipy.io as spio
import h5py
from tqdm.notebook import tqdm
from matplotlib import cm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.patches as mpatches
from scipy.signal import medfilt, medfilt2d
import cv2


def remap_array_to_color_channels(in_array, color_numers_to_match=None, color_list=[0, .5, .2, .3, .75, .85],
                                  cmap_col='inferno'):
    in_array = copy.deepcopy(in_array.astype(int))
    out_array = np.stack((in_array,) * 3, axis=-1)
    color_dict = dict()
    cmap = cm.get_cmap(cmap_col)
    if color_numers_to_match is None:
        color_numers_to_match = np.unique(in_array).astype(int)
        color_numers_to_match = color_numers_to_match.astype(int)

    for i, k1 in enumerate(color_list):
        color_dict[i] = np.asarray(cmap(k1)[:-1]) * 255

    if color_numers_to_match is None:
        for ii, kk in enumerate(np.unique(in_array)):
            out_array[(in_array == kk).astype(bool)] = color_dict[ii]
    else:
        for ii, kk in enumerate(color_numers_to_match):
            out_array[(in_array == kk).astype(bool)] = color_dict[ii]

    return out_array, color_dict


def remap_array_to_color_channels(in_array, color_numers_to_match=None, color_list=[0, .5, .2, .3, .75, .85],
                                  cmap_col='inferno'):
    in_array = copy.deepcopy(in_array).astype(int)
    out_array = np.stack((in_array,) * 3, axis=-1)

    color_dict = dict()
    cmap = cm.get_cmap(cmap_col)
    if color_numers_to_match is None:
        color_numers_to_match = np.unique(in_array).astype(int)
        print(color_numers_to_match)

    for key, k1 in zip(color_numers_to_match, color_list):
        color_dict[key] = (np.asarray(cmap(k1)[:-1]) * 255).astype(int)
    for ii, kk in enumerate(color_numers_to_match):
        out_array[(in_array == kk).astype(bool)] = color_dict[kk]

    return out_array, color_dict


def foo_heatmap(real_bool, pred_bool, in_range, frame_nums, lines_thick=20, title_str=''):
    acc_percentage = ((pred_bool == real_bool) * 1).astype(float)
    acc_percentage[np.invert(in_range.astype(bool))] = np.nan
    acc_percentage = np.nanmean(acc_percentage)
    acc_percentage = str(np.round(acc_percentage * 100, 2)) + '%  '
    title_str = acc_percentage + title_str

    c_list = []
    for n in [2, 3, 4, 5, 8]:
        c_list.append(.0833 / 2 + n * .0833)

    max_ = np.max(frame_nums)
    x = np.zeros([len(frame_nums), int(max_)]) - 2

    d = real_bool - pred_bool
    d = d + (real_bool + pred_bool == 2) * 2  # TP = 2, TN = 0, FP = -1, FN = 1 ...... -2 pole out of range

    for i, (k1, k2) in enumerate(utils.loop_segments(frame_nums)):
        L = frame_nums[i]
        tmp1 = d[k1:k2]

        tmp1[in_range[k1:k2] == 0] = -2
        # in_range[k1:k2]
        x[i, :L] = tmp1

    x2, color_dict = remap_array_to_color_channels(x, color_numers_to_match=[0, 2, 1, -1, -2], color_list=c_list,
                                                   cmap_col='Paired')
    x2 = np.repeat(x2, lines_thick, axis=0)

    plt.figure(figsize=[40, 40])
    _ = plt.imshow(x2)

    # LEGEND
    all_labels = ['TN', 'TP', 'FN', 'FP', 'pole down']
    patches = []
    for i, ii in zip(color_dict, all_labels):
        c = color_dict[i] / 255
        patches.append(mpatches.Patch(color=c, label=ii))
    plt.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0, fontsize=50)
    plt.title(title_str, fontsize=70)

    return x2, color_dict


def foo_heatmap_with_critical_errors(real_bool, pred_bool, in_range, frame_nums, lines_thick=20, title_str='',
                                     figsize=(10, 10)):
    acc_percentage = ((pred_bool == real_bool) * 1).astype(float)
    acc_percentage[np.invert(in_range.astype(bool))] = np.nan
    acc_percentage = np.nanmean(acc_percentage)
    acc_percentage = str(np.round(acc_percentage * 100, 2)) + '%  '
    title_str = acc_percentage + title_str

    c_list = []
    for n in [2, 3, 4, 5, 8]:
        c_list.append(.0833 / 2 + n * .0833)

    max_ = np.max(frame_nums)
    x = np.zeros([len(frame_nums), int(max_)]) - 2

    d = real_bool - pred_bool
    d = d + (real_bool + pred_bool == 2) * 2  # TP = 2, TN = 0, FP = -1, FN = 1 ...... -2 pole out of range

    for i, (k1, k2) in enumerate(utils.loop_segments(frame_nums)):
        L = frame_nums[i]
        tmp1 = d[k1:k2]

        tmp1[in_range[k1:k2] == 0] = -2
        # in_range[k1:k2]
        x[i, :L] = tmp1

    x2, color_dict = remap_array_to_color_channels(x, color_numers_to_match=[0, 2, 1, -1, -2], color_list=c_list,
                                                   cmap_col='Paired')
    x2 = np.repeat(x2, lines_thick, axis=0)

    # get the color coded error type matrix
    a = analysis.error_analysis(real_bool, pred_bool, frame_num_array=frame_nums)
    d = copy.deepcopy(a.coded_array)
    d[d < 0] = -2
    d[d >= 4] = -2

    max_ = np.max(frame_nums)
    x = np.zeros([len(frame_nums), int(max_)]) - 2

    for i, (k1, k2) in enumerate(utils.loop_segments(frame_nums)):
        L = frame_nums[i]
        tmp1 = d[k1:k2]

        tmp1[in_range[k1:k2] == 0] = -2
        x[i, :L] = tmp1
    c_list = []
    for n in [1, 3, 4, 5, 8]:  # ['ghost', 'miss', 'join', 'split', nothing
        c_list.append(.1111 / 2 + n * .1111)
    x2_error_type, color_dict_error_type = remap_array_to_color_channels(x, color_numers_to_match=[0, 1, 2, 3, -2],
                                                                         color_list=c_list, cmap_col='Set1')
    print(np.nanmin(x2_error_type))
    x2_error_type = np.repeat(x2_error_type, lines_thick, axis=0)

    for i, (k1, k2) in enumerate(utils.loop_segments([10, 10] * len(
            frame_nums))):  # nan out certain regions so that we can leave those to be filled in with actual heatmap
        if (i % 2) != 0:
            x2_error_type[k1:k2] = color_dict_error_type[-2]

    x3 = copy.deepcopy(x2).astype(int)
    inds = x2_error_type != color_dict_error_type[-2]
    x3[inds] = x2_error_type[inds]
    plt.figure(figsize=figsize)
    plt.imshow(x3)

    # LEGEND
    all_labels = ['TN', 'TP', 'FN', 'FP', 'pole down']
    patches = []
    for i, ii in zip(color_dict, all_labels):
        c = color_dict[i] / 255
        patches.append(mpatches.Patch(color=c, label=ii))
    all_labels = ['ghost', 'miss', 'join', 'split']
    for i, ii in zip(color_dict_error_type, all_labels):
        c = color_dict_error_type[i] / 255
        patches.append(mpatches.Patch(color=c, label=ii))
    plt.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0, fontsize=15, prop={'size': 6})
    plt.title(title_str, fontsize=20)

    return x3


def plot_segments_with_array_blocks(actual_h5_img_file, list_of_inds_to_plot, in_list_of_arrays=[], seg_num=0, border=4,
                                    height=20, img_width=61,
                                    color_numers_to_match=[0, 1, 2, 3, 4, 5], color_list=[0, .5, .2, .3, .75, .85],
                                    cmap_col='inferno', max_frames=40, min_frames=10):
    # apply max and min
    # in_list_of_arrays[0] needs to be the "true"" values
    if in_list_of_arrays == []:
        print('no input arrays, returning...')
        return
    color_dict = dict()
    cmap = cm.get_cmap(cmap_col)

    for i, k1 in enumerate(color_list):
        color_dict[i] = np.asarray(cmap(k1)[:-1]) * 255

    in_list_of_arrays = copy.deepcopy(in_list_of_arrays)

    # set/adjust size of the array
    inds = list(range(list_of_inds_to_plot[seg_num][0] - border, list_of_inds_to_plot[seg_num][-1] + 1 + border * 2))
    inds = inds[:max_frames]
    if len(inds) < min_frames:
        inds = np.arange(inds[0], inds[0] + min_frames)

    # get the image array with colored blocks
    for i, k in enumerate(in_list_of_arrays):
        k = k.astype(float)
        if i == 0:
            tmp1 = np.tile(np.repeat(k[inds], img_width, axis=0), (height, 1))
        else:
            tmp1 = np.vstack((tmp1, np.tile(np.repeat(k[inds], img_width, axis=0), (height, 1))))
    tmp1 = np.stack((tmp1,) * 3, axis=-1)

    for kk in color_numers_to_match:
        tmp3 = np.where(tmp1 == kk)
        for i1, i2 in zip(tmp3[0], tmp3[1]):
            tmp1[i1, i2, :] = color_dict[kk]

    tmp1 = tmp1.astype(int)
    with h5py.File(actual_h5_img_file, 'r') as h:
        tmp2 = image_tools.img_unstacker(h['images'][inds[0]:inds[-1] + 1], num_frames_wide=len(inds))
        tmp2 = np.vstack((tmp1, tmp2))
    return tmp2


"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""


def foo_rename(instr):
    if isinstance(instr, list):
        for i, k in enumerate(instr):
            instr[i] = foo_rename(k)
        return instr
    else:
        return '/Volumes/GoogleDrive-114825029448473821206/My Drive' + instr.split('My Drive')[-1]


def foo_arg_max_and_smooth(pred_bool_in, kernel_size_in, thresh_in, key_name_in):
    if pred_bool_in.shape[1] == 1:
        pred_bool_in = medfilt(copy.deepcopy(pred_bool_in), kernel_size=kernel_size_in)
        pred_bool_in = ((pred_bool_in > thresh_in) * 1).flatten()
    else:
        pred_bool_in = medfilt2d(copy.deepcopy(pred_bool_in), kernel_size=[kernel_size_in, 1])
        pred_bool_in = np.argmax(pred_bool_in, axis=1)
    L_key_ = '_'.join(key_name_in.split('__')[3].split(' '))
    with io.capture_output() as captured:  # prevent crazy printing
        pred_bool_out = utils.convert_labels_back_to_binary(pred_bool_in, L_key_)
    return pred_bool_out


def foo_arg_max_and_smooth(pred_bool_in, kernel_size_in, thresh_in, key_name_in, L_type_split_ind=3):
    pred_bool_out = medfilt_confidence_scores(pred_bool_in, kernel_size_in)
    pred_bool_out = confidence_score_to_class(pred_bool_out, thresh_in)
    L_key_ = '_'.join(key_name_in.split('__')[L_type_split_ind].split(' '))
    pred_bool_out = utils.convert_labels_back_to_binary(pred_bool_out, L_key_)
    return pred_bool_out


def medfilt_confidence_scores(pred_bool_in, kernel_size_in):
    if len(pred_bool_in.shape) == 1 or pred_bool_in.shape[1] == 1:
        pred_bool_out = medfilt(copy.deepcopy(pred_bool_in).flatten(), kernel_size=kernel_size_in)
    else:
        pred_bool_out = medfilt2d(copy.deepcopy(pred_bool_in), kernel_size=[kernel_size_in, 1])
    return pred_bool_out


def confidence_score_to_class(pred_bool_in, thresh_in):
    if len(pred_bool_in.shape) == 1 or pred_bool_in.shape[1] == 1:
        pred_bool_out = ((pred_bool_in > thresh_in) * 1).flatten()
    else:
        pred_bool_out = np.argmax(pred_bool_in, axis=1)
    #     NOTE: threshold is not used for the multi class models
    return pred_bool_out


h5_file_IMG = [
    '/Volumes/GoogleDrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_AH0407_160613_JC1003_AAAC/3lag/AH0407_160613_JC1003_AAAC_3lag.h5',
    '/Volumes/GoogleDrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_AH0667_170317_JC1241_AAAA/3lag/AH0667_170317_JC1241_AAAA_3lag.h5',
    '/Volumes/GoogleDrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_AH0698_170601_PM0121_AAAA/3lag/AH0698_170601_PM0121_AAAA_3lag.h5',
    '/Volumes/GoogleDrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_AH0705_171105_PM0175_AAAB/3lag/AH0705_171105_PM0175_AAAB_3lag.h5',
    '/Volumes/GoogleDrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_AH1120_200322__/3lag/AH1120_200322___3lag.h5',
    '/Volumes/GoogleDrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_AH1131_200326__/3lag/AH1131_200326___3lag.h5',
    '/Volumes/GoogleDrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_ANM234232_140118_AH1026_AAAA/3lag/ANM234232_140118_AH1026_AAAA_3lag.h5',
    '/Volumes/GoogleDrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_ANM234232_140120_AH1030_AAAA_a/3lag/ANM234232_140120_AH1030_AAAA_a_3lag.h5']
h5_file_IMG = foo_rename(h5_file_IMG)
# '/content/gdrive/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_test_transfer_learning_quick'
H5_file_ind = -1
# model_ind = 22
# model_ind = 45
# model_ind = 46
# model_ind = 23
# model_ind = 0
# model_ind = 2
# model_ind = 4
# model_ind = 1
# model_ind = 3
# model_ind = 5

# model_ind = 3

model_ind = 43
# model_ind = 42
# model_ind = 45
# model_ind = 12

threshold = .5
""" SET THHE SOURCE OF THE LABELS YOU ARE DRAWING FROM MAKE SURE TO ADJUST MODEL IND ACCORDINGLY"""
# to_pred_h5s = foo_rename('/Volumes/GoogleDrive/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED/')
to_pred_h5s = foo_rename('/content/gdrive/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED/')
# to_pred_h5s = foo_rename('/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED_retrain_10_20_30_tmp_save/')

H5_list_subset = utils.get_h5s(to_pred_h5s)
h5_file = H5_list_subset[H5_file_ind]
all_models = utils.lister_it(utils.print_h5_keys(h5_file, 1, 0), keep_strings='MODEL_3_', remove_string='viterbi')
# tmp1 = utils.lister_it(utils.print_h5_keys(h5_file, 1, 0), keep_strings='2021_12_06')
# (utils.print_h5_keys(h5_file, 0, 1)
# utils.print_list_with_inds(tmp1)
key_name = all_models[model_ind]
key_name2 = all_models[12]
print(h5_file)
print(key_name)

real_bool = image_tools.get_h5_key_and_concatenate(h5_file, '[0, 1]- (no touch, touch)')
trial_nums_and_frame_nums = image_tools.get_h5_key_and_concatenate(h5_file, 'trial_nums_and_frame_nums')
frame_nums = trial_nums_and_frame_nums[1, :].astype(int)
in_range = image_tools.get_h5_key_and_concatenate(h5_file, 'in_range')

# h5_img_dir = "/Volumes/GoogleDrive/My Drive/Colab data/curation_for_auto_curator/H5_data/"
# h5_file_IMG = utils.get_h5s(h5_img_dir, 0)

h5_file_IMG = utils.lister_it(h5_file_IMG, keep_strings=''.join(os.path.basename(h5_file).split('_ALT_LABELS'))[:-3])[0]

lines_thick = 20
# real_bool = x3.flatten()
real_bool[np.invert(in_range.astype(bool))] = -1



kernel_size = 7
pred_bool_temp = image_tools.get_h5_key_and_concatenate(h5_file, key_name)
pred_bool_smoothed = foo_arg_max_and_smooth(pred_bool_temp, kernel_size, threshold, key_name, L_type_split_ind = 5)
pred_bool_smoothed[np.invert(in_range.astype(bool))] = -1
# pred_bool_smoothed = medfilt(copy.deepcopy(pred_bool_temp), kernel_size=kernel_size)
# pred_bool_smoothed = (pred_bool_smoothed > threshold) * 1

pred_bool = foo_arg_max_and_smooth(pred_bool_temp, 1, threshold, key_name, L_type_split_ind = 5)
pred_bool[np.invert(in_range.astype(bool))] = -1

x2 = foo_heatmap_with_critical_errors(real_bool, pred_bool, in_range, frame_nums, title_str='regular, no smoothing')

# pred_bool = copy.deepcopy(pred_bool_smoothed)
# pred_bool[np.invert(in_range.astype(bool))] = -1
x2 = foo_heatmap_with_critical_errors(real_bool, pred_bool_smoothed, in_range, frame_nums, title_str='median smoothed')

print(h5_file)
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
pred_bool_temp_2 = image_tools.get_h5_key_and_concatenate(h5_file, key_name2)

pred_bool_temp_2 = np.argmax(pred_bool_temp_2, axis=1)

onsets = np.logical_or(pred_bool_temp_2 == 2, pred_bool_temp_2 == 3)
offsets = np.logical_or(pred_bool_temp_2 == 4, pred_bool_temp_2 == 5)
x3 = copy.deepcopy(x2)
for i, (k1, k2) in enumerate(utils.loop_segments(frame_nums)):
    tmpx = x3[i * lines_thick:i * lines_thick + lines_thick, :, :]
    tmp2 = np.zeros(tmpx.shape[1])
    tmp3 = onsets[k1:k2]
    tmp2[:len(tmp3)] = tmp3
    tmp2 = tmp2.astype(bool)
    tmpx[:, tmp2, :] = 0
    x3[i * lines_thick:i * lines_thick + lines_thick, :, :] = tmpx
    # # tmpx[onsets[k1:k2]] = 0
    # x3[k1:k2] = tmpx
plt.imshow(x3)

for i, (k1, k2) in enumerate(utils.loop_segments(frame_nums)):
    tmpx = x3[k1:k2]

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

real_bool_tmp = copy.deepcopy(real_bool)
real_bool_tmp[np.invert(in_range.astype(bool))] = 0
pred_bool_tmp = copy.deepcopy(pred_bool_smoothed)
# pred_bool_tmp = copy.deepcopy(pred_bool)
pred_bool_tmp[np.invert(in_range.astype(bool))] = 0

a = analysis.error_analysis(real_bool_tmp, pred_bool_tmp, frame_nums)
ind = 0
lines_thick = 20
x = a.all_errors_sorted[ind][0]
error_type = a.all_error_type_sorted[ind]

for i, (k1, k2) in enumerate(utils.loop_segments(frame_nums)):
    if x >= k1 and x < k2:
        print('trial num', i, error_type)
        y_line = i * lines_thick - lines_thick
        x_line = x - k1 - 1
        break

tmp3 = plot_segments_with_array_blocks(h5_file_IMG, [[100]], in_list_of_arrays=[real_bool_tmp, pred_bool_tmp],
                                       seg_num=0,
                                       color_numers_to_match=[0, 1], color_list=[0, .5], cmap_col='nipy_spectral',
                                       max_frames=20, min_frames=20)

new_shape = (x2.shape[1], np.round((x2.shape[1] / tmp3.shape[1]) * tmp3.shape[0]).astype(int))
tmp3 = cv2.resize(tmp3.astype('float32'), new_shape)

x3 = np.vstack((x2, tmp3.astype(int)))

fig1 = plt.figure(figsize=[15, 10])
img_1 = plt.imshow(x3)
plt.tight_layout()

marker_point, = plt.plot(x_line, y_line, 'vk', markersize=5)
marker_point.set_ydata(0)
marker_point.set_xdata(0)


def onclick(event):
    global ix, iy, lines_thick, marker_point, real_bool_tmp, pred_bool_tmp, x2, img_1, frame_nums
    ix, iy = event.xdata - lines_thick, event.ydata - lines_thick

    print('x = %d, y = %d' % (ix, iy))

    # global coords
    # coords.append((ix, iy))
    #
    # if len(coords) == 2:
    #     fig.canvas.mpl_disconnect(cid)
    # foo(frame_nums, lines_thick, x)
    x3 = np.round(ix)
    trial = int(np.floor(iy / lines_thick))
    y3 = np.sum(frame_nums[:trial + 1])
    ind = int(x3 + y3)
    marker_point.set_ydata(trial * lines_thick)
    marker_point.set_xdata(x3)
    print(x3, y3)

    tmp3 = plot_segments_with_array_blocks(h5_file_IMG, [[ind]], in_list_of_arrays=[real_bool_tmp, pred_bool_tmp],
                                           seg_num=0,
                                           color_numers_to_match=[0, 1], color_list=[0, .5], cmap_col='nipy_spectral',
                                           max_frames=20, min_frames=20)
    new_shape = (x2.shape[1], np.round((x2.shape[1] / tmp3.shape[1]) * tmp3.shape[0]).astype(int))
    tmp3 = cv2.resize(tmp3.astype('float32'), new_shape)

    x3 = np.vstack((x2, tmp3.astype(int)))
    img_1.set_data(x3)
    # marker_point, = plt.plot(ix, iy-lines_thick, '+k', markersize=5)
    # return coords


fig = plt.gcf()
cid = fig.canvas.mpl_connect('button_press_event', onclick)

_ = plt.text(50, 1317, 'Real\nPred', fontsize=5, color='w')

plt.figure()
_ = plt.hist(pred_bool_temp, 100)

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

#
#
# all_data = load_obj("/Users/phil/Downloads/all_data")
# all_data2 = []
# all_acc  =[]
# for i, k in enumerate(all_data):
#     n = k['full_name']
#     # print(n)
#
#     all_acc.append(np.max(k['all_logs'][:, 3]))
#     # if 'only on-off set' in n or 'overlap whisker on-off' in n or 'on-off set and one after' in n:
#     if all_acc[-1]>.99:
#
#         print(i)
#         print(n)
#         print(all_acc[-1])
#
# # for k in all_data2:
# #
# #     asdf
# #
# # utils.get_dict_info(k)
