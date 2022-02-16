
from whacc import utils, image_tools, transfer_learning, analysis, error_analysis
from itertools import compress
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
from scipy.signal import medfilt
import cv2
import pickle
from IPython.utils import io
from scipy.signal import medfilt


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


all_data = load_obj('/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/all_data_10_20_30_TL')

to_pred_h5s = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED_retrain_10_20_30_tmp_save'
to_pred_h5s = utils.get_h5s(to_pred_h5s)

# get the label types for all the model predictions
key_dict = model_maker.label_naming_shorthand_dict()
for k in list(key_dict.keys()):
    k2 = key_dict[k]
    key_dict[k2] = k  # make the dict reverse of itself

m_names = utils.print_h5_keys(to_pred_h5s[0], return_list=True, do_print=False)
m_names = utils.lister_it(m_names,
                          keep_strings='MODEL_3')  # keep only the model predictions (not like labels and frame numbers)

label_key = []  # get label key types
for k in m_names:
    label_key.append(k.split('__')[-5])


def error_inds_save(a):
    name_list = ['e_append', 'e_deduct', 'e_ghost', 'e_join', 'e_miss', 'e_split']
    tmp_dict = dict()
    for k in name_list:
        exec("""tmp_dict['""" + k + """'] = a.""" + k)
    return tmp_dict

def error_inds_save(a):
    return [a.all_error_type_sorted, a.all_errors_sorted]

# tmp1 = [a.all_errors_sorted,a.all_error_type_sorted]

# import sys
# getattr(sys.modules[a], "e_append")

vit = dict()
vit['keep_inds'] = range(60)
len(to_pred_h5s)
x = [[] for i in range(8)]
vit['acc_no_pole_mask'] = copy.deepcopy(x)
vit['acc'] = copy.deepcopy(x)
vit['acc_smoothed'] = copy.deepcopy(x)

vit['error_analysis_no_pole_mask'] = copy.deepcopy(x)
vit['error_analysis'] = copy.deepcopy(x)
vit['error_analysis_smoothed'] = copy.deepcopy(x)

vit['m_name'] = copy.deepcopy(x)
vit['L_key'] = copy.deepcopy(x)

vit['h5_img_file_full_dir'] = to_pred_h5s
vit['h5_img_file'] = []
for k in vit['h5_img_file_full_dir']:
    vit['h5_img_file'].append(os.path.basename(k))

for i, h5_img_file in enumerate(to_pred_h5s):
    in_range = image_tools.get_h5_key_and_concatenate([h5_img_file], 'in_range')
    frame_nums = image_tools.get_h5_key_and_concatenate(h5_img_file, 'trial_nums_and_frame_nums')[1, :].astype(int)
    for iii, (m_name, L_key) in enumerate(tzip(m_names, label_key)):
        if iii in vit['keep_inds']:
            pred_m_raw = image_tools.get_h5_key_and_concatenate([h5_img_file], key_name=m_name)
            L_key = key_dict[L_key]
            real = image_tools.get_h5_key_and_concatenate([h5_img_file], key_name=L_key)
            if len(pred_m_raw.shape) == 1 or pred_m_raw.shape[1] == 1:
                pred_m = ((pred_m_raw > 0.5) * 1).flatten()
            else:
                pred_m = np.argmax(pred_m_raw, axis=1)  # turn into integers instead of percentages
            # get everything back to binary (if possible)
            with io.capture_output() as captured:  # prevents crazy printing
                pred_m_bool = utils.convert_labels_back_to_binary(pred_m, L_key)
                real_bool = utils.convert_labels_back_to_binary(real, L_key)
            # if real_bool is None:  # convert labels will return None if it cant convert
            #     # it back to the normal format. i.e. only onset or only offsets...
            #     pass
            if real_bool is not None:
                inds = in_range.astype(bool)
                cmp = real_bool == pred_m_bool
                vit['acc_no_pole_mask'][i].append(np.mean(cmp))
                vit['acc'][i].append(np.mean(cmp[inds]))
                pred_m_bool_smoothed = medfilt(pred_m_bool, kernel_size=7)
                cmp = real_bool == pred_m_bool_smoothed
                vit['acc_smoothed'][i].append(np.mean(cmp[inds]))

                vit['error_analysis_no_pole_mask'][i].append(error_inds_save(error_analysis(real_bool, pred_m_bool, frame_num_array=frame_nums)))
                vit['error_analysis'][i].append(error_inds_save(error_analysis(real_bool[inds], pred_m_bool[inds], frame_num_array=frame_nums)))
                vit['error_analysis_smoothed'][i].append(error_inds_save(error_analysis(real_bool[inds], pred_m_bool_smoothed[inds], frame_num_array=frame_nums)))

                vit['m_name'][i].append(m_names[iii])
                vit['L_key'][i].append(label_key[iii])
                # vit['h5_img_file_full_dir'].append(to_pred_h5s[iii])
                # vit['h5_img_file'].append(os.path.basename(vit['h5_img_file_full_dir'][-1]))
            else:
                vit['keep_inds'].pop(iii)
                vit['h5_img_file_full_dir'].pop(iii)
                vit['h5_img_file'].pop(iii)





# save_obj(vit, '/Users/phil/Downloads/vit_TL_10_20_30')

#### PLOT THE ACCURACY OF ALL THE TL EXAMPLES
vit = load_obj('/Users/phil/Downloads/vit_TL_10_20_30')

plt.plot(vit['acc'][-1], '.')
plt.ylim(.98, 1.001)

type(vit['error_analysis_no_pole_mask'][-1][0])


len(vit['acc'][0])





set1 = np.asarray([True if 'ANM234232 140120 AH1030' in k else False for k in vit['m_name'][0]]) # ind 7 test ind 6
set2 = np.asarray([True if 'AH1131 200326' in k else False for k in vit['m_name'][0]]) # ind 5 test in 4


set_choice = set2
set_num = 5 # 7 or 5
metric = 'acc_smoothed'
plt.figure()

for i, num_curated_trials in enumerate([10, 20, 30]):

    plt.subplot(1, 3, i+1)


    MN = str(num_curated_trials)+' __MobileNetV3Large'
    RN = str(num_curated_trials)+' __ResNet50V2'


    MN = np.asarray([True if MN in k else False for k in vit['m_name'][0]])
    RN = np.asarray([True if RN in k else False for k in vit['m_name'][0]])
    x = []
    for k in list(compress(vit['m_name'][0], set_choice*RN)):
        x.append(int(k.split('unfreeze')[-1][:2]))

    plt.plot(x, list(compress(vit[metric][set_num], set_choice*MN)), '.k') # black is mobilenet
    plt.plot(x, list(compress(vit[metric][set_num-1], set_choice*MN)), '*k') # star is unlearned set

    plt.plot(x, list(compress(vit[metric][set_num], set_choice*RN)), '.r') # red is resnet
    plt.plot(x, list(compress(vit[metric][set_num-1], set_choice*RN)), '*r')

    plt.xlabel('Num layers unfrozen')
    plt.ylabel('Accuracy')
    # mpatches

    plt.ylim([.95, 1.001])



###PLOT THE
set_choice = set2
set_num = 5 # 7 or 5

# set_choice = set1
# set_num = 7 # 7 or 5
metric = 'error_analysis_smoothed'

plt.figure()
def foo_critical_error(tmp1_in):
    num_critical_error = []
    for kk in tmp1_in:
        num_critical_error.append(len(kk[0]) - len([1 for k in kk[0] if k == 'append' or k == 'deduct']))
    return num_critical_error
for i, num_curated_trials in enumerate([10, 20, 30]):

    plt.subplot(1, 3, i+1)

    MN = str(num_curated_trials)+' __MobileNetV3Large'
    RN = str(num_curated_trials)+' __ResNet50V2'


    MN = np.asarray([True if MN in k else False for k in vit['m_name'][0]])
    RN = np.asarray([True if RN in k else False for k in vit['m_name'][0]])
    x = []
    for k in list(compress(vit['m_name'][0], set_choice*RN)):
        x.append(int(k.split('unfreeze')[-1][:2]))

    tmp1 = np.asarray(vit[metric][set_num][0][0])
    y = foo_critical_error(list(compress(vit[metric][set_num], set_choice*MN)))
    plt.plot(x, y, '.k') # black is mobilenet
    y = foo_critical_error(list(compress(vit[metric][set_num-1], set_choice*MN)))
    plt.plot(x, y, '*k') # star is unlearned set

    y = foo_critical_error(list(compress(vit[metric][set_num], set_choice*RN)))
    plt.plot(x, y, '.r') # red is resnet
    y = foo_critical_error(list(compress(vit[metric][set_num-1], set_choice*RN)))
    plt.plot(x, y, '*r')

    plt.xlabel('Num layers unfrozen')
    plt.ylabel('num critical errors')
    # mpatches

    plt.ylim([0, 500])



tmp1 = np.asarray(vit[metric][set_num][:][0]).flatten()




tmp1 = list(compress(vit[metric][set_num], set_choice*MN))
tmp1



# find the correct model ind to plot on the heat map
check_list = ['unfreeze 4', 'ResNet50V2', '__ 10 __', 'ANM234232 140120 AH1030']
check_list = ['unfreeze 4', 'MobileNetV3', '__ 10 __', 'ANM234232 140120 AH1030']

for i, k in enumerate(vit['m_name'][0]):
    test_true = []
    for kk in check_list:
        test_true.append(kk in k)
    if all(test_true):
        print(i, k)
