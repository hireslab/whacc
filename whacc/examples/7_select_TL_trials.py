from whacc import utils, image_tools, transfer_learning, analysis
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
from tqdm.autonotebook import tqdm
from matplotlib import cm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.patches as mpatches
from scipy.signal import medfilt
import cv2
from tqdm import tqdm
from pathlib import Path


def find_trials_with_suspicious_predictions(frame_nums, pred_bool):
    all_lens = []
    for i, (k1, k2) in enumerate(utils.loop_segments(frame_nums)):
        vals = pred_bool[k1:k2]
        a, b = utils.group_consecutives(vals, step=0)
        y, x = np.histogram([len(k) for k in a], np.linspace(1, 5, 5))
        all_lens.append(y)
    all_lens = np.asarray(all_lens)
    tmp_weights = [3, 3, 2, 1]
    all_lens = all_lens * np.asarray(tmp_weights)
    sorted_worst_estimated_trials = np.flip(np.argsort(np.nanmean(all_lens, axis=1)))
    return sorted_worst_estimated_trials


def foo_rename(instr):
    if isinstance(instr, list):
        for i, k in enumerate(instr):
            instr[i] = foo_rename(k)
        return instr
    else:
        return '/Volumes/GoogleDrive-114825029448473821206/My Drive' + instr.split('My Drive')[-1]


def foo_open_folder(indir):
    indir = foo_rename(indir)
    if os.path.isfile(indir):
        indir = os.path.dirname(indir)
    utils.open_folder(indir)


def split_h5_based_on_trial_inds(h5_file, new_name, trial_inds_to_keep, in_range=None, color_channel=True,
                                 overwrite_if_file_exists=False, labels_key='labels'):
    frame_nums = image_tools.get_h5_key_and_dont_concatenate([h5_file], 'frame_nums')[0]
    h5_creator = image_tools.h5_iterative_creator(new_name,
                                                  overwrite_if_file_exists=overwrite_if_file_exists,
                                                  close_and_open_on_each_iteration=True,
                                                  color_channel=color_channel)
    new_frame_nums = []
    all_frames = np.asarray([])
    with h5py.File(h5_file, 'r') as h:
        LS = utils.loop_segments(frame_nums, returnaslist=True)
        for i in tqdm(trial_inds_to_keep):
            k1 = LS[0][i]
            k2 = LS[1][i]
            if in_range is not None:
                pole_up = in_range[k1:k2]
                pole_up = np.where(pole_up == 1)[0]
                k2 = k1 + pole_up[-1] + 1
                k1 = k1 + pole_up[0]

            all_frames = np.concatenate((all_frames, np.asarray(range(k1, k2))))
            h5_creator.add_to_h5(h['images'][k1:k2], h[labels_key][k1:k2])
            new_frame_nums.append(k2 - k1)

    all_frames = np.asarray(all_frames).flatten()
    frame_nums = np.asarray(new_frame_nums).flatten()

    with h5py.File(h5_creator.h5_full_file_name, 'r+') as h2:
        if 'all_frames' not in h2.keys():
            h2.create_dataset('og_frame_inds', data=all_frames)
        else:
            del h['all_frames']
            h2.create_dataset('og_frame_inds', data=all_frames)
        if 'frame_nums' not in h2.keys():
            h2.create_dataset('frame_nums', data=frame_nums)
        else:
            del h['frame_nums']
            h2.create_dataset('frame_nums', data=frame_nums)
    return h5_creator.h5_full_file_name

# basic variables
model_ind = 45
threshold = .5

h5_file = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED/AH1131_200326___ALT_LABELS.h5'
# H5 file with the model predictions in it
## tmp1 = utils.lister_it(utils.print_h5_keys(h5_file, 1, 0), keep_strings='MODEL_3_regular', remove_string='viterbi')
## key_name = tmp1[model_ind]


key_name = 'MODEL_3_regular 80 border aug 0 to 9__ResNet50V2__3lag__regular__acc_test max__epoch 6__L_ind3__LABELS_-_2021_07_22_06_01_29'
h5_file_IMG = '/Volumes/GoogleDrive-114825029448473821206/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_AH1131_200326__/3lag/AH1131_200326___3lag.h5'
# H5 file with the images in it

in_range = image_tools.get_h5_key_and_concatenate(h5_file, 'in_range')
# to crop out pole down times

# get the basic variables needed to pick trials
pred_bool = image_tools.get_h5_key_and_concatenate(h5_file, key_name)
pred_bool = utils.confidence_score_to_class(pred_bool, thresh_in = 0.5)
pred_bool[np.invert(in_range.astype(bool))] = -1

trial_nums_and_frame_nums = image_tools.get_h5_key_and_concatenate(h5_file, 'trial_nums_and_frame_nums')
frame_nums = trial_nums_and_frame_nums[1, :].astype(int)
# frame nums to cute the frames based on trials

sorted_worst_estimated_trials = find_trials_with_suspicious_predictions(frame_nums, pred_bool)
num_trials_ = 10

bd = h5_file_IMG.split('.h5')[0] + os.path.sep
new_name_ = bd+'new_h5_to_curate_by_hand.h5'
h5 = split_h5_based_on_trial_inds(h5_file_IMG, new_name_, sorted_worst_estimated_trials[:num_trials_],
                                             in_range=in_range,
                                             color_channel=True, overwrite_if_file_exists=True)

# split the sata into training and test
Path(bd).mkdir(parents=True, exist_ok=True)
split_h5s = image_tools.split_h5_loop_segments([h5],
                                               split_percentages=[.5, .5],  # **********$%$%$%$%$%%$
                                               temp_base_name=[bd + 'train',
                                                               bd + 'val'],
                                               chunk_size=10000,
                                               add_numbers_to_name=False,
                                               disable_TQDM=False,
                                               set_seed=0,
                                               color_channel=True,
                                               force_random_each_frame=False)# force_random_each_frame makes it super slow only needed for rare cases
