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

def split_h5_based_on_trial_inds(h5_file, new_name, trial_inds_to_keep,frame_nums,  in_range=None, color_channel=True,
                                 overwrite_if_file_exists=False, labels_key='labels'):

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


# sorted_worst_estimated_trials = find_trials_with_suspicious_predictions(frame_nums, pred_bool)
# new_name_ = '/Users/phil/Desktop/test_splitter.h5'
for H5_file_ind in [-1, -3]:
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

    # H5_file_ind = -1
    model_ind = 22
    model_ind = 45

    threshold = .5

    # to_pred_h5s = foo_rename('/Volumes/GoogleDrive/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED/')
    to_pred_h5s = foo_rename('/content/gdrive/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED/')

    H5_list_subset = utils.get_h5s(to_pred_h5s)
    h5_file = H5_list_subset[H5_file_ind]
    tmp1 = utils.lister_it(utils.print_h5_keys(h5_file, 1, 0), keep_strings='MODEL_3_regular', remove_string='viterbi')
    # tmp1 = utils.lister_it(utils.print_h5_keys(h5_file, 1, 0), keep_strings='2021_12_06')

    key_name = tmp1[model_ind]
    print(h5_file)
    print(key_name)
    h5_file_IMG = \
    utils.lister_it(h5_file_IMG, keep_strings=''.join(os.path.basename(h5_file).split('_ALT_LABELS'))[:-3])[0]

    trial_nums_and_frame_nums = image_tools.get_h5_key_and_concatenate(h5_file, 'trial_nums_and_frame_nums')
    frame_nums = trial_nums_and_frame_nums[1, :].astype(int)
    in_range = image_tools.get_h5_key_and_concatenate(h5_file, 'in_range')

    pred_bool = image_tools.get_h5_key_and_concatenate(h5_file, key_name).flatten()
    pred_bool = (copy.deepcopy(pred_bool) > threshold) * 1
    pred_bool[np.invert(in_range.astype(bool))] = -1

    real_bool = image_tools.get_h5_key_and_concatenate(h5_file, '[0, 1]- (no touch, touch)')
    sorted_worst_estimated_trials = find_trials_with_suspicious_predictions(frame_nums, pred_bool)
    """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
    """split select trials with 10 20 or 30 trials"""
    all_h5_to_replace_labels = []
    bd = '/Volumes/GoogleDrive-114825029448473821206/My Drive/Colab data/testing_TL_10_20_30/'
    for num_trials_ in [10, 20, 30]:
        new_name_ = bd + h5_file_IMG.split('DATA_FULL')[-1][:-3] + '___' + str(num_trials_) + '.h5'
        Path(os.path.dirname(new_name_)).mkdir(parents=True, exist_ok=True)
        h5out = split_h5_based_on_trial_inds(h5_file_IMG, new_name_, sorted_worst_estimated_trials[:num_trials_],frame_nums,
                                             in_range=in_range,
                                             color_channel=True, overwrite_if_file_exists=True)
        all_h5_to_replace_labels.append(h5out)
    """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
    """make test data form last 30 trials"""
    for num_trials_ in [30]:
        new_name_ = bd + h5_file_IMG.split('DATA_FULL')[-1][:-3] + '___TEST' + '.h5'
        Path(os.path.dirname(new_name_)).mkdir(parents=True, exist_ok=True)
        h5out = e(h5_file_IMG, new_name_, sorted_worst_estimated_trials[num_trials_:],frame_nums,
                                             in_range=in_range,
                                             color_channel=True, overwrite_if_file_exists=True)
        all_h5_to_replace_labels.append(h5out)
    """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
    """replace with the correct labels"""
    for h5 in all_h5_to_replace_labels:
        with h5py.File(h5, 'r+') as h:
            inds = h['og_frame_inds'][:].astype(int)
            h['labels'][:] = real_bool[inds]
    """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
    """split data into test and training"""
    h5s_to_split = utils.get_h5s('/Volumes/GoogleDrive-114825029448473821206/My Drive/Colab data/testing_TL_10_20_30')
    h5s_to_split = utils.lister_it(h5s_to_split, remove_string=['train', 'val', 'TEST'])
    for h5 in h5s_to_split:
        bd = h5.split('.h5')[0] + '/'
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
                                                       force_random_each_frame=False)

# a = analysis.pole_plot(tmp1)
from whacc import touch_gui, image_tools
import numpy as np
import os

tmp1 = '/Volumes/GoogleDrive-114825029448473821206/My Drive/Colab data/testing_TL_10_20_30/data_AH1131_200326__/3lag/AH1131_200326___3lag___10.h5'

get = image_tools.get_h5_key_and_concatenate

touch_gui(tmp1, 'labels')

get = image_tools.get_h5_key_and_concatenate
np.mean(get(tmp1, 'labels'))

for k in utils.get_h5s('/Volumes/GoogleDrive-114825029448473821206/My Drive/Colab data/testing_TL_10_20_30/',
                       print_h5_list=False):
    print('/'.join(k.split('/')[-3:]))
    print(100 * np.mean(get(k, 'labels')))
    print('\n')

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
tmp1 = '/Volumes/GoogleDrive-114825029448473821206/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_AH0667_170317_JC1241_AAAA/3lag/AH0667_170317_JC1241_AAAA_3lag.h5'
x = image_tools.get_h5_key_and_concatenate([tmp1], '[0, 1]- (no touch, touch)')

# print(*h_names, sep = '\n')
x = x[11997 + 900:11997 + 1000]  # new correctd touches need ot be near 40 here and not near 0
plt.plot(x)

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

import shutil
import os

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


shutil.copytree('/Users/phil/Desktop/tmp1/',
                '/Users/phil/Desktop/aasdf/',
                ignore=ignore_files)


bdir = '/Volumes/GoogleDrive-114825029448473821206/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/'
bdir = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED_retrain_10_20_30'
new_dir = '/Users/phil/Desktop/fake_ALT_LABELS_FINAL_PRED_retrain_10_20_30/'
files = utils.get_files(bdir, '*.h5')
for f in files:
    f2 = new_dir.join(f.split(bdir))
    x = os.path.dirname(f2)
    Path(x).mkdir(parents=True, exist_ok=True)
    # print(f2.split('.')[0]+'.txt')
    f3 = open(f2.split('.')[0]+'.h5', "w+")



def load_model_data(all_models_directory):
  data_files = utils.get_files(all_models_directory, '*model_eval_each_epoch.json')
  to_split = '/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/'
  all_data = []
  for i, k in enumerate(data_files):
    if '/small_h5s/' not in k and '/TL_mods_10_20_30_unfreeze_' in k:
      with open(k, 'r') as f:
        a = json.load(f)
      a['all_logs'] = np.asarray(a['all_logs'])
      new_log_names = []
      for k2 in a['logs_names']:
        new_log_names.append(k2.split('bool_')[-1])
      a['logs_names'] = new_log_names
      a['logs_names'] = np.asarray(a['logs_names'])

      k.split(to_split)[-1]
      a['full_name'] = '__'.join(' '.join(k.split(to_split)[-1].split('/2021')[0].split('_')).split('/'))
      a['dir'] = os.path.dirname(k)
      all_data.append(a)
  for i, k in enumerate(all_data):
    try:
      all_data[i]['info'] = reload_info_dict(k['dir'])
    except:
      print('did not work...-->', k['dir'])
  return all_data

bd2 = "/content/gdrive/My Drive/colab_data2/"
all_models_directory = bd2+"/model_testing/all_data/all_models/"
all_data = load_model_data(all_models_directory)
