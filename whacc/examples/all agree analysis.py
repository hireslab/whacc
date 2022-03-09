import copy
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import medfilt, medfilt2d

from whacc import image_tools
from whacc import utils
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

def foo_rename(instr):
    if isinstance(instr, list):
        for i, k in enumerate(instr):
            instr[i] = foo_rename(k)
        return instr
    else:
        return '/Volumes/GoogleDrive-114825029448473821206/My Drive' + instr.split('My Drive')[-1]


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


def foo_arg_max_and_smooth(pred_bool_in, kernel_size_in, thresh_in, key_name_in, L_key_=None, L_type_split_ind=None):
    pred_bool_out = medfilt_confidence_scores(pred_bool_in, kernel_size_in)
    pred_bool_out = confidence_score_to_class(pred_bool_out, thresh_in)
    if L_key_ is None:
        L_key_ = '_'.join(key_name_in.split('__')[L_type_split_ind].split(' '))

    pred_bool_out = utils.convert_labels_back_to_binary(pred_bool_out, L_key_)
    return pred_bool_out


"""##################################################################################################################"""
"""##################################################################################################################"""
key_name = 'model_5____3lag__regular_labels__MODEL_FULL_TRAIN_rollSTD_3_7_rollMEAN_3_7_shift_neg2_to_2_val_with_real_val_PM_JC_LIGHT_GBM'
h5_file = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED/AH0407_160613_JC1003_AAAC_ALT_LABELS.h5'

kernel_size = 7
threshold = 0.5
L_key_ = '[0, 1]- (no touch, touch)'

"""##################################################################################################################"""
"""##################################################################################################################"""
pred_bool_temp = image_tools.get_h5_key_and_concatenate(h5_file, key_name)
pred_bool_temp = pred_bool_temp.astype(float)
pred_bool_smoothed = foo_arg_max_and_smooth(pred_bool_temp, kernel_size, threshold, key_name, L_key_=L_key_)

real_bool = image_tools.get_h5_key_and_concatenate(h5_file, '[0, 1]- (no touch, touch)')
trial_nums_and_frame_nums = image_tools.get_h5_key_and_concatenate(h5_file, 'trial_nums_and_frame_nums')
frame_nums = trial_nums_and_frame_nums[1, :].astype(int)

in_range = image_tools.get_h5_key_and_concatenate(h5_file, 'in_range')
"""##################################################################################################################"""
"""##################################################################################################################"""

all_h5s = utils.get_h5s(foo_rename('/content/gdrive/My Drive/Colab data/curation_for_auto_curator/finished_contacts/'),
                        print_h5_list=False)
h_cont, h_names = utils._get_human_contacts_(all_h5s)
"""##################################################################################################################"""
"""##################################################################################################################"""
human = h_cont[0][:, in_range.astype(bool)]
whacc = pred_bool_smoothed[in_range.astype(bool)]

human_mean = np.mean(human, axis=0)


def foo_agree_0_1(in1, in2):
    full0agree = np.mean(1 * (in1 == 0) == 1 * (in2 == 0))
    full1agree = np.mean(1 * (in1 == 1) == 1 * (in2 == 1))
    return full0agree, full1agree


# np.mean(full0agree)*100, np.mean(full1agree)*100
d = {'h0': [], 'h1': [],'w0': [], 'w1': []}
for i, a in enumerate(human):
    inds = np.ones(human.shape[0]).astype(bool)
    # inds[i] = False
    tmp_mean = np.mean(human[inds], axis=0)
    tmp1, tmp2 = foo_agree_0_1(a, tmp_mean)
    d['h0'].append(tmp1)
    d['h1'].append(tmp2)
    tmp1, tmp2 = foo_agree_0_1(whacc, tmp_mean)
    d['w0'].append(tmp1)
    d['w1'].append(tmp2)

df = pd.DataFrame(d)

ax = sns.pointplot(x="h0", y="w0", data=df)




# np.mean(full0agree)*100, np.mean(full1agree)*100
d = {'percent correct':[], 'ind':[], 'label':[], 'class':[]}
for i, a in enumerate(human):
    inds = np.ones(human.shape[0]).astype(bool)
    inds[i] = False
    tmp_mean = np.mean(human[inds], axis=0)
    d['percent correct']+=list(foo_agree_0_1(a, tmp_mean))
    d['percent correct']+=list(foo_agree_0_1(whacc, tmp_mean))
    d['ind']+=[i]*4
    d['label']+=['human', 'human', 'WhACC', 'WhACC']
    d['class']+=['no touch', 'touch']*2

df = pd.DataFrame(d)
plt.figure(figsize=(7, 6))
ax = sns.pointplot(x="label", y="percent correct", hue="class", data=df, dodge=True)

plt.figure(figsize=(7, 6))
ax = sns.pointplot(x="label", y="percent correct", data=df, dodge=True)

plt.figure(figsize=(7, 6))
for k in range(3):
    ax = sns.pointplot(x="label", y="percent correct", data=df.loc[df['ind'] == k], dodge=True, ci=None)




def foo_agree_all(in1, in2):
    return np.mean(np.append([1 * (in1 == 0) == 1 * (in2 == 0)], [1 * (in1 == 1) == 1 * (in2 == 1)]))

d = {'percent correct':[], 'ind':[], 'label':[], 'class':[]}
for i, a in enumerate(human):
    inds = np.ones(human.shape[0]).astype(bool)
    inds[i] = False
    tmp_mean = np.mean(human[inds], axis=0)
    d['percent correct'].append(foo_agree_all(a, tmp_mean))
    d['percent correct'].append(foo_agree_all(whacc, tmp_mean))
    d['ind']+=[i]*2
    d['label']+=['human', 'WhACC']
    d['class']+=['all']*2


df = pd.DataFrame(d)
plt.figure(figsize=(7, 6))
ax = sns.pointplot(x="label", y="percent correct", data=df, dodge=True, ci = False, color='k', legend='sadf')

for k in range(3):
    ax = sns.pointplot(x="label", y="percent correct", data=df.loc[df['ind'] == k], dodge=True, ci=None, color='b')

custom_lines = [Line2D([0], [0], color='k', lw=4),
                Line2D([0], [0], color='b', lw=4)]
plt.legend(custom_lines, ['mean', 'individual'])
