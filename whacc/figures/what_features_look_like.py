

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

from whacc.figures import pretty_plots
pretty_plots


def load_selected_features(h5_in, feature_list = None, feature_index=None, time_index=None, return_labels = False, label_key = 'labels'):
    """

    Parameters
    ----------
    h5_in : full directory (or list of full directories of multiple h5s) of the full feature h5 file 84,009
    feature_index : bool array of len 84,009

    Returns
    -------

    """

    def indexit(arr, index):
        out = []
        for k in index:
            out.append(arr[k])
        return np.asarray(out)

    def extract_indexed_features(x, all_x_in, feature_index_in):
        if len(x.shape) > 1:
            inds = feature_index_in[:x.shape[1]]
            del feature_index_in[:x.shape[1]]
            x = x[:, inds]
            if all_x_in is None: # init
                all_x_in = x
            else:
                all_x_in = np.hstack((all_x_in, x))
        else:
            inds = feature_index_in.pop(0)  # single true or false to include the "TOTAL" variables
            if inds:
                if all_x_in is None: # init
                    all_x_in = x[:, None]
                else:
                    all_x_in = np.hstack((all_x_in, x[:, None]))
        return all_x_in, feature_index_in

    d = utils.load_feature_data()
    if feature_list is None:
        feature_list = d['feature_list_unaltered']
    if feature_index is None:
        feature_index= d['final_selected_features_bool']
        # feature_index = utils.get_selected_features(greater_than_or_equal_to=4)
    feature_index = utils.make_list(feature_index)
    feature_index = copy.deepcopy(feature_index)

    if isinstance(h5_in, list):
        all_x = []
        all_y = []
        for k in h5_in:
            tmp_x, tmp_y = load_selected_features(k, feature_list, feature_index, time_index, return_labels, label_key)
            all_x.append(tmp_x)
            all_y.append(tmp_y)
            del tmp_x, tmp_y

        all_x = np.vstack(all_x)
        if return_labels:
            all_y = np.hstack(all_y)
            return all_x, all_y
        return all_x

    all_x = None
    for k in tqdm(feature_list):
        if time_index is not None:
            x = []
            with h5py.File(h5_in, 'r') as h:
                for i in time_index:
                    x.append(h[k][i])
            x = np.asarray(x)
        else:
            x = image_tools.get_h5_key_and_concatenate(h5_in, k).astype('float32')

        all_x, feature_index = extract_indexed_features(x, all_x, feature_index)

    if return_labels:
        all_y = image_tools.get_h5_key_and_concatenate(h5_in, label_key).astype('float32')
        all_y = np.hstack(all_y)
        if time_index is not None:
            all_y = indexit(all_y, time_index)
        return all_x, all_y
    return all_x




"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$ _______________________ $$$$$$$$$$$$$$$$$$$$$$$$$$$"""
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


h5_in = '/Users/phil/Desktop/LIGHT_GBM/reg_80_border_and_andrew_both_samson_both__test_num_0_percent_15.h5'
mod_fn = '/Users/phil/Desktop/LIGHT_GBM/model_saves/model_num_0_of_10.pkl'

with open(mod_fn, 'rb') as f:
    model = pickle.load(f)

explainer = shap.TreeExplainer(model)

feature_list = natsorted(utils.lister_it(utils.print_h5_keys(h5_in, 1, 0), keep_strings='FD__'))
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
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$ SELECTING AN INFORMATIVE TOUCH TRACE $$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

h5_in = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/3lag/train_3lag.h5'
# here I am look for a good slice of touches in the orignal H5 file
# training
383810 # aug - 3 border X 10
292217 # reg - 80 border (676027 - 383810)
y = image_tools.get_h5_key_and_concatenate(h5_in)[383810:]
smooth_by = 151
ind = 0

y2 = np.convolve(np.diff(y), np.ones(smooth_by)/smooth_by)
bst_inds = np.flip(np.argsort(y2))
add_to = smooth_by//2
fig, ax = plt.subplots(5, 4)
for ind, ax2 in enumerate(ax.flatten()):
    ind+=100
    ax2.plot(y[bst_inds[ind]-add_to:bst_inds[ind]+add_to])

ind = 115
add_to = 200
plt.plot(y[bst_inds[ind]-add_to:bst_inds[ind]+add_to])

plt.plot(y[271792:271792+200])

plt.plot(y[271792:271792+200-50])
# 271792+383810

with h5py.File(h5_in, 'r') as h:
    test_image = h['images'][bst_inds[ind]+383810]
plt.imshow(test_image)
# CHECKING THE MAKE SURE THE IMAGES LOOK GOOD
from whacc import analysis
pp = analysis.pole_plot(h5_in)
pp.current_frame = bst_inds[ind]+383810-20
pp.plot_it()
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
        h2['frame_nums'] = [len(h2['labels'][:])]

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

# from natsort import natsorted, ns
# xxxx = natsorted(xxxx, alg=ns.REAL)
d = utils.load_obj('/Users/phil/Dropbox/HIRES_LAB/GitHub/whacc/whacc/whacc_data/feature_data/feature_data_dict.pkl')

fi_path ='/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/feature_index_for_feature_selection/only_gain_more_than_2_features_bool_2105_features.npy'
feature_index = np.load(fi_path)

# all_x, all_y = load_selected_features(h5_feature_data, feature_list = d['feature_list_unaltered'], feature_index=feature_index, return_labels = True, label_key = 'labels')

with h5py.File(h5_feature_data ,'r') as h:
    all_x = []
    for k in d['feature_list_unaltered']:
        x = h[k][:]
        if len(x.shape)==1:
            x = x[:, None]
        all_x.append(x)
all_x = np.hstack(all_x)

all_x = all_x[:, feature_index]
# all_x = utils.load_selected_features(h5_feature_data)

all_x.shape

with h5py.File(h5_feature_data, 'r+') as h:
    h['final_selected_features_TEMP'] = all_x



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
"""$$$$$$$$$$$$$$$$$$$ PLOT 3095 -- final features features   $$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""


def foo_norm(x):
    x = x-np.min(x)
    x = x/np.max(x)
    return x
# FD = d['final_3095_features']
FD = d['final_selected_features_TEMP']
FD[0, :] = FD[0, :]+0.0000001 # to avoid true divide error
# FD = FD[:, ::10]
labels = d['labels']
FD2 = copy.deepcopy(FD)
for i, k in enumerate(FD.T):
    FD2[:, i] = foo_norm(k)
FD = FD2
# FD[0, :] = FD[0, :]+0.0000001 # to avoid true divide error
cc = np.corrcoef(FD, rowvar=False)
cc, inds = cluster_corr(cc, 1.5)
data = FD[:, inds].T
sns.heatmap(cc)

fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [10, 1]})
plt.sca(ax[0])

ax_heat = sns.heatmap(data, cbar=False)
# ax_heat.set_yticklabels(key_labels, fontsize= 7)

# x = np.linspace(0, 40, 5).astype(int)
# ax_heat.set_yticks(x)
# ax_heat.set_yticklabels(x)
plt.ylabel('6730 feature')
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








# sns.heatmap(total_type_features)
#
# sns.heatmap(FD)
#
# lu = []
# for k in FD.T:
#     lu.append(len(np.unique(k)))
# plt.plot(np.sort(lu))
#
# np.sum(np.asarray(lu)==1)
#
# from whacc import analysis
# file_name = 'ANM234232_140120_AH1030_AAAA_a_3lag.h5'
# file_name = 'AH0407_160613_JC1003_AAAC_3lag.h5'
# file_name = utils.lister_it(utils.get_h5s('/Volumes/GoogleDrive-114825029448473821206/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/'), file_name)
#
#
# a = analysis.pole_plot(file_name[0])
# ind = 14467 # 14467 to 14567
# a.current_frame = ind
# a.plot_it()
#
# cut_index = [ind-150, ind+150]
# plt.plot(np.arange(*cut_index), a.true_val[cut_index[0]:cut_index[1]])
#
#
# cut_index = [ind+70, ind+140]  # [14537, 14607]
# print(np.diff(cut_index))
# plt.plot(np.arange(*cut_index), a.true_val[cut_index[0]:cut_index[1]])
#
#
#
#
#
#
#
# def load_data(h5_in, feature_list, start_ind, end_ind, label_key='labels'):
#     if isinstance(h5_in, list):
#         all_x = [];
#         all_y = []
#         for k in h5_in:
#             tmp_x, tmp_y = load_data(k, feature_list, split_int_n_chunks=start_ind,
#                                      split_chunk_ind=end_ind, label_key=label_key)
#             all_x.append(tmp_x)
#             all_y.append(tmp_y)
#             del tmp_x, tmp_y
#
#         all_x = np.vstack(all_x)
#         all_y = np.hstack(all_y)
#         return all_x, all_y
#     all_x = None
#     for k in feature_list:
#         if all_x is None:
#             all_x = image_tools.get_h5_key_and_concatenate(h5_in, k)[start_ind:end_ind].astype('float32')
#         else:
#             x = image_tools.get_h5_key_and_concatenate(h5_in, k)[start_ind:end_ind].astype('float32')
#             if len(x.shape) > 1:
#                 all_x = np.hstack((all_x, x))
#             else:
#                 all_x = np.hstack((all_x, x[:, None]))
#
#     all_y = image_tools.get_h5_key_and_concatenate(h5_in, label_key)[start_ind:end_ind].astype('float32')
#     return all_x, all_y
#
#
#
# ## a) use this range [14537, 14607]
# base_name = 'AH0407_160613_JC1003_AAAC_3lag.h5'
# # /Users/phil/Dropbox/HIRES_LAB/curation_for_auto_curator/Data/Jon/AH0407/160613/AH0407x160609-144.mp4
# img_h5 = utils.lister_it(utils.get_h5s('/Volumes/GoogleDrive-114825029448473821206/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/'), base_name)
# feature_h5 = utils.lister_it(utils.get_h5s('/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/FEATURE_DATA/colab data/curation_for_auto_curator/DATA_FULL_in_range_only/data_AH0407_160613_JC1003_AAAC/3lag/'), base_name)
# inds = [14419, 16417]
# with h5py.File(img_h5[0], 'r') as h:
#     images = h['images'][inds[0]:inds[1]]
#     labels = h['labels'][inds[0]:inds[1]]
#     frame_nums = np.asarray([len(labels)])
# with h5py.File(feature_h5[0], 'r') as h:
#     FD__original = h['FD__original'][inds[0]:inds[1]]
#
# with h5py.File('/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/example_feature.h5', 'w') as h:
#     h['images'] = images
#     h['labels'] = labels
#     h['frame_nums'] = frame_nums
#     h['FD__original'] = FD__original
#     h['base_name'] = base_name.encode("ascii", "ignore")
#     h['img_h5'] = img_h5[0].encode("ascii", "ignore")
#     h['feature_h5'] = feature_h5[0].encode("ascii", "ignore")
#
#
# #
# # for i, (i1, i2) in enumerate(utils.loop_segments(frame_nums)):
# #     if i1<=14587 and i2>14587:
# #         print(i)
# #         print(i1, i2)
# #         break
# #
# # frame_nums[12]
# #
# #
# #
# # feature_list = natsorted(utils.lister_it(utils.print_h5_keys(h5_in, 1, 0), keep_strings='FD__'))
# # final_feature_names, feature_names, feature_nums, feature_list_short = get_feature_data_names(feature_list)
# # all_x, all_y = load_data(feature_h5[0], feature_list, inds[0], inds[1], label_key='labels')
# #
# # all_y[inds[0]:inds[1]]
# # utils.print_h5_keys(feature_h5[0])
# ## b) get full 2048 features from resnet heatmap it
#
# ## c) take one example neuron and plot all normal feature transformations, group the transforms base on type (shift,
# # SD, diff etc.)
#
# ## d) use b) to show how I do the TOTAL transforms
