"""
for this not sure what I want, I think since I am just using SHAP as a >0 for 3/10 models, I will think about this and
plot it when I know what I want, also gain plots could be useful but this is super not interesting and not the main
point so it doesnt have to be perfect

"""
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

shap_values = explainer.shap_values(x_train)
shap_values_1 = shap_values[1]

shap.summary_plot(shap_values_1, x_train, feature_names=final_feature_names, max_display=20)

shap_feature_mean = np.nanmean(np.abs(shap_values_1), axis=0)

_, bi = utils.lister_it(final_feature_names, 'FD_TOTAL', return_bool_index=True)
shap_feature_mean_TOTAL = shap_feature_mean[bi]
shap_feature_mean_other = shap_feature_mean[bi == 0]
final_feature_names_other = np.asarray(final_feature_names)[bi == 0]

num_neurons = 2048
neurons_mean = []
for k in range(num_neurons):
    neurons_mean.append(np.nanmean(shap_feature_mean_other[k::num_neurons]))
feature_mean = []
for k in range(len(shap_feature_mean_other) // num_neurons):
    feature_mean.append(np.nanmean(shap_feature_mean_other[k * num_neurons: k * num_neurons + num_neurons]))

max_bin = 1.0001 * np.max(feature_mean + neurons_mean + list(shap_feature_mean_TOTAL))

bins = np.linspace(1 * 10 ** -12, max_bin, 100)
label_names = ['Feature', 'Neurons', 'TOTAL feature']
tmp1 = plt.hist([feature_mean, neurons_mean, shap_feature_mean_TOTAL], bins,
                label=label_names)
list_of_each = [feature_mean, neurons_mean, list(shap_feature_mean_TOTAL)]
group_name = np.concatenate([[name] * len(k) for name, k in zip(label_names, list_of_each)])
df = pd.DataFrame(zip(np.concatenate(list_of_each), group_name), columns=['mean value', 'group'])

tmp1 = df.groupby('group')
tmp1.plot.kde()
tmp1 = df.sort_values('mean value')
tmp1.plot()

for k, name in zip(list_of_each, label_names):
    x = np.arange(len(k))
    x = x / x[-1]
    plt.plot(sorted(k), x, '.', label=name)
plt.legend()

print(np.round(100 * np.mean(shap_feature_mean > 0), 2), '%')

num_neurons = 2048
neurons_not_0 = []
for k in range(num_neurons):
    neurons_not_0.append(np.sum(shap_feature_mean_other[k::num_neurons] > 0))
feature_not_0 = []
num_transforms = len(shap_feature_mean_other) // num_neurons
for k in range(num_transforms):
    feature_not_0.append(np.sum(shap_feature_mean_other[k * num_neurons: k * num_neurons + num_neurons] > 0))

bins = np.linspace(0, .45, 45)
plt.hist([np.asarray(neurons_not_0) / num_transforms], bins=bins)
# plt.xlim([0, .42])
plt.figure()
plt.hist([np.asarray(feature_not_0) / num_neurons], bins=bins)
# plt.xlim([0, .42])


# final_feature_names, feature_names, feature_nums, feature_list_short
np.argmax(np.asarray(neurons_not_0) / num_transforms)

tmp1 = np.flip(np.argsort(np.asarray(feature_not_0) / num_neurons))

x = np.asarray(feature_not_0) / num_neurons
for k in tmp1:
    tmp2.append()
    print(feature_list_short[k])
    print(feature_not_0[k])







h5_in = '/Users/phil/Desktop/LIGHT_GBM/reg_80_border_and_andrew_both_samson_both__test_num_0_percent_15.h5'




for n in feature_list:
    x = n.split('____')[0][-3:]
    i = np.where([k in '-0123456789' for k in x])[0][0]
    print(x[i:])
    # print(np.asarray(x)[[k in '-0123456789' for k in x]])
    # asdf




#
# with h5py.File(h5_in, 'r') as h:
#    for k in h.keys():
#        if 'FD__' in k:
#            print(k)
h5_final_keys_sorted = utils.print_h5_keys(h5_in, 1, 0)




feature_list == natsorted(utils.lister_it(utils.print_h5_keys(h5_in, 1, 0), keep_strings='FD__')) # I do use os sorted here
final_feature_names, feature_names, feature_nums, feature_list_short = get_feature_data_names(feature_list)
# directories with data and models
bd = '/Volumes/rig1 EPHUS pc backup /LIGHT_GBM/FEATURE_DATA'
shap_contribution_list = []
shap_len_samples_list = []
fn_list = []
mod_list = []
all_models = utils.get_files('/Users/phil/Desktop/LIGHT_GBM/model_saves/', '*.pkl')
for mod_fn in tqdm(all_models):
    with open(mod_fn, 'rb') as f:
        model = pickle.load(f)
    # load model
    explainer = shap.TreeExplainer(model)
    for k in tqdm(model.info_dict['val_data_files']):
        mod_list.append(mod_fn.split('/')[-1].split('_of')[0])
        # load data
        k = k.split('\\')[-1].split('/')[-1]
        h5_in = bd + os.sep + k
        fn_list.append(h5_in)
        x_train, y_train = load_data(h5_in, feature_list, split_int_n_chunks=1, split_chunk_ind=0)
        shap_values = explainer.shap_values(x_train)
        shap_values_1 = shap_values[1]
        shap_contribution_list.append(np.nanmean(np.abs(shap_values_1), axis=0))
        shap_len_samples_list.append(shap_values_1.shape[0])

df = pd.DataFrame(zip(shap_contribution_list, shap_len_samples_list, fn_list, mod_list), columns= ['shap', 'len_x', 'file name', 'mod name'])
df['']
utils.save_obj(df, '/Users/phil/Desktop/all_model_shap_dataframe')
tmp1 = df.groupby(['mod name'])
tmp1['shap'].mean()

tmp1 = []
tmp2 = np.asarray(shap_contribution_list)
for mn in np.unique(mod_list):
    tmp1.append(np.nanmean(tmp2[np.asarray(mod_list)==mn, :], axis = 0))

tmp1 = np.asarray(tmp1)
print(tmp1.shape)

tmp5 = np.sum(tmp1>0, axis=0)

plt.imshow(tmp1[:, :100])

tmp3 = []
for k in tmp1:
    tmp3.append(np.where(k>0)[0])
tmp3 = np.concatenate(tmp3)

out_0 = plt.hist(tmp3, bins = np.arange(-.5, 84009+.5, 1))

features_out_of_10 = copy.deepcopy(out_0[0])

out_1 = plt.hist(tmp3, bins = np.arange(-.5, 84009+.5, 2048))

out_2 = plt.hist(tmp3, bins = np.arange(-.5, 84009+.5, 2048))



for k in range(10):
    print(np.round(100 * np.mean(features_out_of_10 > k), 2), '% have ', k+1, ' or more')
    print('count ', np.sum(features_out_of_10 > k))



np.save('/Users/phil/Desktop/LIGHT_GBM/features_out_of_10', features_out_of_10)

greater_than_or_equal_to = 10
np.sum(features_out_of_10 >= greater_than_or_equal_to)
best_features = np.where(features_out_of_10[:-41] >= greater_than_or_equal_to)[0]
best_TOTAL_features = len(features_out_of_10[:-41])+np.where(features_out_of_10[-41:] >= greater_than_or_equal_to)[0]

print(best_features//2048)
print(sorted(best_features%2048))

print(np.unique(best_features//2048))
print(np.unique(best_features%2048))

print(len(np.unique(best_features//2048)))
print(len(np.unique(best_features%2048)))


feature_inds = np.unique(best_features//2048)
neuron_inds = np.unique(best_features%2048)

keep_features_index = np.repeat(neuron_inds, len(feature_inds)).reshape([len(neuron_inds), len(feature_inds)])
keep_features_index = keep_features_index + feature_inds*2048 - 2048
keep_features_index = np.append(keep_features_index.flatten(),  best_TOTAL_features)
# final_feature_names, feature_names, feature_nums, feature_list_short

tmp1 = [features_out_of_10[k]for k in keep_features_index]


for k in np.unique(tmp1//2048):
    print(feature_list_short[k])


final_feature_names_TOTAL, bi = utils.lister_it(final_feature_names, 'FD_TOTAL', return_bool_index=True)
features_out_of_10_TOTAL = features_out_of_10[bi]
features_out_of_10_other = features_out_of_10[bi == 0]
final_feature_names_other = np.asarray(final_feature_names)[bi == 0]


num_neurons = 2048
neurons_not_0 = []
for k in range(num_neurons):
    neurons_not_0.append(np.sum(features_out_of_10_other[k::num_neurons] > 0))
feature_not_0 = []
num_transforms = len(features_out_of_10_other) // num_neurons
for k in range(num_transforms):
    feature_not_0.append(np.sum(features_out_of_10_other[k * num_neurons: k * num_neurons + num_neurons] > 0))


# bins = np.linspace(0, .45, 45)

neurons_not_0_tmp = sorted(np.asarray(neurons_not_0)/num_transforms)
plt.bar( range(len(neurons_not_0_tmp)), neurons_not_0_tmp)
plt.grid(b=True, which='both')
plt.minorticks_on()
plt.ylim([0, .92])

plt.figure()
feature_not_0_tmp = sorted(np.asarray(feature_not_0)/num_neurons)
plt.bar( range(len(feature_not_0_tmp)), feature_not_0_tmp)
plt.grid(b=True, which='both')
plt.minorticks_on()
plt.ylim([0, .92])

plt.figure()
plt.bar( range(len(features_out_of_10_TOTAL)), features_out_of_10_TOTAL/10)
plt.grid(b=True, which='both')
plt.minorticks_on()
plt.ylim([0, .92])






mean_50 = []
for k in range(1+len(neurons_not_0_tmp)//50):
    mean_50.append(np.mean(neurons_not_0_tmp[k*50:k*50+50]))
plt.figure()
plt.bar(np.arange(len(mean_50)),mean_50)
plt.grid(b=True, which='both')
plt.minorticks_on()
plt.ylim([0, .92])




plt.figure()
plt.plot(range(len(feature_not_0_tmp)), feature_not_0_tmp, '.k', label='augmented features')
plt.plot(range(len(feature_not_0_tmp)), mean_50, '.r', label='neurons')
plt.grid(b=True, which='both')
plt.minorticks_on()
plt.ylim([0, .92])
plt.legend()

# plt.bar(range(len(feature_not_0_tmp)), [feature_not_0_tmp, mean_50])
#
# plt.bar([range(len(feature_not_0_tmp)), range(len(feature_not_0_tmp))], [feature_not_0_tmp, mean_50])

# plt.grid(b=True, which='both')
# plt.minorticks_on()
# plt.ylim([0, .92])




# tmp1 = np.asarray(shap_contribution_list)
# for k in range(tmp1.shape[0]//2):
#     np.mean()
#

# loop through zipped models and h5 (val/test pairs load together)

# get full shap values

# get abs and mean shap values of index [1] (contributions to touch)
"""
I guess I could just go with all the features that overlap for each model
then just run the model with those features (selecting using indexing from the h5) 
show that they are similar 
then choose those and train with the full dataset (maybe even augmented)
then show that they can work with unseen data 
then show they can work with transfer learning 


one test I can do is train on different looking images one model for each session, to see if the overlap is the same 
or if there is some selection for each 
if there is some selection for each then TL wont work as well if I only select the top ~8% x neurons 
so like if there is a set that like this 10% and another set that likes 8% and half the neurons overlap but I choose the 
10% neurons then only 5% of those will be useful for predicitng making the model not TL well 


which group has a lot of clear 0's and then clear winners to choose from how should I choose features
do the plots 
separate by neuron take mean for each plot red hist
separate by feature (not total features) type take mean for each plot blue hist 
separate by total features take mean for each plot green hist 

plot sorted of these groups 

choose the ones we want, test how good the model is


"""
# df_x = pd.DataFrame(x_train, columns=final_feature_names)
# for name in df_x.columns:
#     shap.dependence_plot(name, shap_values[1], df_x, display_features=df_x)
#     asdfasdfsdfasdf

#
#
# """
# shap.summary_plot
# need to figure out how this work or just get the shap values and make predicitons and shap values again and use the
# shap package otherwise IDK if they jsut take the mean or whatever
# """
#
#
#
# np_in = '/Users/phil/Desktop/LIGHT_GBM/SHAP_VAL_num_0.npy'
# shap_vals = np.load(np_in, mmap_mode='r')
#
# _, bi_2048_set = utils.lister_it(final_feature_names, remove_string=['FD_TOTAL'], return_bool_index=True)
# feature_nums_set = feature_nums[bi_2048_set]
#
# shap_neurons = dict = {}
# for k in np.unique(feature_nums_set):
#   shap_neurons[str(k)]
#   shap_vals[feature_nums_set==k]
#
#
#
# _, bi_2048_set = utils.lister_it(final_feature_names, remove_string=['FD_TOTAL'], return_bool_index=True)
#
#
# # shap_vals.shape  # (23435, 84010)
# tmp_shape = shap_vals[:100, :]
# """$$$$%%%%$$$$$%%%%$$$$%%%%$$$$$%%%%$$$$%%%%$$$$$%%%%$$$$%%%%$$$$$%%%%$$$$%%%%$$$$$%%%%$$$$%%%%$$$$$%%%%$$$$%%%%$$$$$%%%%"""
#
#
#
# # fd_keys = utils.print_h5_keys(h5_in, 1, 0)
# # fd_keys = utils.lister_it(fd_keys, 'FD__')
# # fd_k_normal = utils.lister_it(fd_keys, remove_string='FD_TOTAL')
# # fd_k_total = utils.lister_it(fd_keys, keep_strings='FD_TOTAL')
#
#
#
#
#
# def load_SHAP_data(np_in, feature_list, max_size=100):
#     if isinstance(h5_in, list):
#         all_x = [];
#         all_y = []
#         for k in h5_in:
#             tmp_x, tmp_y = load_data(k, feature_list, split_int_n_chunks=split_int_n_chunks,
#                                      split_chunk_ind=split_chunk_ind, label_key=label_key)
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
#             all_x = image_tools.get_h5_key_and_concatenate(h5_in, k)[split_chunk_ind::split_int_n_chunks].astype(
#                 'float32')
#         else:
#             x = image_tools.get_h5_key_and_concatenate(h5_in, k)[split_chunk_ind::split_int_n_chunks].astype('float32')
#             if len(x.shape) > 1:
#                 all_x = np.hstack((all_x, x))
#             else:
#                 all_x = np.hstack((all_x, x[:, None]))
#
#     all_y = image_tools.get_h5_key_and_concatenate(h5_in, label_key)[split_chunk_ind::split_int_n_chunks].astype(
#         'float32')
#     return all_x, all_y
#
#
# def load_data(h5_in, feature_list, split_int_n_chunks=1, split_chunk_ind=0, label_key='labels'):
#     if isinstance(h5_in, list):
#         all_x = [];
#         all_y = []
#         for k in h5_in:
#             tmp_x, tmp_y = load_data(k, feature_list, split_int_n_chunks=split_int_n_chunks,
#                                      split_chunk_ind=split_chunk_ind, label_key=label_key)
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
#             all_x = image_tools.get_h5_key_and_concatenate(h5_in, k)[split_chunk_ind::split_int_n_chunks].astype(
#                 'float32')
#         else:
#             x = image_tools.get_h5_key_and_concatenate(h5_in, k)[split_chunk_ind::split_int_n_chunks].astype('float32')
#             if len(x.shape) > 1:
#                 all_x = np.hstack((all_x, x))
#             else:
#                 all_x = np.hstack((all_x, x[:, None]))
#
#     all_y = image_tools.get_h5_key_and_concatenate(h5_in, label_key)[split_chunk_ind::split_int_n_chunks].astype(
#         'float32')
#     return all_x, all_y
