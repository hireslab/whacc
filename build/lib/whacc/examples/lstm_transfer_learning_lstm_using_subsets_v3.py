# from google.colab import drive
# drive.mount('/content/gdrive')
# drive._mount('/content/gdrive')
from whacc import model_maker

from whacc.model_maker import *


import h5py

from tensorflow.keras.layers import Dense, LSTM, Flatten, TimeDistributed, Conv2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras import applications


from tqdm.notebook import tqdm
from tqdm import tqdm
import pickle
from whacc import utils, image_tools, transfer_learning, analysis
import matplotlib.pyplot as plt

from sklearn.utils import class_weight
import os
import copy
import numpy as np

from pathlib import Path
import shutil
import zipfile
from datetime import datetime
import pytz
import json
from math import isclose, sqrt
from IPython import display


# !pip install twilio
# from twilio.rest import Client
def send_text(mess):#keep
    print('oops texting failed but the code goes on')


def save_obj(obj, name):#keep
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):#keep
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def inds_sorted_data(a, key_name, max_or_min):#keep
    """
  a is the data form all_data looped through 
  """
    log_ind = np.where(key_name == a['logs_names'])[0][0]
    val_list = a['all_logs'][:, log_ind]
    if max_or_min == 'max':
        max_arg_sort = np.flip(np.argsort(val_list)) + 1
    elif max_or_min == 'min':
        max_arg_sort = np.argsort(val_list) + 1
    else:
        raise ValueError("""max_or_min must be a string set to 'max' or 'min'""")
    return max_arg_sort


def sorted_loadable_epochs(a, key_name, max_or_min):#keep
    """
  a is the data form all_data looped through
  """
    arg_sort_inds = inds_sorted_data(data, key_name, max_or_min)
    arg_sort_inds[np.argmax(arg_sort_inds)] = -1
    saved_epoch_numbers = np.asarray(list(data['info']['epoch_dict'].keys()))
    sorted_loadable_epochs_out = []
    for k in arg_sort_inds:
        if k in saved_epoch_numbers:
            sorted_loadable_epochs_out.append(k)
    return sorted_loadable_epochs_out


def reload_info_dict(base_dir):
    info_dict_path = utils.get_files(base_dir, '*info_dict.json')[0]
    with open(info_dict_path, 'r') as f:
        D = json.load(f)
    # replace the directory to the main directory
    x = 'model_testing'
    old_base_dir = ''
    for k in D.keys():
        try:
            if x in D[k]:
                D[k] = base_dir.split(x)[0] + x + D[k].split(x)[-1]
                old_base_dir = D[k].split(x)[0]
        except:
            pass
    D['old_base_dir'] = old_base_dir
    D['checkpoints'] = utils.get_files(D['model_save_dir_checkpoints'], '*hdf5')
    D['label_num_names'] = D['label_key_name'].split('- (')[-1][:-1].split(', ')  # here use this to label the y limits
    D['label_nums'] = eval(D['label_key_name'].split(']')[0] + ']')

    checkpoint_epochs = []
    for k in D['checkpoints'][:-1]:
        checkpoint_epochs.append(int(k.split('_cp.hdf5')[0][-4:]))
    D['checkpoint_epochs'] = checkpoint_epochs

    epoch_dict = dict()
    for i, k in enumerate(checkpoint_epochs):
        epoch_dict[k] = D['checkpoints'][i]
    epoch_dict[-1] = D['checkpoints'][-1]
    D['epoch_dict'] = epoch_dict
    return D


def re_build_model(model_name_str, class_numbers, base_learning_rate=0.00001, dropout_val=None, IMG_SIZE=96,
                   labels=None):
    num_classes = len(class_numbers)
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    model_function = eval('applications.' + model_name_str)
    base_model = model_function(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
    base_model.trainable = True
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)  # global spatial average pooling layer
    x = Dense(2048, activation='relu')(x)  # fully-connected layer
    if dropout_val is not None:
        x = Dropout(dropout_val)(x)
    ###### i need to name the layers
    if num_classes == 2:
        predictions = Dense(1, activation='sigmoid')(x)  # fully connected output/classification layer
    else:
        predictions = Dense(num_classes, activation='softmax')(x)  # fully connected output/classification layer
    model = Model(inputs=base_model.input, outputs=predictions)

    if num_classes == 2:

        optimizer = keras.optimizers.RMSprop(learning_rate=base_learning_rate)
        loss = keras.losses.BinaryCrossentropy()
        metrics = [keras.metrics.BinaryAccuracy(name="bool_acc", threshold=0.5),
                   keras.metrics.AUC(name='auc')]
    else:
        optimizer = keras.optimizers.Adam(learning_rate=base_learning_rate)
        loss = keras.losses.SparseCategoricalCrossentropy()
        metrics = [keras.metrics.SparseCategoricalAccuracy(name='acc')]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if labels is not None:
        rebalance = class_weight.compute_class_weight('balanced', classes=class_numbers, y=labels.flatten())
        class_weights = {i: rebalance[i] for i in class_numbers}
        wrap_vars_list = ['class_numbers',
                          'num_classes',
                          'base_learning_rate',
                          'model_name_str',
                          'IMG_SIZE',
                          'dropout_val']
        return model, class_weights
    else:
        return model


def load_model_data(all_models_directory):
    data_files = utils.get_files(all_models_directory, '*model_eval_each_epoch.json')
    to_split = '/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/'
    all_data = []
    for i, k in enumerate(data_files):
        if '/small_h5s/' not in k:
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
        all_data[i]['info'] = reload_info_dict(k['dir'])
    return all_data




def build_model(info_dict, labels, model_name_str, base_learning_rate=0.00001, dropout_val=None, class_numbers=None,
                IMG_SIZE=96):
    if class_numbers is None:
        class_numbers = np.unique(labels)
    num_classes = len(class_numbers)
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    model_function = eval('applications.' + model_name_str)
    base_model = model_function(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
    base_model.trainable = True
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)  # global spatial average pooling layer
    x = Dense(2048, activation='relu')(x)  # fully-connected layer
    if dropout_val is not None:
        x = Dropout(dropout_val)(x)
    ###### i need to name the layers
    if num_classes == 2:
        predictions = Dense(1, activation='sigmoid')(x)  # fully connected output/classification layer
    else:
        predictions = Dense(num_classes, activation='softmax')(x)  # fully connected output/classification layer
    model = Model(inputs=base_model.input, outputs=predictions)

    if num_classes == 2:

        optimizer = keras.optimizers.RMSprop(learning_rate=base_learning_rate)
        loss = keras.losses.BinaryCrossentropy()
        metrics = [keras.metrics.BinaryAccuracy(name="bool_acc", threshold=0.5),
                   keras.metrics.AUC(name='auc')]
    else:
        optimizer = keras.optimizers.Adam(learning_rate=base_learning_rate)
        loss = keras.losses.SparseCategoricalCrossentropy()
        metrics = [keras.metrics.SparseCategoricalAccuracy(name='acc')]
    for k in model.layers:
        k.trainable = False
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    rebalance = class_weight.compute_class_weight('balanced', classes=class_numbers, y=labels.flatten())
    class_weights = {i: rebalance[i] for i in class_numbers}
    wrap_vars_list = ['class_numbers',
                      'num_classes',
                      'base_learning_rate',
                      'model_name_str',
                      'IMG_SIZE',
                      'dropout_val']
    for k in wrap_vars_list:
        info_dict[k] = eval(k)
    info_dict['class_weights'] = str(class_weights)
    return model, class_weights, info_dict


# def label_naming_shorthand_dict(name_key=None):
#     label_naming_shorthand_dict = {
#         '[0, 1, 2, 3, 4, 5]- (no touch, touch, onset, one after onset, offset, one after offset)': 'on-off_set_and_one_after',
#         '[0, 1, 2, 3]- (no touch, touch, onset, offset': 'on-off_set',
#         '[0, 1, 2]- (no event, onset, offset)': 'only_on-off_set',
#         '[0, 1]- (no touch, touch)': 'regular',
#         '[0, 1]- (not offset, offset)': 'only_offset',
#         '[0, 1]- (not onset, onset)': 'only_onset'}
#     if name_key is None:
#         return label_naming_shorthand_dict
#     else:
#         return label_naming_shorthand_dict[name_key]


# def info_dict_wrapper(info_dict, local_dict):
#     for k in local_dict.keys():
#         info_dict[k] = local_dict[k]
#     return info_dict


# def make_initial_folder(all_models_directory, unique_h5_train_val_dir):
#     single_frame_dir = all_models_directory + os.sep + unique_h5_train_val_dir + os.sep + 'data' + os.sep + 'single_frame'
#     Path(single_frame_dir).mkdir(parents=True, exist_ok=True)
#     return locals()
#     # return single_frame_dir


def get_automated_model_info_TL(BASE_H5, image_source_h5_directory_ending, test_data_dir, data_string_key="data"):#keep
    tz = pytz.timezone('America/Los_Angeles')
    loc_dt = pytz.utc.localize(datetime.utcnow())
    LA_TIME = loc_dt.astimezone(tz)
    todays_version = LA_TIME.strftime("%Y_%m_%d_%H_%M_%S")
    del tz
    del loc_dt
    del LA_TIME
    a = os.sep
    base_data_dir = BASE_H5 + a + data_string_key + a
    base_dir_all_h5s = BASE_H5 + a + data_string_key + a + 'single_frame' + a
    data_dir = base_data_dir + image_source_h5_directory_ending
    print('\nFOR IMAGES, 0 is train set, 1 is val set')
    print(data_dir)
    image_h5_list = utils.get_h5s(data_dir)
    # pdb.set_trace()
    h5_train = image_h5_list[0]
    h5_val = image_h5_list[1]
    # labels_dir = base_data_dir + a + "ALT_LABELS" + a
    # print('\nFOR LABELS,0 is train set, 1 is val set')
    # label_h5_list = utils.get_h5s(labels_dir)
    # print('\nSelect from the following label structures...')
    # print(labels_dir)
    # label_key_name_list = utils.print_h5_keys(label_h5_list[0], return_list=True)
    # h5_test_labels = utils.get_h5s(test_data_dir + a + "ALT_LABELS" + a, print_h5_list=False)[0]
    # h5_test = utils.get_h5s(test_data_dir + a + image_source_h5_directory_ending + a, print_h5_list=False)[0]
    return locals()


def copy_over_new_labels(label_key_name, image_h5_list, label_h5_list):
    label_key_shorthand = label_naming_shorthand_dict(label_key_name)
    for img_src, lab_src in zip(image_h5_list, label_h5_list):
        utils.copy_h5_key_to_another_h5(lab_src, img_src, label_key_name, 'labels')
    return locals()


# get list of pre trained models to choose from
def get_keras_model_names():
    names_, types_ = utils.get_class_info(applications, return_name_and_type=True)
    model_names = np.asarray(names_)['function' == np.asarray(types_)]
    utils.print_list_with_inds(model_names)
    return model_names


# def make_model_save_directory(info_dict, make_folder=True):
#     naming_list = ['model_name_str', 'image_source_h5_directory_ending', 'label_key_shorthand', 'todays_version']
#     model_save_dir: str = copy.deepcopy(info_dict['BASE_H5'])
#     for k in naming_list:
#         model_save_dir += os.sep + info_dict[k] + os.sep
#     info_dict['model_save_dir'] = model_save_dir
#     info_dict['model_save_dir_checkpoints'] = model_save_dir + os.sep + 'checkpoints'
#     if make_folder:
#         Path(info_dict['model_save_dir_checkpoints']).mkdir(parents=True, exist_ok=True)
#     return model_save_dir


def basic_callbacks(save_checkpoint_filepath, monitor='val_loss', patience=10,
                    save_best_only=False, save_weights_only=True, save_freq="epoch", period=1):
    callbacks = []
    callbacks.append(keras.callbacks.EarlyStopping(monitor=monitor, patience=patience))
    add_path_name = "{loss:.8f}_{epoch:04d}_cp.hdf5"
    add_path_name = "{epoch:04d}_cp.hdf5"
    callbacks.append(keras.callbacks.ModelCheckpoint(
        save_checkpoint_filepath + os.sep + add_path_name,
        monitor=monitor,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        save_freq=save_freq,
        period=period))
    return callbacks


def unzip_and_place_h5s(bd, do_delete_zips=False):
    bd2 = ''.join(bd.split('gdrive/My Drive'))
    shutil.copytree(bd, bd2)
    a = utils.get_files(bd2, '*.zip')
    for k in a:
        with zipfile.ZipFile(k, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(k))
        if do_delete_zips:
            os.remove(k)


def change_to_local_dir(bd):
    bd2 = ''.join(bd.split('gdrive/My Drive'))
    return bd2


def change_to_gdrive_dir(bd):
    bd2 = '/content/gdrive/My Drive/'.join(bd.split('/content/'))
    return bd2


def pack_training_info(save_and_plot_history_var):
    training_info = dict()
    for k in ['L', 'all_logs', 'all_colors', 'logs_names', 'markers', 'matching_inds']:
        training_info[k] = eval('save_and_plot_history_var.' + k)
    for k in training_info.keys():
        if 'numpy.ndarray' in str(type(training_info[k])):
            training_info[k] = training_info[k].tolist()
    return training_info


def save_info_dict(info_dict_tmp):
    for k in info_dict_tmp.keys():
        if 'numpy.ndarray' in str(type(info_dict_tmp[k])):
            info_dict_tmp[k] = info_dict_tmp[k].tolist()
    with open(info_dict_tmp['model_save_dir'] + 'info_dict' + '.json', 'w') as f:
        json.dump(info_dict_tmp, f)


def foo_save_and_plot(training_info, save_and_plot_history_1, save_loc):
    text_str = transfer_learning.make_text_for_fig(training_info)
    training_info = transfer_learning.pack_training_info(training_info, save_and_plot_history_1)
    # save_num = foo_get_next_save_num(save_loc)

    # make figure and save
    transfer_learning.plot_train_hist(training_info, [1], [.9, 1], text_str)
    plt.savefig(save_loc + 'mod_test_fig_ACC' + '.png', bbox_inches="tight")
    transfer_learning.plot_train_hist(training_info, [0], [0, 0.25], text_str)
    plt.savefig(save_loc + 'mod_test_fig_LOSS' + '.png', bbox_inches="tight")

    # save training info
    with open(save_loc + 'model_eval_each_epoch' + '.json', 'w') as f:
        json.dump(training_info, f)


# In[17]:


# def foo_run_all():
#     # copy and unzip for colab
#     global all_models_directory, test_data_dir, info_dict, transfer_learning
#
#     local_dict = make_initial_folder(all_models_directory,
#                                      unique_h5_train_val_dir)  # make data single frame directory and get that directory
#     info_dict = info_dict_wrapper(info_dict, local_dict)  # wrap output into a dict
#
#     BASE_H5 = info_dict['all_models_directory'] + os.sep + info_dict[
#         'unique_h5_train_val_dir']  # directory for all the data for a certain type og images(lag or regular etc)
#
#     local_dict = get_automated_model_info(BASE_H5, image_source_h5_directory_ending,
#                                           test_data_dir)  # basic data like directories and train and val set (automated form the directory)
#     info_dict = info_dict_wrapper(info_dict, local_dict)  # wrap output into a dict
#
#     # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
#     local_dict = copy_over_new_labels(label_key_name, info_dict['image_h5_list'] + [info_dict['h5_test']],
#                                       info_dict['label_h5_list'] + [info_dict[
#                                                                         'h5_test_labels']])  # copy specific labels to the H5 of interest (this is all after the conversion from single frame)
#     info_dict = info_dict_wrapper(info_dict, local_dict)  # wrap output into a dict
#
#     # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$      make model     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
#     # get all the labels
#     train_gen = image_tools.ImageBatchGenerator(batch_size, [info_dict['h5_train']])
#     val_gen = image_tools.ImageBatchGenerator(batch_size, [info_dict['h5_val']])
#     test_gen = image_tools.ImageBatchGenerator(batch_size, [info_dict['h5_test']])
#
#     train_labels = image_tools.get_h5_key_and_concatenate([info_dict['h5_train']])
#     val_labels = image_tools.get_h5_key_and_concatenate([info_dict['h5_val']])
#     labels = np.concatenate((train_labels, val_labels))
#
#     model, class_weights, info_dict = build_model(info_dict, labels, model_name_str,
#                                                   base_learning_rate=base_learning_rate,
#                                                   dropout_val=dropout_val)
#
#     #   info_dict['BASE_H5'] = change_to_gdrive_dir(info_dict['BASE_H5'])
#     model_save_dir = make_model_save_directory(
#         info_dict)  # make a unique folder using standard folder struct ending in a unique date/time folder
#     # ###change directory to gdrive in case it crashes
#     # dir_2_change = ['model_save_dir_checkpoints', 'model_save_dir']
#     # for k in dir_2_change:
#     #   info_dict[k] = change_to_gdrive_dir(info_dict[k])
#
#     model, class_weights, info_dict = build_model(info_dict,
#                                                   labels,
#                                                   model_name_str,
#                                                   base_learning_rate=base_learning_rate,
#                                                   dropout_val=dropout_val)
#
#     callbacks = basic_callbacks(info_dict['model_save_dir_checkpoints'], monitor=monitor, patience=patience,
#                                 save_best_only=save_best_only, save_weights_only=True,
#                                 save_freq=save_freq, period=period)
#
#     plot_callback = transfer_learning.save_and_plot_history(test_gen, key_name='labels', plot_metric_inds=[0])
#     callbacks.append(plot_callback)
#
#     ###FITTTTTTT
#     history = model.fit(train_gen,
#                         epochs=epochs,
#                         validation_data=val_gen,
#                         callbacks=callbacks,
#                         class_weight=class_weights)
#     # save finals checkpoint after model finishes
#     model.save_weights(info_dict['model_save_dir_checkpoints'] + os.sep + 'final_epoch_cp.hdf5')
#
#     training_info = pack_training_info(plot_callback)
#
#     xx = '/content/colab_data2/model_testing/all_data/all_models/small_h5s/InceptionV3/3lag/on-off_set_and_one_after/'
#     utils.get_files(xx, '*_cp.hdf5')
#
#     transfer_learning.foo_save_and_plot
#
#     foo_save_and_plot(training_info, plot_callback, info_dict['model_save_dir'])
#
#     save_info_dict(info_dict)




# def smooth(y, box_pts):#$%
#     box = np.ones(box_pts) / box_pts
#     y_smooth = np.convolve(y, box, mode='same')
#     return y_smooth


# def search_sequence_numpy(arr, seq): #$%
#     # Store sizes of input array and sequence
#     Na, Nseq = arr.size, seq.size
#
#     # Range of sequence
#     r_seq = np.arange(Nseq)
#
#     # Create a 2D array of sliding indices across the entire length of input array.
#     # Match up with the input sequence & get the matching starting indices.
#     M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)
#     # Get the range of those indices as final output
#     if M.any() > 0:
#         # return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
#         return np.where(M)[0]
#     else:
#         return []  # No match found


# def search_sequence_numpy(arr, seq, return_type='indices'):#$%
#     Na, Nseq = arr.size, seq.size
#     r_seq = np.arange(Nseq)
#     M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)
#
#     if return_type == 'indices':
#         return np.where(M)[0]
#     elif return_type == 'bool':
#         return M


# def get_pairs_that_extend_out(num1, num2, array_in):#$%
#     a = np.where(array_in == num1)[0]
#     b = np.where(array_in == num2)[0]
#     c = sorted(np.hstack((a, b)))
#     c, _ = np.asarray(utils.group_consecutives(c))
#     bool_inds = np.zeros((3, len(c)), dtype=bool)
#     for i, k in enumerate(c):
#         N2 = array_in[k]
#         if np.all(N2 == num1):
#             bool_inds[0, i] = True
#         elif np.all(N2 == num2):
#             bool_inds[1, i] = True
#         else:
#             bool_inds[2, i] = True
#     return c, bool_inds


# def replotYlabels_everyX(D, every_x=10, len_plot=0):#$%
#     x = np.floor((len_plot - 1) / every_x)
#     for kk in range(int(x)):
#         kk += 1
#         for k, k2 in zip(D['label_nums'], D['label_num_names']):
#             plt.text(x=kk * every_x + .1, y=k, s=k2)


# class pole_plot():#$%
#     def __init__(self, img_h5_file, pred_val=None, true_val=None, threshold=0.5,
#                  len_plot=10, current_frame=0, figsize=[10, 5], label_nums=None, label_num_names=None, shift_by=0):
#         """
#         Examples
#         ________
#         a = analysis.pole_plot(
#             '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/master_train_test1.h5',
#             pred_val = [0,0,0,0,0,0,0,.2,.4,.5,.6,.7,.8,.8,.6,.4,.2,.1,0,0,0,0],
#             true_val = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
#             len_plot = 10)
#
#         a.plot_it()
#         """
#         if true_val is None:
#             if 'labels' in utils.print_h5_keys(img_h5_file, return_list=True, do_print=False):
#                 true_val = image_tools.get_h5_key_and_concatenate([img_h5_file])
#         tmp_loc = locals()
#         self.error_group_inds = None
#         if 'true_val' in tmp_loc and 'pred_val' in tmp_loc:
#             error_group_inds = np.where(true_val - pred_val != 0)[0]
#             error_group_inds, _ = utils.group_consecutives(error_group_inds)
#             self.error_group_inds = error_group_inds
#
#         self.isnotebook = utils.isnotebook()
#         self.img_h5_file = img_h5_file
#         self.pred_val = np.asarray(pred_val)
#         self.true_val = np.asarray(true_val)
#         self.threshold = threshold
#         self.len_plot = len_plot
#         self.current_frame = current_frame
#         self.figsize = figsize
#         self.fig_created = False
#         tmp3 = np.unique(np.concatenate((self.pred_val, self.true_val)))
#         self.range_labels = np.arange(np.nanmin(tmp3), np.nanmax(tmp3) + 1)
#         self.ylims = [np.nanmin(tmp3) - .5, np.nanmax(tmp3) + .5]
#
#         self.label_nums = label_nums
#         self.label_num_names = label_num_names
#         self.shift_by = shift_by
#         self.xlims = [-.5, self.len_plot - .5]
#         try:
#             self.pred_val_bool = (1 * (self.pred_val > threshold)).flatten()
#         except:
#             self.pred_val_bool = np.asarray(None)
#
#     def plot_it(self):
#
#         if self.fig_created is False or self.isnotebook:  # we need to create a new fig every time if we are in colab or jupyter
#             self.fig, self.axs = plt.subplots(2, figsize=self.figsize, gridspec_kw={'height_ratios': [1, 1]})
#             self.fig_created = True
#             # plt.subplots_adjust(hspace = .001)
#
#         ax1 = self.axs[1]
#         box = ax1.get_position()
#         box.y0 = box.y0 + self.shift_by
#         box.y1 = box.y1 + self.shift_by
#         ax1.set_position(box)
#
#         self.axs[0].clear()
#         self.axs[1].clear()
#         self.fig.suptitle('Touch prediction')
#         s1 = self.current_frame
#         s2 = self.current_frame + self.len_plot
#         self.xticks = np.arange(0, self.len_plot)
#         # plt.axis('off')
#         with h5py.File(self.img_h5_file, 'r') as h:
#             self.current_imgs = image_tools.img_unstacker(h['images'][s1:s2], s2 - s1)
#             # plt.imshow(self.current_imgs)
#             self.axs[0].imshow(self.current_imgs)
#             self.axs[0].axis('off')
#
#         leg = []
#         # axs[1].plot([None])
#         if len(self.pred_val.shape) != 0:
#             plt.plot(self.pred_val[s1:s2].flatten(), color='black', marker='*', linestyle='None')
#             leg.append('pred')
#         # if len(self.pred_val_bool.shape) != 0:
#         #     plt.plot(self.pred_val_bool[s1:s2].flatten(), '.g', markersize=10)
#         #     leg.append('bool_pred')
#         if len(self.true_val.shape) != 0:
#             tmp1 = self.true_val[s1:s2].flatten()
#             plt.scatter(range(len(tmp1)), tmp1, s=80, facecolors='none', edgecolors='r')
#             leg.append('actual')
#         if leg:
#             plt.legend(leg, bbox_to_anchor=(1.04, 1), borderaxespad=0)
#             plt.ylim(self.ylims)
#             plt.xlim(self.xlims)
#             _ = plt.xticks(ticks=self.xticks)
#         if self.label_nums is not None and self.label_num_names is not None:
#             plt.yticks(ticks=self.label_nums, labels=self.label_num_names)
#         plt.grid(axis='y')
#
#     def next(self):
#         self.current_frame = self.current_frame + self.len_plot
#         self.plot_it()
#
#     def move(self, move_val):
#         self.current_frame = self.current_frame + move_val
#         self.plot_it()


# def get_class_info(c, sort_by_type=True, include_underscore_vars=False, return_name_and_type=False, end_prev_len=40):#$%
#     names = []
#     type_to_print = []
#     for k in dir(c):
#         if include_underscore_vars is False and k[0] != '_':
#             tmp1 = str(type(eval('c.' + k)))
#             type_to_print.append(tmp1.split("""'""")[-2])
#             names.append(k)
#         elif include_underscore_vars:
#             tmp1 = str(type(eval('c.' + k)))
#             type_to_print.append(tmp1.split("""'""")[-2])
#             names.append(k)
#     len_space = ' ' * max(len(k) for k in names)
#     len_space_type = ' ' * max(len(k) for k in type_to_print)
#     if sort_by_type:
#         ind_array = np.argsort(type_to_print)
#     else:
#         ind_array = np.argsort(names)
#
#     for i in ind_array:
#         k1 = names[i]
#         k2 = type_to_print[i]
#         # k3 = str(c[names[i]])
#         k3 = str(eval('c.' + names[i]))
#         # [-40:]
#         k1 = (k1 + len_space)[:len(len_space)]
#         k2 = (k2 + len_space_type)[:len(len_space_type)]
#         if len(k3) > end_prev_len:
#             k3 = '...' + k3[-end_prev_len:]
#         else:
#             k3 = '> ' + k3[-end_prev_len:]
#
#         print(k1 + ' type->   ' + k2 + '  ' + k3)
#     if return_name_and_type:
#         return names, type_to_print


# def get_dict_info(c, sort_by_type=True, include_underscore_vars=False, return_name_and_type=False, end_prev_len=40):#$%
#     names = []
#     type_to_print = []
#     for k in c.keys():
#         if include_underscore_vars is False and k[0] != '_':
#             tmp1 = str(type(c[k]))
#             type_to_print.append(tmp1.split("""'""")[-2])
#             names.append(k)
#         elif include_underscore_vars:
#             tmp1 = str(type(c[k]))
#             type_to_print.append(tmp1.split("""'""")[-2])
#             names.append(k)
#     len_space = ' ' * max(len(k) for k in names)
#     len_space_type = ' ' * max(len(k) for k in type_to_print)
#     if sort_by_type:
#         ind_array = np.argsort(type_to_print)
#     else:
#         ind_array = np.argsort(names)
#
#     for i in ind_array:
#         k1 = names[i]
#         k2 = type_to_print[i]
#         k3 = str(c[names[i]])
#
#         k1 = (k1 + len_space)[:len(len_space)]
#         k2 = (k2 + len_space_type)[:len(len_space_type)]
#         if len(k3) > end_prev_len:
#             k3 = '...' + k3[-end_prev_len:]
#         else:
#             k3 = '> ' + k3[-end_prev_len:]
#
#         print(k1 + ' type->   ' + k2 + '  ' + k3)
#     if return_name_and_type:
#         return names, type_to_print


"""# download the retrain augmented and OG files to local """


def foo_rename(instr):#keep
    if isinstance(instr, list):
        for i, k in enumerate(instr):
            instr[i] = foo_rename(k)
        return instr
    else:
        return '/Volumes/GoogleDrive-114825029448473821206/My Drive' + instr.split('My Drive')[-1]


"""# download the retrain augmented and OG files to local """
train_and_val_dir = '/Users/phil/Desktop/content/ALL_RETRAIN_H5_data____temp'
srs_TandV_h5s, dst_TandV_h5s = utils.copy_file_filter(
    foo_rename('/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/small_h5s/data'),
    train_and_val_dir,
    keep_strings='/3lag/',
    remove_string='.zip',
    overwrite=True,
    just_print_what_will_be_copied=False, return_list_of_files=True)

h5_imgs, h5_labels, inds_of_files = utils.copy_alt_labels_based_on_directory(srs_TandV_h5s)
for i in inds_of_files:
    utils.copy_over_all_non_image_keys(h5_labels[i], dst_TandV_h5s[i])

base_test_dir = '/Users/phil/Desktop/content/DATA_FULL____temp'
srs_test_h5s, dst_test_h5s = utils.copy_file_filter(
    foo_rename('/content/gdrive/My Drive/colab_data2/model_testing/all_data/test_data/small_h5s'),
    base_test_dir,
    keep_strings='/3lag/',
    remove_string='.zip',
    overwrite=True,
    just_print_what_will_be_copied=False, return_list_of_files=True)

h5_imgs, h5_labels, inds_of_files = utils.copy_alt_labels_based_on_directory(srs_test_h5s)
for i in inds_of_files:
    utils.copy_over_all_non_image_keys(h5_labels[i], dst_test_h5s[i])

"""#trasnform h5s to LSTM H5s"""

base_dir = '/Users/phil/Desktop/content/'
for h5 in utils.lister_it(utils.get_h5s(base_dir), keep_strings='____temp'):  # only go through the non converted h5s
    # utils.print_h5_keys(h5)
    h5_out = ''.join(h5.split('____temp'))
    print(h5, h5_out)
    image_tools.convert_h5_to_LSTM_h5(h5, h5_out)
    utils.copy_over_all_non_image_keys(h5, h5_out)

train_and_val_dir = ''.join(train_and_val_dir.split('____temp')) + os.sep
base_test_dir = ''.join(base_test_dir.split('____temp')) + os.sep
"""#copy the alt labels to the h5 files """

"""#working TL"""


# bd2 = "/content/gdrive/My Drive/colab_data2/"
# all_models_directory = bd2+"/model_testing/all_data/all_models/"
# all_data = load_model_data(all_models_directory)


# save_obj(all_data, '/content/all_data2')
all_data = load_obj('/Users/phil/Downloads/all_data')

"""# code for model TMP TMP """

class save_and_plot_history(keras.callbacks.Callback):
    def __init__(self, my_test_batch_generator, key_name='labels', plot_metric_inds=[0]):
        self.all_y = image_tools.get_h5_key_and_concatenate(my_test_batch_generator.H5_file_list, key_name=key_name)
        self.my_test_batch_generator = my_test_batch_generator
        self.all_logs = np.asarray([])
        self.markers = np.tile(['-', '--', ':'], len(plot_metric_inds))
        c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.all_colors = c
        self.color = np.repeat([c[k] for k in plot_metric_inds], 3)
        self.plot_metric_inds = plot_metric_inds

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or 'logs_names' not in dir(self):

            self.L = len(self.model.metrics_names)
            L = len(self.model.metrics_names)
            self.matching_inds = np.reshape(np.linspace(0, L * 3 - 1, L * 3), [3, -1]).astype('int')
            self.plot_metric_inds = self.plot_metric_inds + [k + L for k in self.plot_metric_inds] + [k + L * 2 for k in
                                                                                                      self.plot_metric_inds]
            self.plot_metric_inds = np.swapaxes(np.reshape(self.plot_metric_inds, [3, -1]), 1, 0).flatten()
            self.fig = plt.figure(self.plot_metric_inds[0], figsize=[15, 10])
            self.logs_names = self.model.metrics_names + [k + '_val' for k in self.model.metrics_names] + [k + '_test'
                                                                                                           for k in
                                                                                                           self.model.metrics_names]

            self.logs_names_selected = []
            plt.ylim([.95, 1])
            for k in self.plot_metric_inds:
                self.logs_names_selected.append(self.logs_names[k])
        if logs is not None:
            log_list = []
            _ = [log_list.append(logs[k]) for k in logs.keys()]
            logs_test = self.model.evaluate(self.my_test_batch_generator)
            log_list = log_list + logs_test
            if len(self.all_logs) != 0:
                self.all_logs = np.vstack((self.all_logs, np.asarray(log_list)))
            else:
                self.all_logs = np.concatenate((self.all_logs, np.asarray(log_list)), axis=0)
            # if epoch > 0:
            if len(self.all_logs.shape) > 1:
                display.clear_output(wait=True)
                for kmark, kcol, k in zip(self.markers, self.color, self.plot_metric_inds):
                    plt.plot(self.all_logs[:, k], color=kcol, linestyle=kmark)

                plt.legend(self.logs_names_selected)
                plt.grid(True)
                display.display(plt.gcf())


def save_info_dict(info_dict_tmp):  # added os.sep to saveing json file
    for k in info_dict_tmp.keys():
        if 'numpy.ndarray' in str(type(info_dict_tmp[k])):
            info_dict_tmp[k] = info_dict_tmp[k].tolist()
    with open(info_dict_tmp['model_save_dir'] + os.sep + 'info_dict' + '.json', 'w') as f:
        json.dump(info_dict_tmp, f)


# def foo_training_validation_type_controller(ind_number, target_path="/content/ALL_RETRAIN_H5_data"):
#     tmp1 = utils.get_files(target_path, '*.hh55')
#     for k in tmp1:
#         os.rename(k, k.split('.hh55')[0] + '.h5')
#
#     b = utils.get_h5s(target_path, 0)
#     if ind_number == 0:
#         # KEEP train_AUG and val_REG
#         remove_list = list(utils.lister_it(b, remove_string=['_AUG.h5', 'val']))
#         remove_list = remove_list + list(utils.lister_it(b, keep_strings='AUG', remove_string=['train']))
#         for k in remove_list:
#             print(k)
#             os.rename(k, k.split('.h5')[0] + '.hh55')
#     elif ind_number == 1:
#         # KEEP train_AUG and val_AUG
#         remove_list = list(utils.lister_it(b, remove_string=['_AUG.h5']))
#         print(remove_list)
#         for k in remove_list:
#             print(k)
#             os.rename(k, k.split('.h5')[0] + '.hh55')
#     elif ind_number == 2:
#         # KEEP train_REG and val_REG
#         remove_list = list(utils.lister_it(b, keep_strings=['_AUG.h5']))
#         print(remove_list)
#         for k in remove_list:
#             print(k)
#             os.rename(k, k.split('.h5')[0] + '.hh55')


def re_build_model_TL(model_name_str, class_numbers, base_learning_rate=0.00001,
                      dropout_val=None, IMG_SIZE=96, labels=None, reload_weights_file=None, num_layers_unfreeze=0):#keep
    num_classes = len(class_numbers)
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    model_function = eval('applications.' + model_name_str)
    base_model = model_function(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
    num_layers_in_base_model = len(base_model.layers)
    base_model_layer_names = [k.name for k in base_model.layers]
    base_model.trainable = False  ##&%&$&$&%&%&$&%&$&$&%&%&$&&$&&%&%&$&&$&$&&&$##&%&$&$&%&%&$&%&$&$&%&%&$&&$&&%&%&$&&$&$&&&$##&%&$&$&%&%&$&%&$&$&%&%&$&&$&&%&%&$&&$&$&&&$##&%&$&$&%&%&$&%&$&$&%&%&$&&$&&%&%&$&&$&$&&&$##&%&$&$&%&%&$&%&$&$&%&%&$&&$&&%&%&$&&$&$&&&$
    len_base_model = len(base_model.layers)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)  # global spatial average pooling layer
    x = Dense(2048, activation='relu')(x)  # fully-connected layer
    if dropout_val is not None:
        x = Dropout(dropout_val)(x)
    ###### i need to name the layers
    if num_classes == 2:
        predictions = Dense(1, activation='sigmoid')(x)  # fully connected output/classification layer
    else:
        predictions = Dense(num_classes, activation='softmax')(x)  # fully connected output/classification layer
    model = Model(inputs=base_model.input, outputs=predictions)

    if num_classes == 2:

        optimizer = keras.optimizers.RMSprop(learning_rate=base_learning_rate)
        loss = keras.losses.BinaryCrossentropy()
        metrics = [keras.metrics.BinaryAccuracy(name="bool_acc", threshold=0.5),
                   keras.metrics.AUC(name='auc')]
    else:
        optimizer = keras.optimizers.Adam(learning_rate=base_learning_rate)
        loss = keras.losses.SparseCategoricalCrossentropy()
        metrics = [keras.metrics.SparseCategoricalAccuracy(name='acc')]
    if reload_weights_file is not None:
        model.load_weights(reload_weights_file)  # load model weights

    # for i, k in enumerate(model.layers):
    #   if i >= len_base_model-5:
    #     k.trainable = True
    #   else:
    #     k.trainable = False
    relu_layers = []
    for i, k in enumerate(model.layers):
        if 'relu' in k.name.lower():
            relu_layers.append(i)
    relu_layers.append(9999999)
    # relu_layers = np.flip(np.asarray(relu_layers)+1)
    relu_layers = np.flip(np.asarray(relu_layers))

    # num_layers_unfreeze =1          #0 means freeze entire model 1 means base model forzen 2 one more (group)laye runfrozen etc
    for i, k in enumerate(model.layers):
        if i >= relu_layers[
            num_layers_unfreeze] and 'batchnorm' not in k.name.lower():  # LK:J:LKJD:LKJD:LKDJ:LDKJ:DLKJD:LKJD:LKJD:LKJD:LKDJ:LKDJLK:DJ
            k.trainable = True
        else:
            k.trainable = False

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.num_layers_in_base_model = num_layers_in_base_model
    model.base_model_layer_names = base_model_layer_names
    if labels is not None:
        rebalance = class_weight.compute_class_weight('balanced', classes=class_numbers, y=labels.flatten())
        class_weights = {i: rebalance[i] for i in class_numbers}
        wrap_vars_list = ['class_numbers',
                          'num_classes',
                          'base_learning_rate',
                          'model_name_str',
                          'IMG_SIZE',
                          'dropout_val']
        return model, class_weights
    else:
        return model


for TMP_basename in next(os.walk(train_and_val_dir))[1]:
    if 'ipynb_checkpoints' not in TMP_basename:
        print(TMP_basename)
# righthere2
"""##train    asdfasfasdfasfasdfasdfasdf"""


def foo_rename2(instr):#keep
    if isinstance(instr, str) and '/My Drive/' in instr:
        return '/Volumes/GoogleDrive-114825029448473821206/My Drive' + instr.split('My Drive')[-1]
    else:
        return instr


def make_D_local(D):#keep
    for key in D:
        if isinstance(D[key], str):
            D[key] = foo_rename2(D[key])
    for key in D['epoch_dict']:
        if isinstance(D['epoch_dict'][key], str):
            D['epoch_dict'][key] = foo_rename2(D['epoch_dict'][key])
    return D


def add_lstm_to_model(model_in, num_layers_in_base_model, base_learning_rate=10 ** -5, lstm_len=7):#keep
    # model_in.base_model_layer_names
    base_model = Model(model_in.input, model_in.layers[num_layers_in_base_model - 1].output)
    model_out = Sequential()
    model_out.add(TimeDistributed(base_model, input_shape=(lstm_len, 96, 96, 3)))
    model_out.add(TimeDistributed(Flatten()))
    model_out.add(LSTM(256, activation='relu', return_sequences=False))
    model_out.add(Dense(64, activation='relu'))
    # model_out.add(Dropout(.5))
    model_out.add(Dense(1, activation='sigmoid'))

    optimizer = keras.optimizers.RMSprop(learning_rate=base_learning_rate)
    loss = keras.losses.BinaryCrossentropy()
    metrics = [keras.metrics.BinaryAccuracy(name="bool_acc", threshold=0.5), keras.metrics.AUC(name='auc')]
    model_out.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model_out


def rename_content_folder(instr):# keep
    if isinstance(instr, str) and '/content/' in instr:
        return '/Users/phil/Desktop/content' + instr.split('content')[-1]
    else:
        return instr


# train_and_val_dir
# base_test_dir
key_name = 'acc_test'
max_or_min = 'max'
models_2_load = [45]
# $%^&&&& user hard coded settings
patience = 8  # DETERMINES early stopping
save_freq = "epoch"  # leave this as epoch
period = 1  # DETERMINES how often it saves the checkpoints
epochs = 12  # DETERMINES how many epochs the model trains for if early stopping is never triggered
batch_size = 2 ** 7  # DETERMINES number of images per batch
save_best_only = True
monitor = 'val_loss'
###

lstm_len = 7
send_text('training started, you will receive a text when all training is done')

num_layers_to_unfreeze_list = [0]
for num_layers_unfreeze in num_layers_to_unfreeze_list:  # in the resnet base model
    # all_models_directory =  copy.deepcopy(train_and_val_dir)   # DETERMINES location for all model you will run #del
    # FOR EACH UNIQUE BASE DATSET start
    for TMP_basename in next(os.walk(train_and_val_dir))[1]:
        test_data_dir = base_test_dir + os.sep + TMP_basename  # # doesnt determine anything just filler, I define test data based on variable "test_path"
        unique_h5_train_val_dir = copy.deepcopy(
            TMP_basename)  # ^^^^^^^DETERMINES^^^^^^^ the name of the folder where each type of data is stored
        # FOR EACH UNIQUE BASE DATSET end
        for iiii, model_ind in enumerate(tqdm(models_2_load)):
            if 'ipynb_checkpoints' not in TMP_basename:
                data = all_data[model_ind]
                best_epochs = sorted_loadable_epochs(data, key_name, max_or_min)
                reload_model_epoch = best_epochs[0]
                D = make_D_local(copy.deepcopy(data['info']))
                test_path = utils.get_h5s(base_test_dir + os.sep + TMP_basename, 0)[0]

                LEARNING_RATE = D['base_learning_rate']
                # LEARNING_RATE = LEARNING_RATE/2
                LEARNING_RATE = 10 ** -5
                dropout_val = .1
                # righthere4
                model = re_build_model_TL(D['model_name_str'], D['class_numbers'],
                                          base_learning_rate=LEARNING_RATE,
                                          dropout_val=dropout_val, IMG_SIZE=D['IMG_SIZE'],
                                          reload_weights_file=D['epoch_dict'][reload_model_epoch],
                                          num_layers_unfreeze=num_layers_unfreeze)

                model = add_lstm_to_model(model, model.num_layers_in_base_model, base_learning_rate=LEARNING_RATE,
                                          lstm_len=lstm_len)

                # def add_LSTM_to_model(model, num_layers_in_base_model, base_model_layer_names):
                """
        image_tools.ImageBatchGenerator_simple(batch_size, [h5s], label_key = 'labels')
        3) ensure saving is correct saving the model to local and the moving it to its final destination, 
        4) ensure training val and test are using the correct simple generator (maybe dont need the test data??????? likely do though)
        5) copy all the alt labels to the H5s automatically  
        """

                label_ind = np.where(np.asarray(D['label_key_name']) == np.asarray(
                    list(model_maker.label_naming_shorthand_dict().keys())))[0][0]
                pred_key_save_name = 'MODEL_2_' + data[
                    'full_name'] + '__' + key_name + ' ' + max_or_min + '__epoch ' + str(
                    reload_model_epoch) + '__L_ind' + str(label_ind) + '__LABELS'
                print(pred_key_save_name)

                # FOR EACH UNIQUE MODEL start
                image_source_h5_directory_ending = D[
                    'image_source_h5_directory_ending']  # ^^^^^^^DETERMINES^^^^^^^ THE IMAGE SOURCE
                label_key = D[
                    'label_key_name']  # ^^^^^^^DETERMINES^^^^^^^ THE LABEL SOURCE choose the ind based on the print out
                # label_key = 'labels'#TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_

                tz = pytz.timezone('America/Los_Angeles')
                loc_dt = pytz.utc.localize(datetime.utcnow())
                LA_TIME = loc_dt.astimezone(tz)
                todays_version = LA_TIME.strftime("%Y_%m_%d_%H_%M_%S")

                a = D['model_save_dir_checkpoints']
                b = '/'.join([k for k in a.split('/') if k != ''])
                b = b.split('all_models/')[-1].split('/')[1:]

                base_data = unique_h5_train_val_dir.split('data_')[-1]
                b.insert(4, todays_version)
                # b.insert(0, type_trained_on)
                b.insert(0, base_data)
                new_base_folder_name = "LSTM_mod_" + str(model_ind) + "_" + str(num_layers_unfreeze)
                b.insert(0, new_base_folder_name)  ##################
                end_save_dir = os.sep.join(b)
                model_save_dir_checkpoints = rename_content_folder('/content/' + end_save_dir)

                ################## '/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/'+end_save_dir

                Path(model_save_dir_checkpoints).mkdir(parents=True, exist_ok=True)

                # info_dict = get_automated_model_info(train_and_val_dir, image_source_h5_directory_ending, test_data_dir, data_string_key = unique_h5_train_val_dir)
                info_dict = get_automated_model_info_TL(train_and_val_dir, image_source_h5_directory_ending, test_data_dir,
                                                        data_string_key='')

                # val_gen = image_tools.ImageBatchGenerator(batch_size, info_dict['h5_val'], label_key = label_key)
                val_gen = image_tools.ImageBatchGenerator_simple(batch_size, info_dict['h5_val'], label_key=label_key)
                # train_gen = image_tools.ImageBatchGenerator(batch_size, info_dict['h5_train'], label_key = label_key)
                train_gen = image_tools.ImageBatchGenerator_simple(batch_size, info_dict['h5_train'],
                                                                   label_key=label_key)
                # test_path = info_dict['h5_val']#TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_

                # test_gen = image_tools.ImageBatchGenerator(batch_size, test_path, label_key = label_key)
                test_gen = image_tools.ImageBatchGenerator_simple(batch_size, test_path, label_key=label_key)
                # test_gen = val_gen#TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_
                # test_gen = image_tools.ImageBatchGenerator(batch_size, '/content/small_test.h5', label_key = label_key)#TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_

                # FOR EACH UNIQUE MODEL end

                train_labels = image_tools.get_h5_key_and_concatenate([info_dict['h5_train']], label_key)
                val_labels = image_tools.get_h5_key_and_concatenate([info_dict['h5_val']], label_key)
                labels = np.concatenate(
                    (train_labels, val_labels))  # flkjasklfjklasjdflkjaslkfjalksdjflasdfkasdfjalsdfkjasdflk
                labels = train_labels
                class_numbers = np.unique(labels)  #### can use label info here but no need
                rebalance = class_weight.compute_class_weight('balanced', classes=class_numbers, y=labels.flatten())
                class_weights = {i: rebalance[i] for i in class_numbers}

                callbacks = basic_callbacks(model_save_dir_checkpoints, monitor=monitor, patience=patience,
                                            save_best_only=save_best_only, save_weights_only=True,
                                            save_freq=save_freq, period=period)

                plot_callback = transfer_learning.save_and_plot_history(test_gen, key_name=label_key, plot_metric_inds=[
                    1])  # TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_
                plot_callback = save_and_plot_history(test_gen, key_name=label_key, plot_metric_inds=[
                    1])  # TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_
                callbacks.append(
                    plot_callback)  # TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_

                ###FITTTTTTT

                print(D['image_source_h5_directory_ending'], label_key)
                history = model.fit(train_gen,
                                    epochs=epochs,
                                    validation_data=val_gen,
                                    class_weight=class_weights,
                                    callbacks=callbacks)

                for k in info_dict.keys():
                    D[k] = info_dict[k]

                training_info = pack_training_info(plot_callback)
                foo_save_and_plot(training_info, plot_callback, os.path.dirname(model_save_dir_checkpoints) + os.sep)

                # for saving purposed
                D['base_learning_rate'] = LEARNING_RATE
                D['num_layers_unfreeze'] = num_layers_unfreeze
                D['dropout_val'] = dropout_val

                base_dir = '/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/'

                model_save_dir_checkpoints_Gdrive = copy.deepcopy(model_save_dir_checkpoints)
                model_save_dir_checkpoints_Gdrive = base_dir + model_save_dir_checkpoints_Gdrive.split('/content')[-1]
                D['model_save_dir_checkpoints'] = model_save_dir_checkpoints_Gdrive
                D['model_save_dir'] = os.path.dirname(model_save_dir_checkpoints_Gdrive)

                # once finished transfer the completed trained model to the final directory in gdrive
                target_path = os.path.dirname(base_dir + os.sep + end_save_dir)
                Path(target_path).mkdir(parents=True, exist_ok=True)
                save_info_dict(D)  # save file after folder is created
                sync(os.path.dirname(model_save_dir_checkpoints), target_path, 'sync')

                shutil.rmtree(os.path.dirname(model_save_dir_checkpoints))

send_text('finished with all of it')
