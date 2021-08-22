from whacc import utils, transfer_learning

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow import keras

from sklearn.utils import class_weight

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


# import sys
# sys.path.append('/opt/conda/lib/python3.7/site-packages')
# bd2 = '/home/jupyter/'


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


def label_naming_shorthand_dict(name_key=None):
    label_naming_shorthand_dict = {
        '[0, 1, 2, 3, 4, 5]- (no touch, touch, onset, one after onset, offset, one after offset)': 'on-off_set_and_one_after',
        '[0, 1, 2, 3]- (no touch, touch, onset, offset': 'on-off_set',
        '[0, 1, 2]- (no event, onset, offset)': 'only_on-off_set',
        '[0, 1]- (no touch, touch)': 'regular',
        '[0, 1]- (not offset, offset)': 'only_offset',
        '[0, 1]- (not onset, onset)': 'only_onset',
        '[0, 1, 2, 3]- (no touch, touch, one after onset, offset)': 'overlap_whisker_on-off'}
    if name_key is None:
        return label_naming_shorthand_dict
    else:
        return label_naming_shorthand_dict[name_key]


def info_dict_wrapper(info_dict, local_dict):
    for k in local_dict.keys():
        info_dict[k] = local_dict[k]
    return info_dict


def make_initial_folder(all_models_directory, unique_h5_train_val_dir):
    single_frame_dir = all_models_directory + os.sep + unique_h5_train_val_dir + os.sep + 'data' + os.sep + 'single_frame'
    Path(single_frame_dir).mkdir(parents=True, exist_ok=True)
    return locals()
    # return single_frame_dir


def get_automated_model_info(BASE_H5, image_source_h5_directory_ending, test_data_dir):
    tz = pytz.timezone('America/Los_Angeles')
    loc_dt = pytz.utc.localize(datetime.utcnow())
    LA_TIME = loc_dt.astimezone(tz)
    todays_version = LA_TIME.strftime("%Y_%m_%d_%H_%M_%S")
    del tz
    del loc_dt
    del LA_TIME
    a = os.sep
    base_data_dir = BASE_H5 + a + "data" + a
    base_dir_all_h5s = BASE_H5 + a + 'data' + a + 'single_frame' + a
    data_dir = base_data_dir + image_source_h5_directory_ending
    print('\nFOR IMAGES, 0 is train set, 1 is val set')
    print(data_dir)
    image_h5_list = utils.get_h5s(data_dir)
    h5_train = image_h5_list[0]
    h5_val = image_h5_list[1]
    labels_dir = base_data_dir + a + "ALT_LABELS" + a
    print('\nFOR LABELS,0 is train set, 1 is val set')
    label_h5_list = utils.get_h5s(labels_dir)
    print('\nSelect from the following label structures...')
    label_key_name_list = utils.print_h5_keys(label_h5_list[0], return_list=True)
    h5_test_labels = utils.get_h5s(test_data_dir + a + "ALT_LABELS" + a, print_h5_list=False)[0]
    h5_test = utils.get_h5s(test_data_dir + a + image_source_h5_directory_ending + a, print_h5_list=False)[0]
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


def make_model_save_directory(info_dict, make_folder=True):
    naming_list = ['model_name_str', 'image_source_h5_directory_ending', 'label_key_shorthand', 'todays_version']
    model_save_dir: str = copy.deepcopy(info_dict['BASE_H5'])
    for k in naming_list:
        model_save_dir += os.sep + info_dict[k] + os.sep
    info_dict['model_save_dir'] = model_save_dir
    info_dict['model_save_dir_checkpoints'] = model_save_dir + os.sep + 'checkpoints'
    if make_folder:
        Path(info_dict['model_save_dir_checkpoints']).mkdir(parents=True, exist_ok=True)
    return model_save_dir


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
    return D


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
    return all_data
# def reload_info(base_dir):
#   s = os.sep
#   b = base_dir.split('all_models'+s)[-1].split(s)
#   reload_dict = {'data_type':b[0],
#                 'model_name':b[1],
#                 'img_type':b[2],
#                 'label_name_shorthand':b[3],
#                 'date_string':b[4]}
#
#   label_name_dict = label_naming_shorthand_dict()
#   for k in label_name_dict.keys():
#     if reload_dict['label_name_shorthand'] == label_name_dict[k]:
#       reload_dict['label_type'] = k
#       class_numbers = eval(k.split('- ')[0])
#   reload_dict['class_numbers'] = class_numbers
#   return reload_dict
