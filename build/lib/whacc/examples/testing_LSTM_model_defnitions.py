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

"""
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""

def send_text(mess):#keep
    print('oops texting failed but the code goes on')


def save_obj(obj, name):#keep
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):#keep
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def foo_rename(instr):#keep
    if isinstance(instr, list):
        for i, k in enumerate(instr):
            instr[i] = foo_rename(k)
        return instr
    else:
        return '/Volumes/GoogleDrive-114825029448473821206/My Drive' + instr.split('My Drive')[-1]

# bd2 = "/content/gdrive/My Drive/colab_data2/"
# all_models_directory = bd2+"/model_testing/all_data/all_models/"
# all_data = load_model_data(all_models_directory)


# save_obj(all_data, '/content/all_data2')
all_data = load_obj('/Users/phil/Downloads/all_data')
"""
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""
train_and_val_dir = '/Users/phil/Desktop/content/ALL_RETRAIN_H5_data____temp'
srs_TandV_h5s, dst_TandV_h5s = utils.copy_file_filter(
    foo_rename('/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/small_h5s/data'),
    train_and_val_dir,
    keep_strings='/3lag/',
    remove_string='.zip',
    overwrite=True,
    just_print_what_will_be_copied=False, return_list_of_files=True)



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

"""
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""
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






"""
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""


from whacc.transfer_learning import save_and_plot_history
# train_and_val_dir
# base_test_dir
key_name = 'acc_test'
max_or_min = 'max'
models_2_load = [45]
# $%^&&&& user hard coded settings
patience = 8  # DETERMINES early stopping
save_freq = "epoch"  # leave this as epoch
period = 1  # DETERMINES how often it saves the checkpoints
epochs = 2  # DETERMINES how many epochs the model trains for if early stopping is never triggered
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

                # info_dict = get_automated_model_info_TL(train_and_val_dir, image_source_h5_directory_ending, test_data_dir, data_string_key = unique_h5_train_val_dir)
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
                utils.copy_file_filter(os.path.dirname(model_save_dir_checkpoints), target_path)
                shutil.rmtree(os.path.dirname(model_save_dir_checkpoints))

send_text('finished with all of it')

def copy_file_filter(src, dst, keep_strings='', remove_string=None, overwrite=False,
                     just_print_what_will_be_copied=False, disable_tqdm=False, return_list_of_files=False):

utils.copy_file_filtercopy_file_filter(os.path.dirname(model_save_dir_checkpoints), target_path)

