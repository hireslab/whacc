
from whacc import model_maker

from whacc.model_maker import *


import h5py

from tensorflow.keras.layers import Dense, LSTM, Flatten, TimeDistributed, Conv2D, Dropout, GRU

from tensorflow.keras.utils import plot_model
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

"""# define some funcitons"""

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
    # base_model.summary()
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


def add_lstm_to_model(model_in, num_layers_in_base_model, base_learning_rate=10 ** -5, lstm_len=7):#keep
    # model_in.base_model_layer_names
    base_model = Model(model_in.input, model_in.layers[num_layers_in_base_model - 1].output)
    model_out = Sequential()
    model_out.add(TimeDistributed(base_model, input_shape=(lstm_len, 96, 96, 3)))
    model_out.add(TimeDistributed(Flatten()))
    model_out.add(LSTM(256, activation='relu', return_sequences=False))
    model_out.add(Dense(64, activation='relu'))
    # is 64 here the final number of features if so i want both of these to be bigger, check what I did for the OG models 
    model_out.add(Dropout(.2))
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

# these last 2 are used to change the directories easily when running on local 
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

"""# Load all_data for reload models and settings """

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
##### all_data = load_obj('/Users/phil/Downloads/all_data')
all_data = load_obj(foo_rename('/content/gdrive/My Drive/colab_data2/all_data'))

# """# copy over the small test data """
#
# train_and_val_dir = '/content/data/ALL_RETRAIN_H5_data____temp'
# srs_TandV_h5s, dst_TandV_h5s = utils.copy_file_filter(
#     '/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/small_h5s/data',
#     train_and_val_dir,
#     keep_strings='/3lag/',
#     remove_string='.zip',
#     overwrite=True,
#     just_print_what_will_be_copied=False, return_list_of_files=True)
#
# h5_imgs, h5_labels, inds_of_files = utils.copy_alt_labels_based_on_directory(srs_TandV_h5s)
# for i in inds_of_files:
#     utils.copy_over_all_non_image_keys(h5_labels[i], dst_TandV_h5s[i])
#
# base_test_dir = '/content/data/DATA_FULL____temp'
# srs_test_h5s, dst_test_h5s = utils.copy_file_filter(
#     '/content/gdrive/My Drive/colab_data2/model_testing/all_data/test_data/small_h5s',
#     base_test_dir,
#     keep_strings='/3lag/',
#     remove_string='.zip',
#     overwrite=True,
#     just_print_what_will_be_copied=False, return_list_of_files=True)
#
# h5_imgs, h5_labels, inds_of_files = utils.copy_alt_labels_based_on_directory(srs_test_h5s)
# for i in inds_of_files:
#     utils.copy_over_all_non_image_keys(h5_labels[i], dst_test_h5s[i])

# h5_imgs, h5_labels, inds_of_files = utils.copy_alt_labels_based_on_directory(srs_test_h5s)
# for i in inds_of_files:
#     utils.copy_over_all_non_image_keys(h5_labels[i], dst_test_h5s[i])
"""#set folder if not transforming """

h5_in = '/Users/phil/Desktop/pipeline_test/AH0407x160609_3lag.h5'

model_ind = 45
key_name = 'acc_test'
max_or_min = 'max'

data = all_data[model_ind]
best_epochs = sorted_loadable_epochs(data, key_name, max_or_min)
reload_model_epoch = best_epochs[0]
D = copy.deepcopy(data['info'])

epoch_model_file = D['epoch_dict'][reload_model_epoch]
epoch_model_file = foo_rename(epoch_model_file)

model = re_build_model_TL(D['model_name_str'],
                          D['class_numbers'],
                          base_learning_rate=0,
                          dropout_val=0,
                          IMG_SIZE=D['IMG_SIZE'],
                          reload_weights_file=epoch_model_file,
                          num_layers_unfreeze=0)

model_in, num_layers_in_base_model, base_learning_rate, lstm_len = model, model.num_layers_in_base_model, 0, 0
model_pred_features = Model(model_in.input, model_in.layers[num_layers_in_base_model +1].output)

model_pred_features.compile()

def convert_h5_to_feature_h5(model_out,in_generator, h5_new_full_file_name):
  h5c = image_tools.h5_iterative_creator(h5_new_full_file_name,
                                        overwrite_if_file_exists=True,
                                        max_img_height=1,
                                        max_img_width=2048,
                                        close_and_open_on_each_iteration=True,
                                        color_channel=False,
                                        add_to_existing_H5=False,
                                        ignore_image_range_warning=False,
                                        dtype_img = h5py.h5t.IEEE_F32LE,
                                        dtype_labels = h5py.h5t.IEEE_F32LE)

  for k in tqdm(range(in_generator.__len__())):
    x, y = in_generator.__getitem__(k)
    features = model_out.predict(x)
    h5c.add_to_h5(features, y)

in_gen = image_tools.ImageBatchGenerator(500, h5_in, label_key='labels')
new_name = h5_in.replace('.h5', '_features.h5')

convert_h5_to_feature_h5(model_pred_features,in_gen, new_name)


lstm_len = 7
base_dir = '/content/data/'

"""# set training settings"""

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
epochs = 100  # DETERMINES how many epochs the model trains for if early stopping is never triggered
batch_size = 2 ** 8  # DETERMINES number of images per batch
save_best_only = True
monitor = 'val_loss'
# lstm_len = 7 # set in "trasnform h5s" section 
num_layers_to_unfreeze_list = [0, 1, 2, 3, 4]

num_layers_to_unfreeze_list = [3]


lstm_base_folder_name = "TCN_test_12"
"""
######### add these to info dict #########

lstm_len
lstm_base_folder_name
num_layers_unfreeze -- should already be there 
"""

# send_text('training started, you will receive a text when all training is done')

"""#train it """

num_layers_unfreeze = 0 

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
              # D = make_D_local(copy.deepcopy(data['info']))
              D = copy.deepcopy(data['info'])
              test_path = utils.get_h5s(base_test_dir + os.sep + TMP_basename, 0)[0]

              LEARNING_RATE = D['base_learning_rate']
              # LEARNING_RATE = LEARNING_RATE/2
              LEARNING_RATE = 10 ** -5
              dropout_val = .05

              model = re_build_model_TL(D['model_name_str'], D['class_numbers'],
                                        base_learning_rate=LEARNING_RATE,
                                        dropout_val=dropout_val, IMG_SIZE=D['IMG_SIZE'],
                                        reload_weights_file=D['epoch_dict'][reload_model_epoch],
                                        num_layers_unfreeze=num_layers_unfreeze)
              
              # model = add_lstm_to_model(model, model.num_layers_in_base_model, base_learning_rate=LEARNING_RATE,
              #                           lstm_len=lstm_len)
              
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
              # b.insert(0, base_data)
              new_base_folder_name = lstm_base_folder_name  + '_' + "unfreeze_" + str(num_layers_unfreeze)
              b.insert(0, new_base_folder_name)  ##################
              end_save_dir = os.sep.join(b)
              model_save_dir_checkpoints = rename_content_folder('/content/' + end_save_dir)

              ################## '/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/'+end_save_dir

              Path(model_save_dir_checkpoints).mkdir(parents=True, exist_ok=True)

              # info_dict = get_automated_model_info_TL(train_and_val_dir, image_source_h5_directory_ending, test_data_dir, data_string_key = unique_h5_train_val_dir)
              info_dict = get_automated_model_info_TL(train_and_val_dir, image_source_h5_directory_ending, test_data_dir,
                                                    data_string_key='')
              # val_gen = image_tools.ImageBatchGenerator(batch_size, info_dict['h5_val'], label_key=label_key)
              # train_gen = image_tools.ImageBatchGenerator(batch_size, info_dict['h5_train'],label_key=label_key)
              # test_gen = image_tools.ImageBatchGenerator(batch_size, test_path, label_key=label_key)
              
              val_gen = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, info_dict['h5_val'], label_key=label_key, IMG_SIZE = 96)
              train_gen = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, info_dict['h5_train'],label_key=label_key, IMG_SIZE = 96)
              test_gen = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, test_path, label_key=label_key, IMG_SIZE = 96)

              # val_gen = ImageBatchGenerator_simple(batch_size, info_dict['h5_val'], label_key=label_key)
              # train_gen = ImageBatchGenerator_simple(batch_size, info_dict['h5_train'],label_key=label_key)
              # test_gen = ImageBatchGenerator_simple(batch_size, test_path, label_key=label_key)
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

              plot_callback = transfer_learning.save_and_plot_history(test_gen, key_name=label_key, plot_metric_inds=[1])  # TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_
              plot_callback = save_and_plot_history(test_gen, key_name=label_key, plot_metric_inds=[1])  # TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_
              callbacks.append(plot_callback)  # TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_
              print('your model is ready an. loaded now just chop off the head and get your features below')
              asdfasdfasdfasdf

data = all_data[model_ind]
best_epochs = sorted_loadable_epochs(data, key_name, max_or_min)
reload_model_epoch = best_epochs[0]
D = copy.deepcopy(data['info'])
model = re_build_model_TL(D['model_name_str'], D['class_numbers'],
                                        base_learning_rate=LEARNING_RATE,
                                        dropout_val=dropout_val, IMG_SIZE=D['IMG_SIZE'],
                                        reload_weights_file=D['epoch_dict'][reload_model_epoch],
                                        num_layers_unfreeze=num_layers_unfreeze)

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
                # D = make_D_local(copy.deepcopy(data['info']))
                D = copy.deepcopy(data['info'])
                test_path = utils.get_h5s(base_test_dir + os.sep + TMP_basename, 0)[0]

                LEARNING_RATE = D['base_learning_rate']
                # LEARNING_RATE = LEARNING_RATE/2
                LEARNING_RATE = 10 ** -5
                dropout_val = .05

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
                # b.insert(0, base_data)
                new_base_folder_name = lstm_base_folder_name  + '_' + "unfreeze_" + str(num_layers_unfreeze)
                b.insert(0, new_base_folder_name)  ##################
                end_save_dir = os.sep.join(b)
                model_save_dir_checkpoints = rename_content_folder('/content/' + end_save_dir)

                ################## '/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/'+end_save_dir

                Path(model_save_dir_checkpoints).mkdir(parents=True, exist_ok=True)

                # info_dict = get_automated_model_info_TL(train_and_val_dir, image_source_h5_directory_ending, test_data_dir, data_string_key = unique_h5_train_val_dir)
                info_dict = get_automated_model_info_TL(train_and_val_dir, image_source_h5_directory_ending, test_data_dir,
                                                     data_string_key='')
                # val_gen = image_tools.ImageBatchGenerator(batch_size, info_dict['h5_val'], label_key=label_key)
                # train_gen = image_tools.ImageBatchGenerator(batch_size, info_dict['h5_train'],label_key=label_key)
                # test_gen = image_tools.ImageBatchGenerator(batch_size, test_path, label_key=label_key)
                
                val_gen = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, info_dict['h5_val'], label_key=label_key, IMG_SIZE = 96)
                train_gen = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, info_dict['h5_train'],label_key=label_key, IMG_SIZE = 96)
                test_gen = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, test_path, label_key=label_key, IMG_SIZE = 96)

                # val_gen = ImageBatchGenerator_simple(batch_size, info_dict['h5_val'], label_key=label_key)
                # train_gen = ImageBatchGenerator_simple(batch_size, info_dict['h5_train'],label_key=label_key)
                # test_gen = ImageBatchGenerator_simple(batch_size, test_path, label_key=label_key)
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

                plot_callback = transfer_learning.save_and_plot_history(test_gen, key_name=label_key, plot_metric_inds=[1])  # TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_
                plot_callback = save_and_plot_history(test_gen, key_name=label_key, plot_metric_inds=[1])  # TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_
                callbacks.append(plot_callback)  # TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_TEMP_CHANGE_
                print('your model is ready an. loaded now just chop off the head and get your features below')
                asdfasdfasdfasdf

"""#convert images to features """

def convert_h5_to_feature_h5(model,in_generator, h5_new_full_file_name):
  h5c = image_tools.h5_iterative_creator(h5_new_full_file_name, 
                                        overwrite_if_file_exists=True, 
                                        max_img_height=1, 
                                        max_img_width=2048, 
                                        close_and_open_on_each_iteration=True, 
                                        color_channel=False, 
                                        add_to_existing_H5=False, 
                                        ignore_image_range_warning=False,
                                        dtype_img = h5py.h5t.IEEE_F32LE,
                                        dtype_labels = h5py.h5t.IEEE_F32LE)

  for k in tqdm(range(in_generator.__len__())):
    x, y = in_generator.__getitem__(k)
    features = model_out.predict(x)
    h5c.add_to_h5(features, y)
# for in_gen in [test_gen, val_gen, train_gen]:
#   new_name = '/feature_data/'.join(in_gen.H5_file_list[0].split('/data/'))
#   new_name = ''.join(new_name.split('____temp'))
#   Path(os.path.dirname(new_name)).mkdir(parents=True, exist_ok=True)
#   convert_h5_to_feature_h5(model_out,in_gen, new_name)
#   utils.copy_over_all_non_image_keys(in_gen.H5_file_list[0], new_name)

# num_layers_unfreeze = 0
# model = re_build_model_TL(D['model_name_str'], D['class_numbers'],
#                           base_learning_rate=LEARNING_RATE,
#                           dropout_val=dropout_val, IMG_SIZE=D['IMG_SIZE'],
#                           reload_weights_file=D['epoch_dict'][reload_model_epoch],
#                           num_layers_unfreeze=num_layers_unfreeze)
# # model.summary()

# val_gen = image_tools.ImageBatchGenerator(batch_size, info_dict['h5_val'], label_key=label_key)
# train_gen = image_tools.ImageBatchGenerator(batch_size, info_dict['h5_train'],label_key=label_key)
# test_gen = image_tools.ImageBatchGenerator(batch_size, test_path, label_key=label_key)

num_layers_unfreeze = 0
model = re_build_model_TL(D['model_name_str'], D['class_numbers'],
                          base_learning_rate=LEARNING_RATE,
                          dropout_val=dropout_val, IMG_SIZE=D['IMG_SIZE'],
                          reload_weights_file=D['epoch_dict'][reload_model_epoch],
                          num_layers_unfreeze=num_layers_unfreeze)

model_in, num_layers_in_base_model, base_learning_rate, lstm_len = model, model.num_layers_in_base_model, LEARNING_RATE, lstm_len
model_out = Model(model_in.input, model_in.layers[num_layers_in_base_model +1].output)


optimizer = keras.optimizers.RMSprop(learning_rate=base_learning_rate)
loss = keras.losses.BinaryCrossentropy()
metrics = [keras.metrics.BinaryAccuracy(name="bool_acc", threshold=0.5), keras.metrics.AUC(name='auc')]
model_out.compile(optimizer=optimizer, loss=loss, metrics=metrics)



all_lists = ['/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data',
              '/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border/data',
              '/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/small_h5s/data',
              '/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/']

h5_file_IMG

h5_file_IMG = []
for src in all_lists:
  dst = '/content/gdrive/My Drive/colab_data2/model_testing_features_data/feature_data/'
  tmp1 = utils.copy_file_filter(src,
                        dst, 
                        keep_strings='/3lag/', 
                        remove_string=['.DS_Store', '.zip'], 
                        overwrite=False, 
                        just_print_what_will_be_copied=True, 
                        disable_tqdm=False, 
                        return_list_of_files=True)

h5_file_IMG = ['/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/small_h5s/data/3lag/small_train_3lag.h5',
'/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/small_h5s/data/3lag/small_val_3lag.h5',
'/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/3lag/train_3lag.h5',
'/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/3lag/val_3lag.h5',
'/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border/data/3lag/train_3lag.h5',
'/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border/data/3lag/val_3lag.h5',
'/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/data_AH0407_160613_JC1003_AAAC/3lag/AH0407_160613_JC1003_AAAC_3lag.h5',
'/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/data_AH0667_170317_JC1241_AAAA/3lag/AH0667_170317_JC1241_AAAA_3lag.h5',
'/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/data_AH0698_170601_PM0121_AAAA/3lag/AH0698_170601_PM0121_AAAA_3lag.h5',
'/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/data_AH0705_171105_PM0175_AAAB/3lag/AH0705_171105_PM0175_AAAB_3lag.h5',
'/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/data_AH1120_200322__/3lag/AH1120_200322___3lag.h5',
'/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/data_AH1131_200326__/3lag/AH1131_200326___3lag.h5',
'/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/data_ANM234232_140118_AH1026_AAAA/3lag/ANM234232_140118_AH1026_AAAA_3lag.h5',
'/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/data_ANM234232_140120_AH1030_AAAA_a/3lag/ANM234232_140120_AH1030_AAAA_a_3lag.h5']

h5_file_IMG = ['/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/small_h5s/data/3lag/small_train_3lag.h5',
'/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/small_h5s/data/3lag/small_val_3lag.h5']

# tmp1 = ['/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/3lag/val_3lag.h5',
# '/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border/data/3lag/val_3lag.h5']


# tmp2 = ['/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/ALT_LABELS/val_regular_ALT_LABELS.h5',
#  '/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border/data/ALT_LABELS/val_ALT_LABELS.h5']

# for i1, i2 in zip(tmp1, tmp2):
#   utils.copy_over_all_non_image_keys(i2, i1)

batch_size = 500
temp_path = '/content/temp/'
for k in h5_file_IMG:
  fname = os.path.basename(k)
  dname = os.path.dirname(k)
  
  utils.copy_file_filter(dname, 
                        temp_path, 
                        keep_strings=fname, 
                        remove_string=['.DS_Store', '.zip'])
  
  h5_to_convert = utils.get_h5s(temp_path)[0]
  if utils.h5_key_exists(h5_to_convert, '[0, 1]- (no touch, touch)'):
    in_gen = image_tools.ImageBatchGenerator(batch_size, h5_to_convert, label_key='[0, 1]- (no touch, touch)')
  else:
    in_gen = image_tools.ImageBatchGenerator(batch_size, h5_to_convert, label_key='labels')


  new_name = '/feature_data/'.join(in_gen.H5_file_list[0].split('/temp/'))
  print(new_name)
  Path(os.path.dirname(new_name)).mkdir(parents=True, exist_ok=True)
  convert_h5_to_feature_h5(model_out,in_gen, new_name)
  utils.copy_over_all_non_image_keys(in_gen.H5_file_list[0], new_name)
  

  src = '/content/feature_data/'
  dst = '/content/gdrive/My Drive/LIGHT_GBM/FEATURE_DATA/'+dname.split('My Drive/')[-1]  
  
  utils.copy_file_filter(src,
                        dst, 
                        keep_strings='', 
                        remove_string=None, 
                        overwrite=False, 
                        just_print_what_will_be_copied=False, 
                        disable_tqdm=False, 
                        return_list_of_files=False)
  
  shutil.rmtree(temp_path)
  shutil.rmtree(src)

dst = '/content/gdrive/My Drive/LIGHT_GBM/FEATURE_DATA/'+dname.split('My Drive/')[-1]
dst

# tmpdir = '/content/gdrive/My Drive/colab_data2/model_testing/all_data/final_predictions'
# utils.copy_file_filter(tmpdir, 
#                         temp_path, 
#                         keep_strings=k_string)

# h5_file_IMG = [
#     # '/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_ANM234232_140118_AH1026_AAAA/3lag/ANM234232_140118_AH1026_AAAA_3lag.h5',
#     # '/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_ANM234232_140120_AH1030_AAAA_a/3lag/ANM234232_140120_AH1030_AAAA_a_3lag.h5',
#     # '/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_AH0407_160613_JC1003_AAAC/3lag/AH0407_160613_JC1003_AAAC_3lag.h5',
#     '/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_AH0667_170317_JC1241_AAAA/3lag/AH0667_170317_JC1241_AAAA_3lag.h5',
#     '/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_AH0698_170601_PM0121_AAAA/3lag/AH0698_170601_PM0121_AAAA_3lag.h5',
#     '/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_AH0705_171105_PM0175_AAAB/3lag/AH0705_171105_PM0175_AAAB_3lag.h5',
#     '/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_AH1120_200322__/3lag/AH1120_200322___3lag.h5',
#     '/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL/data_AH1131_200326__/3lag/AH1131_200326___3lag.h5']
# shutil.rmtree(temp_path)
# src = '/content/feature_data/'
# shutil.rmtree(src)

# h5_file_IMG = ['/content/gdrive/My Drive/Colab data/testing_TL_10_20_30/data_ANM234232_140120_AH1030_AAAA_a/3lag/ANM234232_140120_AH1030_AAAA_a_3lag___30/train.h5',
#                '/content/gdrive/My Drive/Colab data/testing_TL_10_20_30/data_ANM234232_140120_AH1030_AAAA_a/3lag/ANM234232_140120_AH1030_AAAA_a_3lag___30/val.h5']

h5_file_IMG = ['/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/small_h5s/data',
              '/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data',
              '/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border/data',
              '/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/']

# label_key = '[0, 1]- (no touch, touch)' # make above when pre loading the model 
# label_key = 'labels' # make above when pre loading the model 

batch_size = 500
temp_path = '/content/temp/'
for k_string in h5_file_IMG:
  k_string = os.sep.join(k_string.split(os.sep)[-2:])

  utils.copy_file_filter('/content/gdrive/My Drive/Colab data/testing_TL_10_20_30/', 
                        temp_path, 
                        keep_strings=k_string)
  
  h5_to_convert = utils.get_h5s(temp_path)[0]
  try:
    in_gen = image_tools.ImageBatchGenerator(batch_size, h5_to_convert, label_key='[0, 1]- (no touch, touch)')
  except:
    in_gen = image_tools.ImageBatchGenerator(batch_size, h5_to_convert, label_key='labels')


  new_name = '/feature_data/testing_TL_10_20_30/'.join(in_gen.H5_file_list[0].split('/temp/'))
  print(new_name)
  asdfasdf
  Path(os.path.dirname(new_name)).mkdir(parents=True, exist_ok=True)
  convert_h5_to_feature_h5(model_out,in_gen, new_name)
  utils.copy_over_all_non_image_keys(in_gen.H5_file_list[0], new_name)
  

  src = '/content/feature_data/'
  dst = '/content/gdrive/My Drive/colab_data2/model_testing_features_data/feature_data/'
  utils.copy_file_filter(src,
                        dst, 
                        keep_strings='', 
                        remove_string=None, 
                        overwrite=False, 
                        just_print_what_will_be_copied=False, 
                        disable_tqdm=False, 
                        return_list_of_files=False)
  shutil.rmtree(temp_path)
  shutil.rmtree(src)

new_name

# label_key = '[0, 1]- (no touch, touch)' # make above when pre loading the model 
batch_size = 4000
h5_to_convert = utils.get_h5s(temp_path)[0]
in_gen = image_tools.ImageBatchGenerator(batch_size, h5_to_convert, label_key=label_key)


new_name = '/feature_data/DATA_FULL/'.join(in_gen.H5_file_list[0].split('/temp/'))
print(new_name)

Path(os.path.dirname(new_name)).mkdir(parents=True, exist_ok=True)
convert_h5_to_feature_h5(model_out,in_gen, new_name)
utils.copy_over_all_non_image_keys(in_gen.H5_file_list[0], new_name)
os.remove(h5_to_convert)

os.sep.join(k_strings.split(os.sep)[-2:])
# k_strings.split(os.sep)

utils.copy_file_filter('/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/', 
                       '/content/temp/', 
                       keep_strings=[ '/3lag/AH0667_170317_JC1241_AAAA_3lag.h5'])

# /Volumes/GoogleDrive-114825029448473821206/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/data_AH0698_170601_PM0121_AAAA/3lag/AH0698_170601_PM0121_AAAA_3lag.h5
utils.copy_file_filter('/content/gdrive/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/', 
                       '/content/temp/', 
                       keep_strings=[ '/3lag/AH0698_170601_PM0121_AAAA_3lag.h5'])

label_key = '[0, 1]- (no touch, touch)'
batch_size = 4000
for h5_to_convert in utils.get_h5s('/content/temp/'):
  in_gen = image_tools.ImageBatchGenerator(batch_size, h5_to_convert, label_key=label_key)


  new_name = '/feature_data/DATA_FULL_in_range_only/'.join(in_gen.H5_file_list[0].split('/temp/'))
  print(new_name)
  Path(os.path.dirname(new_name)).mkdir(parents=True, exist_ok=True)
  convert_h5_to_feature_h5(model_out,in_gen, new_name)
  utils.copy_over_all_non_image_keys(in_gen.H5_file_list[0], new_name)

src = '/content/feature_data/'
dst = '/content/gdrive/My Drive/colab_data2/model_testing_features_data/feature_data/'
utils.copy_file_filter(src,
                       dst, 
                       keep_strings='', 
                       remove_string=None, 
                       overwrite=False, 
                       just_print_what_will_be_copied=False, 
                       disable_tqdm=False, 
                       return_list_of_files=False)

h5_to_convert = '/content/data/data_ANM234232_140118_AH1026_AAAA/3lag/ANM234232_140118_AH1026_AAAA_3lag.h5'
label_key = '[0, 1]- (no touch, touch)'
in_gen = image_tools.ImageBatchGenerator(batch_size, h5_to_convert, label_key=label_key)


new_name = '/feature_data/'.join(in_gen.H5_file_list[0].split('/data/'))

Path(os.path.dirname(new_name)).mkdir(parents=True, exist_ok=True)
convert_h5_to_feature_h5(model_out,in_gen, new_name)
utils.copy_over_all_non_image_keys(in_gen.H5_file_list[0], new_name)

src = '/content/feature_data/'
dst = '/content/gdrive/My Drive/colab_data2/model_testing_features_data/feature_data/8_testing_h5s/'
utils.copy_file_filter(src,
                       dst, 
                       keep_strings='', 
                       remove_string=None, 
                       overwrite=False, 
                       just_print_what_will_be_copied=False, 
                       disable_tqdm=False, 
                       return_list_of_files=False)

2

new_name = '/feature_data/'.join(in_gen.H5_file_list[0].split('/data/'))
new_name = ''.join(new_name.split('____temp'))
Path(os.path.dirname(new_name)).mkdir(parents=True, exist_ok=True)
convert_h5_to_feature_h5(model,in_gen, new_name)
utils.copy_over_all_non_image_keys(in_gen.H5_file_list[0], new_name)

# for k in tqdm(range(test_gen.__len__())):
#     x, y = test_gen.__getitem__(k)
#     features = model_out.predict(x)
#     asdf

# '/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/',
# '/content/gdrive/My Drive/colab_data2/model_testing_features_data/feature_data
utils.copy_file_filter('/content/feature_data/', 
                       '/content/gdrive/My Drive/colab_data2/model_testing_features_data/feature_data/regular_80_border_aug_0_to_9/', 
                       keep_strings='', 
                       remove_string=None, 
                       overwrite=False, 
                       just_print_what_will_be_copied=False, 
                       disable_tqdm=False, 
                       return_list_of_files=False)

# '/content/gdrive/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/',
# '/content/gdrive/My Drive/colab_data2/model_testing_features_data/feature_data
utils.copy_file_filter('/content/feature_data/', 
                       '/content/gdrive/My Drive/colab_data2/model_testing_features_data/feature_data/regular_80_border/', 
                       keep_strings='', 
                       remove_string=None, 
                       overwrite=False, 
                       just_print_what_will_be_copied=False, 
                       disable_tqdm=False, 
                       return_list_of_files=False)
