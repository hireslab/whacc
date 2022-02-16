import sys
sys.path.append('/opt/conda/lib/python3.7/site-packages')
bd2 = '/home/jupyter/'

from whacc import utils, image_tools, transfer_learning, analysis
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow import keras
import tensorflow as tf
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


# In[16]:


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
        '[0, 1]- (not onset, onset)': 'only_onset'}
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


def get_automated_model_info(BASE_H5, image_source_h5_directory_ending, test_data_dir, data_string_key = "data"):
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
                    save_best_only=False, save_weights_only=True, save_freq="epoch", period = 1):
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
def unzip_and_place_h5s(bd, do_delete_zips = False):
  bd2= ''.join(bd.split('gdrive/My Drive'))
  shutil.copytree(bd, bd2)
  a = utils.get_files(bd2, '*.zip')
  for k in a:
    with zipfile.ZipFile(k, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(k))
    if do_delete_zips:
      os.remove(k)
def change_to_local_dir(bd):
  bd2= ''.join(bd.split('gdrive/My Drive'))
  return bd2
def change_to_gdrive_dir(bd):
  bd2= '/content/gdrive/My Drive/'.join(bd.split('/content/'))
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


def foo_run_all():
  #copy and unzip for colab
  global all_models_directory, test_data_dir, info_dict, transfer_learning

  local_dict = make_initial_folder(all_models_directory,
                                  unique_h5_train_val_dir)  # make data single frame directory and get that directory
  info_dict = info_dict_wrapper(info_dict, local_dict)  # wrap output into a dict

  BASE_H5 = info_dict['all_models_directory'] + os.sep + info_dict['unique_h5_train_val_dir']  # directory for all the data for a certain type og images(lag or regular etc)

  local_dict = get_automated_model_info(BASE_H5, image_source_h5_directory_ending, test_data_dir)  # basic data like directories and train and val set (automated form the directory)
  info_dict = info_dict_wrapper(info_dict, local_dict)  # wrap output into a dict

  # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
  local_dict = copy_over_new_labels(label_key_name, info_dict['image_h5_list'] + [info_dict['h5_test']],
                                    info_dict['label_h5_list'] + [info_dict[ 'h5_test_labels']])  # copy specific labels to the H5 of interest (this is all after the conversion from single frame)
  info_dict = info_dict_wrapper(info_dict, local_dict)  # wrap output into a dict

  # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$      make model     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
  # get all the labels
  train_gen = image_tools.ImageBatchGenerator(batch_size, [info_dict['h5_train']])
  val_gen = image_tools.ImageBatchGenerator(batch_size, [info_dict['h5_val']])
  test_gen = image_tools.ImageBatchGenerator(batch_size, [info_dict['h5_test']])

  train_labels = image_tools.get_h5_key_and_concatenate([info_dict['h5_train']])
  val_labels = image_tools.get_h5_key_and_concatenate([info_dict['h5_val']])
  labels = np.concatenate((train_labels, val_labels))

  model, class_weights, info_dict = build_model(info_dict, labels, model_name_str,
                                                base_learning_rate=base_learning_rate,
                                                dropout_val=dropout_val)

#   info_dict['BASE_H5'] = change_to_gdrive_dir(info_dict['BASE_H5'])
  model_save_dir = make_model_save_directory(info_dict)  # make a unique folder using standard folder struct ending in a unique date/time folder
  # ###change directory to gdrive in case it crashes
  # dir_2_change = ['model_save_dir_checkpoints', 'model_save_dir']
  # for k in dir_2_change:
  #   info_dict[k] = change_to_gdrive_dir(info_dict[k])

  model, class_weights, info_dict = build_model(info_dict,
                                                labels,
                                                model_name_str,
                                                base_learning_rate=base_learning_rate,
                                                dropout_val=dropout_val)

  callbacks = basic_callbacks(info_dict['model_save_dir_checkpoints'], monitor=monitor, patience=patience,
                              save_best_only=save_best_only, save_weights_only=True,
                              save_freq=save_freq, period=period)

  plot_callback = transfer_learning.save_and_plot_history(test_gen, key_name='labels', plot_metric_inds=[0])
  callbacks.append(plot_callback)

  ###FITTTTTTT
  history = model.fit(train_gen,
                      epochs=epochs,
                      validation_data=val_gen,
                      callbacks=callbacks,
                      class_weight=class_weights)
  #save finals checkpoint after model finishes
  model.save_weights(info_dict['model_save_dir_checkpoints']+os.sep+'final_epoch_cp.hdf5')

  training_info = pack_training_info(plot_callback)

  xx = '/content/colab_data2/model_testing/all_data/all_models/small_h5s/InceptionV3/3lag/on-off_set_and_one_after/'
  utils.get_files(xx, '*_cp.hdf5')

  transfer_learning.foo_save_and_plot

  foo_save_and_plot(training_info, plot_callback, info_dict['model_save_dir'])

  save_info_dict(info_dict)
  ### dont need to copy we changed the directory above
  # x1 = info_dict['model_save_dir']
  # x2 = change_to_gdrive_dir(x1)
  # shutil.copytree(x1, x2)


# In[18]:


## start

label_key_name_list = label_naming_shorthand_dict()  # get a list of label key names... they are really long to be specific
utils.print_list_with_inds(label_key_name_list)  # print them, then below use their index to choose them

all_models_directory = bd2+"model_testing/all_data/all_models/"  # DETERMINES location for all model you will run
test_data_dir = bd2+"model_testing/all_data/test_data/10_percent_holy_set/"  # ^^^^^^^DETERMINES^^^^^^^ location for test data (folder determined by the "image_source_h5_directory_ending" variable)
unique_h5_train_val_dir = 'regular_80_border'  # ^^^^^^^DETERMINES^^^^^^^ the name of the folder where each type of data is stored
image_source_h5_directory_ending = "/3lag/"  # ^^^^^^^DETERMINES^^^^^^^ THE IMAGE SOURCE
label_key_name = list(label_key_name_list.keys())[2]  # ^^^^^^^DETERMINES^^^^^^^ THE LABEL SOURCE choose the ind based on the print out

re_copy_and_unzip = True

model_name_str = 'ResNet50V2'  # ^^^^^^^DETERMINES^^^^^^^ model base
base_learning_rate = 10**-6  # DETERMINES rate of change for each epoch step
dropout_val = 0.5  # DETERMINES percentage of dropout for training data
patience = 15  # DETERMINES early stopping
save_freq = "epoch"  # leave this as epoch
period = 2 # DETERMINES how often it saves the checkpoints
epochs = 5000  # DETERMINES how many epochs the model trains for if early stopping is never triggered
batch_size = 200  # DETERMINES number of images per batch
save_best_only = True
monitor = 'val_loss'

info_dict = dict()


##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

from twilio.rest import Client

def send_text(mess):
  account_sid = "AC403675d69d88a93207f1cb80e9187c5e"
  auth_token = '111876a8b222ce0eff99b2ddaa47f2fb'
  client = Client(account_sid, auth_token)
  message = client.messages .create(
                      body =  mess, #Message you send
                      from_ = "+19199754333",#Provided phone number
                      to =    "+15023109622")#Your phone number
  message.sid


label_ind = 0
unique_h5_train_val_dir = 'regular_80_border'  # ^^^^^^^DETERMINES^^^^^^^ the name of the folder where each type of data is stored
image_source_h5_directory_ending = "/3lag/"  # ^^^^^^^DETERMINES^^^^^^^ THE IMAGE SOURCE
# 0 [0, 1, 2, 3, 4, 5]- (no touch, touch, onset, one after onset, offset, one after offset)
# 1 [0, 1, 2, 3]- (no touch, touch, onset, offset
# 2 [0, 1, 2]- (no event,from twilio.rest import Client
#
# def send_text(mess):
#   account_sid = "AC403675d69d88a93207f1cb80e9187c5e"
#   auth_token = '111876a8b222ce0eff99b2ddaa47f2fb'
#   client = Client(account_sid, auth_token)
#   message = client.messages .create(
#                       body =  mess, #Message you send
#                       from_ = "+19199754333",#Provided phone number
#                       to =    "+15023109622")#Your phone number
#   message.sid
#
#
# label_ind = 0
# unique_h5_train_val_dir = 'regular_80_border'  # ^^^^^^^DETERMINES^^^^^^^ the name of the folder where each type of data is stored
# image_source_h5_directory_ending = "/3lag/"  # ^^^^^^^DETERMINES^^^^^^^ THE IMAGE SOURCE
# # 0 [0, 1, 2, 3, 4, 5]- (no touch, touch, onset, one after onset, offset, one after offset)
# # 1 [0, 1, 2, 3]- (no touch, touch, onset, offset
# # 2 [0, 1, 2]- (no event, onset, offset)
# # 3 [0, 1]- (no touch, touch)
# # 4 [0, 1]- (not offset, offset)
# # 5 [0, 1]- (not onset, onset)
# label_key_name = list(label_key_name_list.keys())[label_ind]  # ^^^^^^^DETERMINES^^^^^^^ THE LABEL SOURCE choose the ind based on the print out
#
#
# batch_size = 2**8
#
# info_dict = dict()
# model_name_str = 'MobileNetV3Small'  # ^^^^^^^DETERMINES^^^^^^^ model base
# send_text('starting model --> ' + model_name_str)
# foo_run_all()
#
#
# info_dict = dict()
# model_name_str = 'InceptionV3'  # ^^^^^^^DETERMINES^^^^^^^ model base
# send_text('starting model --> ' + model_name_str)
# foo_run_all()
#
# info_dict = dict()
# model_name_str = 'MobileNetV3Large'  # ^^^^^^^DETERMINES^^^^^^^ model base
# send_text('starting model --> ' + model_name_str)
# foo_run_all()
#
# info_dict = dict()
# model_name_str = 'ResNet50V2'  # ^^^^^^^DETERMINES^^^^^^^ model base
# send_text('starting model --> ' + model_name_str)
# foo_run_all()
#
# # did not finish
# patience = 6  # DETERMINES early stopping
# batch_size = 2**6  # DETERMINES number of images per batch
# info_dict = dict()
# model_name_str = 'EfficientNetB7'  # ^^^^^^^DETERMINES^^^^^^^ model base
# send_text('starting model --> ' + model_name_str)
# foo_run_all()
#
#
#
# send_text('FINISHED FINISHED FINISHED FINISHED FINISHED FINISHED FINISHED FINISHED FINISHED ')
# send_text('cuda xxx') onset, offset)
# 3 [0, 1]- (no touch, touch)
# 4 [0, 1]- (not offset, offset)
# 5 [0, 1]- (not onset, onset)
label_key_name = list(label_key_name_list.keys())[label_ind]  # ^^^^^^^DETERMINES^^^^^^^ THE LABEL SOURCE choose the ind based on the print out


batch_size = 2**8

info_dict = dict()
model_name_str = 'MobileNetV3Small'  # ^^^^^^^DETERMINES^^^^^^^ model base
send_text('starting model --> ' + model_name_str)
foo_run_all()


info_dict = dict()
model_name_str = 'InceptionV3'  # ^^^^^^^DETERMINES^^^^^^^ model base
send_text('starting model --> ' + model_name_str)
foo_run_all()

info_dict = dict()
model_name_str = 'MobileNetV3Large'  # ^^^^^^^DETERMINES^^^^^^^ model base
send_text('starting model --> ' + model_name_str)
foo_run_all()

info_dict = dict()
model_name_str = 'ResNet50V2'  # ^^^^^^^DETERMINES^^^^^^^ model base
send_text('starting model --> ' + model_name_str)
foo_run_all()

# did not finish
patience = 6  # DETERMINES early stopping
batch_size = 2**6  # DETERMINES number of images per batch
info_dict = dict()
model_name_str = 'EfficientNetB7'  # ^^^^^^^DETERMINES^^^^^^^ model base
send_text('starting model --> ' + model_name_str)
foo_run_all()



send_text('FINISHED FINISHED FINISHED FINISHED FINISHED FINISHED FINISHED FINISHED FINISHED ')
send_text('cuda xxx')
