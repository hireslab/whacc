ind_to_do = [0,1,2,3,4,5,6,7]

import sys
sys.path.append('/opt/conda/lib/python3.7/site-packages')
bd2 = '/home/jupyter/'

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
# drive.mount('/content/gdrive')
from whacc import model_maker


from whacc.model_maker import *
import itertools
from tqdm import tqdm
import h5py
# !pip install twilio
from twilio.rest import Client

# utils.get_dict_info(data['info']['epoch_dict'])

# utils.get_dict_info(data['info']['epoch_dict'])


def inds_sorted_data(a, key_name, max_or_min):
  """
  a is the data form all_data looped through 
  """
  log_ind = np.where(key_name == a['logs_names'])[0][0]
  val_list = a['all_logs'][:, log_ind]
  if max_or_min == 'max':
    max_arg_sort = np.flip(np.argsort(val_list))+1
  elif max_or_min == 'min':
    max_arg_sort = np.argsort(val_list)+1
  else:
    raise ValueError("""max_or_min must be a string set to 'max' or 'min'""")
  return max_arg_sort
def sorted_loadable_epochs(a, key_name, max_or_min):
  """
  a is the data form all_data looped through 
  """
  arg_sort_inds = inds_sorted_data(data, key_name, max_or_min)
  arg_sort_inds[np.argmax(arg_sort_inds)] = -1
  saved_epoch_numbers = np.asarray(list(data['info']['epoch_dict'].keys()))
  sorted_loadable_epochs = []
  for k in arg_sort_inds:
    if k in saved_epoch_numbers:
      sorted_loadable_epochs.append(k)
  return sorted_loadable_epochs

# load all saved data
def reload_info_dict(base_dir):
  info_dict_path = utils.get_files(base_dir, '*info_dict.json')[0]
  with open(info_dict_path, 'r') as f:
        D = json.load(f)
  # replace the directory to the main directory
  x = 'all_models'
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
  D['label_num_names'] = D['label_key_name'].split('- (')[-1][:-1].split(', ')# here use this to label the y limits
  D['label_nums'] = eval(D['label_key_name'].split(']')[0]+']')

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

def re_build_model(model_name_str, class_numbers, base_learning_rate=0.00001, dropout_val=None, IMG_SIZE=96, labels = None):

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

all_models_directory = '/home/jupyter/all_models/'
h5_data = '/home/jupyter/H5_data/'

all_data = load_model_data(all_models_directory)

# /home/jupyter/
base_dir = '/home/jupyter/H5_data/'
base_loc_dir = base_dir
alt_label_dir= base_loc_dir+os.sep+ 'ALT_LABELS/'
h5files = utils.get_h5s(base_dir+'/OG')
h5_alt_labels = utils.get_h5s(alt_label_dir)
h5_alt_labels_drop_dir = base_dir+os.sep+ 'ALT_LABELS/'

h5_base_names = []
for k1 in h5files:
  k = '-'.join(k1.split('__'))
  k = k.split('-')[0]
  h5_base_names.append(os.path.basename(k))


# do it this way so we can use the inds to name the alt_labels

H5_list_subset = [h5_base_names[k] for k in ind_to_do]

key_name = 'acc_test'
max_or_min = 'max'
start_at = -9999999
#####
key_names = []
for data in (all_data):
  key_names.append(data['info']['label_key_name'])
model_inds_sorted_by_label_type = np.argsort(key_names)

kn = ''
for i, model_ind in enumerate(tqdm(model_inds_sorted_by_label_type)):
  if i>= start_at:
    data = all_data[model_ind]
    best_epochs = sorted_loadable_epochs(data, key_name, max_or_min)
    reload_model_epoch = best_epochs[0]
    D = data['info']

    model = re_build_model(D['model_name_str'], D['class_numbers'],
                        base_learning_rate=D['base_learning_rate'],
                        dropout_val=D['dropout_val'], IMG_SIZE=D['IMG_SIZE'])
    model.load_weights(D['epoch_dict'][reload_model_epoch]) # load model weights
    label_ind = np.where(np.asarray(D['label_key_name'])  == np.asarray(list(model_maker.label_naming_shorthand_dict().keys())))[0][0]
    pred_key_save_name = 'MODEL_2_' + data['full_name'] + '__' + key_name + ' ' + max_or_min + '__epoch '+ str(reload_model_epoch)+ '__L_ind'+ str(label_ind)+'__LABELS'

    h5_on_local_subset = utils.lister_it(utils.get_h5s(base_loc_dir, print_h5_list=False), keep_strings=D['image_source_h5_directory_ending'])
    rematch = utils.lister_it(utils.get_h5s(base_loc_dir, print_h5_list=False), 'single_frame')

    h5_on_local_subset = utils.lister_it(h5_on_local_subset, keep_strings=H5_list_subset)
    rematch = utils.lister_it(rematch, keep_strings=H5_list_subset)


    for h5_image_SRC, retmp in zip(h5_on_local_subset, rematch):
      label_src = utils.get_files(h5_alt_labels_drop_dir, os.path.basename(retmp)[:-3]+'*')[0]
      img_gen = image_tools.ImageBatchGenerator(50, [h5_image_SRC])
      pred = model.predict(img_gen) #predict
      with h5py.File(label_src, 'r+') as hf:
        try:
          hf.create_dataset(pred_key_save_name, data=pred)
        except:
          del hf[pred_key_save_name]
          time.sleep(10)
          hf.create_dataset(pred_key_save_name, data=pred)
    kn = D['label_key_name']





