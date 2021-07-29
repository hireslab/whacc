from whacc import utils, image_tools, transfer_learning
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

"""

for running this with the real dataset I will have to make subset of the 10% dataset using 
    split_h5_loop_segments
i might also have to run 
    make_sure_frame_nums_exist
on all the files first  
to make sure frame_nums is in the keys 


here at the beginning I need to run the combine and split the touches. make sure they are all the same. I want to make the 
aug and split it before I do anything that is the OG that I upload. 
"""


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
        metrics = [keras.metrics.AUC(name='auc'),
                   keras.metrics.BinaryAccuracy(name="bool_acc", threshold=0.5)]
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
                      'dropout_val',
                      'class_weights']
    for k in wrap_vars_list:
        info_dict[k] = eval(k)
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
    todays_version = time.strftime("%Y_%m_%d_%H_%S", time.gmtime())
    a = os.sep
    base_data_dir = BASE_H5 + a + "data" + a
    base_dir_all_h5s = BASE_H5 + a + 'data' + a + 'single_frame' + a
    data_dir = base_data_dir + image_source_h5_directory_ending
    print('\nFOR IMAGES, 0 is train set, 1 is val set')
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
                    save_best_only=False, save_weights_only=True, save_freq=10):
    callbacks = []
    callbacks.append(keras.callbacks.EarlyStopping(monitor=monitor, patience=patience))
    add_path_name = "{loss:.8f}_{epoch:04d}_cp.hdf5"

    callbacks.append(keras.callbacks.ModelCheckpoint(
        save_checkpoint_filepath + os.sep + add_path_name,
        monitor=monitor,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        save_freq=save_freq))
    return callbacks



"""$$$$$$$$$$$$$$$$$$$$$$$$$$$      make/setup data     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

info_dict = dict()
all_models_directory = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/"  # DETERMINES location for all model you will run
# info_dict['test_data_dir'] =
test_data_dir =        '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/small_h5s/'  # DETERMINES location for test data (folder determined by the "image_source_h5_directory_ending" variable)

# '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data'

unique_h5_train_val_dir = 'small_h5s'  # DETERMINES the name of the folder where each type of data is stored
unique_h5_train_val_dir = 'regular_80_border'
image_source_h5_directory_ending = "/regular/"  # DETERMINES THE IMAGE SOURCE
make_labels_and_h5s = True  # DETERMINES if you re run the data transformation form the single frames into the regular, 3lag and diff_3lag

model_names = get_keras_model_names()  # get and print model names to choose from
model_name_str = 'InceptionV3'  # DETERMINES model base
base_learning_rate = 0.00001  # DETERMINES rate of change for each epoch step
dropout_val = 0.25  # DETERMINES percentage of dropout for training data
patience = 10  # DETERMINES early stopping
save_freq = 2  # DETERMINES how often it saves the checkpoints
epochs = 5000  # DETERMINES how many epochs the model trains for if early stopping is never triggered
batch_size = 1000  # DETERMINES number of images per batch

label_key_name_list = label_naming_shorthand_dict()  # get a list of label key names... they are really long to be specific
utils.print_list_with_inds(label_key_name_list, )  # print them, then below use their index to choose them
label_key_name = list(label_key_name_list.keys())[
    0]  # DETERMINES THE LABEL SOURCE choose the ind based on the print out

local_dict = make_initial_folder(all_models_directory,
                                 unique_h5_train_val_dir)  # make data single frame directory and get that directory
info_dict = info_dict_wrapper(info_dict, local_dict)  # wrap output into a dict

BASE_H5 = info_dict['all_models_directory'] + os.sep + info_dict[
    'unique_h5_train_val_dir']  # directory for all the data for a certain type og images(lag or regular etc)
if make_labels_and_h5s:  # make all the h5 files with lag and subtraction and regular
    base_dir_all_h5s = BASE_H5 + os.sep + 'data' + os.sep + 'single_frame' + os.sep
    utils.make_all_H5_types(
        base_dir_all_h5s)  # auto generate all the h5 types using a single set of flat (no color) image H5
    utils.make_alt_labels_h5s(base_dir_all_h5s)  # auto generate the different types of labels
local_dict = get_automated_model_info(BASE_H5, image_source_h5_directory_ending,
                                      test_data_dir)  # basic data like directories and train and val set (automated form the directory)
info_dict = info_dict_wrapper(info_dict, local_dict)  # wrap output into a dict

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
local_dict = copy_over_new_labels(label_key_name, info_dict['image_h5_list'] + [info_dict['h5_test']],
                                  info_dict['label_h5_list'] + [info_dict[
                                                                    'h5_test_labels']])  # copy specific labels to the H5 of interest (this is all after the conversion from single frame)
info_dict = info_dict_wrapper(info_dict, local_dict)  # wrap output into a dict

# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$      make model     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # get all the labels
# train_labels = image_tools.get_h5_key_and_concatenate([info_dict['h5_train']])
# val_labels = image_tools.get_h5_key_and_concatenate([info_dict['h5_val']])
# labels = np.concatenate((train_labels, val_labels))
#
# model, class_weights, info_dict = build_model(info_dict, labels, model_name_str, base_learning_rate=base_learning_rate,
#                                               dropout_val=dropout_val)
# model_save_dir = make_model_save_directory(
#     info_dict)  # make a unique folder using standard folder struct ending in a unique date/time folder
#
# model, class_weights, info_dict = build_model(info_dict, labels, model_name_str, base_learning_rate=base_learning_rate,
#                                               dropout_val=dropout_val)
#
# callbacks = basic_callbacks(info_dict['model_save_dir_checkpoints'], monitor='val_loss', patience=patience,
#                             save_best_only=False, save_weights_only=True, save_freq=save_freq)
#
# train_gen = image_tools.ImageBatchGenerator(batch_size, [info_dict['h5_train']])
# val_gen = image_tools.ImageBatchGenerator(batch_size, [info_dict['h5_val']])
# test_gen = image_tools.ImageBatchGenerator(batch_size, [info_dict['h5_test']])
#
# plot_callback = transfer_learning.save_and_plot_history(test_gen, key_name='labels', plot_metric_inds=[0])
# # transfer_learning.foo_save_and_plot(training_info, plot_callback, "/content/gdrive/My Drive/Colab data/SAMSON/model_tests/")
# callbacks.append(plot_callback)
# history = model.fit(train_gen,
#                     epochs=epochs,
#                     validation_data=val_gen,
#                     callbacks=callbacks,
#                     class_weight=class_weights)
#
#
# def pack_training_info(save_and_plot_history_var):
#     training_info = dict()
#     for k in ['L', 'all_logs', 'all_colors', 'logs_names', 'markers', 'matching_inds']:
#         training_info[k] = eval('save_and_plot_history_var.' + k)
#     # for k in training_info.keys():
#     #     if 'numpy.ndarray' in str(type(training_info[k])):
#     #         training_info[k] = training_info[k].tolist()
#     return training_info
# training_info = pack_training_info(plot_callback)
# utils.get_class_info(plot_callback)
#








# """
# make test set locations and add to info_dict
#
# add my custom callback with test set
#
#
# """
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
#
#
#
# # # #1 get a single folder named "single_frame" and put the test val and train sets in there, each of these folders wiwill be a different
# # # # combination of the of regular vs augmented images ... should i use only augmented images for the aug set? NO jsut add teh partial aug set
# # # base_dir_all_h5s = BASE_H5 + '/data/single_frame/' # MUST but a separate folder
# # # # with final 'single frame' H5s. i.e. split and aug and put into train val and test sets (ONLY THESE 3!!)
# # # # base_dir = str(Path(base_dir_all_h5s).parent.absolute())
# # make_labels_and_h5s = False
# # if make_labels_and_h5s: # make all the h5 files with lag and subtraction and regular
# #     utils.make_all_H5_types(base_dir_all_h5s) # auto generate all the h5 types using a single set of flat (no color) image H5
# #     utils.make_alt_labels_h5s(base_dir_all_h5s) # auto generate the different types of labels
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
#
# todays_version = time.strftime("%Y_%m_%d_%H_%S", time.gmtime())
#
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  DETERMINES THE IMAGE SOURCE  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# base_data_dir = BASE_H5+ "/data/"
# image_source_h5_directory_ending = "/regular/" # DETERMINES THE IMAGE SOURCE
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# data_dir = base_data_dir + image_source_h5_directory_ending
# image_h5_list = utils.get_h5s(data_dir)
# h5_test = image_h5_list[0]
# h5_train = image_h5_list[1]
# h5_val =image_h5_list[2]
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  DETERMINES THE LABEL SOURCE  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# labels_dir = base_data_dir + "/ALT_LABELS/"
# label_h5_list = utils.get_h5s(labels_dir)
# label_key_name_list = utils.print_h5_keys(label_h5_list[0], return_list=True)
# label_key_name = label_key_name_list[0] # DETERMINES THE LABEL SOURCE
# label_key_shorthand = label_naming_shorthand_dict(label_key_name)
#
# for img_src, lab_src in zip(image_h5_list, label_h5_list):
#     utils.copy_h5_key_to_another_h5(lab_src, img_src, label_key_name, 'labels')
#
#
# # make model for each model type
# """$$$$$$$$$$$  make a model maker for each of these model types with their respective characteristics  $$$$$$$$$$$$$$"""
# """
# break down into modules
#
# make_model
# class weights module
# num classes auto
# dropout input
# """
#
# model_names = get_keras_model_names()
#
# train_labels = image_tools.get_h5_key_and_concatenate([info_dict['h5_train']])
# val_labels = image_tools.get_h5_key_and_concatenate([info_dict['h5_val']])
# labels = np.concatenate((train_labels, val_labels))
#
# model_name_str = 'InceptionV3'# DETERMINES
# base_learning_rate = 0.00001# DETERMINES
# dropout_val = 0.25 # DETERMINES
#
#
#
#
# model, class_weights, info_dict = build_model(info_dict, labels, model_name_str, base_learning_rate = base_learning_rate, dropout_val = dropout_val)
#
#
# def basic_callbacks(save_checkpoint_filepath, monitor = 'val_loss', patience = 10,
#                     save_best_only = False, save_weights_only = True, save_freq = 10):
#     callbacks = []
#     callbacks.append(keras.callbacks.EarlyStopping(monitor = monitor,patience = patience)])
#
#     callbacks.append(keras.callbacks.ModelCheckpoint(
#         save_checkpoint_filepath,
#         monitor=monitor,
#         save_best_only=save_best_only,
#         save_weights_only=save_weights_only,
#         save_freq=save_freq))
#
# """
#
# add the
# .ckpt
# to save all the checkpoints
#
# and use .pb to save the whole model at the end
#
# check the save number using the
# filepath = ".\checkpoints\cp-{epoch:04d}.hdf5"
#
#
# """
#
# """
# sadfasdf
# """
# base_learning_rate = 0.00001
#
# class_numbers = [0, 1, 2, 3] # auto use unique(labels)
# num_classes = len(class_numbers)
# IMG_SIZE = 96 # All images will be resized to 96x96. This is the size of MobileNetV2 input sizes
#
#
# IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
# base_model = keras.applications.InceptionV3(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
# base_model.trainable = True
# x = base_model.output
# x = keras.layers.GlobalAveragePooling2D()(x)# global spatial average pooling layer
# x = Dense(512, activation='relu')(x)#  fully-connected layer
# x = Dropout(0.5)(x)
# predictions = Dense(num_classes, activation='softmax')(x)# fully connected output/classification layer
#
# base_learning_rate = 0.00001
# model = Model(inputs=base_model.input, outputs=predictions)
# METRICS = [
#       keras.metrics.SparseCategoricalAccuracy(name = 'acc')]
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=base_learning_rate),
#               loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), # default from_logits=False
#               metrics=METRICS)
#
#
#
# utils.print_h5_keys(f)
# import h5py
# with h5py.File(f, 'r')
#
#
#
# rebal_labels = image_tools.get_h5_key_and_concatenate([h5_train, h5_val], 'labels')
# start = time.time()
#
# # Fit model with a couple parameters
# EPOCHS = 1
#
# # Class imbalance weighting
# rebalance = class_weight.compute_class_weight('balanced',
#                                   [class_numbers], four_class_labels.flatten())
# class_weights = {i : rebalance[i] for i in range(4)}
#
# # Early stopping
# callbacks = [keras.callbacks.EarlyStopping (monitor = 'val_loss',patience = 10)]
#
#
#
# history = model.fit(my_training_batch_generator, epochs=EPOCHS,
#               validation_data= my_validation_batch_generator,
#               callbacks = callbamodel_name_strcks,
#               class_weight = class_weights)
#
#
#
#
# total_seconds = time.time() - start
# print('total run time :' + str(round(total_seconds/60)), ' minutes')
#
# todays_version = time.strftime("%Y%m%d%s", time.gmtime())
#
# SAVE_NAME = 'lagged_4_class_inceptionV3'
#
# end_dir = model_save_dir + '/' + SAVE_NAME + todays_version +'.ckpt'
# model.save_weights(end_dir)
# end_dir = model_save_dir + '/' + SAVE_NAME + todays_version +'.ckpt'
# model.save(end_dir)
# # # TO SAVE HISTORY INFO
# # hist_save_name = [model_save_dir+ 'HISTORY_'+ 1 + todays_version+'.json']
# # with open(hist_save_name[0], 'w') as f:
# #     json.dump(history.history, f)
#
# tmp1 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_single_frames/data/single_frame/'
# utils.make_all_H5_types(tmp1) # auto generate all the h5 types using a single set of flat (no color) image H5
# utils.make_alt_labels_h5s(tmp1) # auto generate the different types of labels
#
#
# tmp1 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/10_percent_holy_set/data/single_frame/'
# utils.make_all_H5_types(tmp1) # auto generate all the h5 types using a single set of flat (no color) image H5
# utils.make_alt_labels_h5s(tmp1) # auto generate the different types of labels
