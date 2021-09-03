"""
things to do

fix the update plotting for the regular (not Ipython) python

allow for either saving all of the weights

allow for plotting test set every X
OOOORRRRRR
plot X% of the test set... nah this is not as useful

allow for plotting different plot using subplot I guess is the easiest way
"""
from whacc import utils

 # setting up plotting when using google colab or notebook (though only tested in colab)
if utils.isnotebook():

    from IPython import get_ipython

    ipython = get_ipython()
    ipython.magic("matplotlib inline")
from IPython import display

import copy
from whacc import image_tools
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from sklearn.utils import class_weight
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import json


def foo_make_model(model_2_load, dropout=None):
    """
  test if the trainable setting stay to what they are for output model
  """
    model2 = tf.keras.models.load_model(model_2_load)
    if dropout is not None:
        feature_batch = model2.layers[0].output
        dropout_layer = tf.keras.layers.Dropout(dropout, input_shape=model2.layers[0].input.shape)
        # Model Stacking
        model = tf.keras.Sequential([
            model2.layers[0],
            model2.layers[1],
            dropout_layer,
            model2.layers[2]
        ])
        model.layers[0].trainable = False  # base
        model.layers[1].trainable = False  #
        model.layers[2].trainable = False  # dropout layer
        model.layers[3].trainable = True  # class head

    else:

        model = tf.keras.Sequential([
            model2.layers[0],
            model2.layers[1],
            model2.layers[2]
        ])
        model.layers[0].trainable = False  # base
        model.layers[1].trainable = False  #
        model.layers[2].trainable = True  # class head
    METRICS = [
        keras.metrics.AUC(name='auc'),
        keras.metrics.BinaryAccuracy(name="bool_acc", threshold=0.5),
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
    model.compile(optimizer=tf.keras.optimizers.RMSprop(10 ** -6),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=METRICS)

    return model


###########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$###########
def foo_start_running(model_2_load, training_h5_list, validation_h5_list, dropout=None, epochs=100,
                      batch_size=50, learning_rate=10 ** -6, patience=2, monitor='val_loss',
                      key_name='labels', verbose_=1, save_name=None, model_save_dir=None,
                      save_best_only=False, add_callback_list=[]):
    if save_name is not None or model_save_dir is not None:
        assert os.path.isdir(model_save_dir), "model_save_dir must be a directory"
        assert save_name is not None, "if you want to save the model model_save_dir and save_name must not be None"
        MOD_BASE_NAME = model_2_load.split(os.path.sep)[-1].split('.')[0]
        ModelCheckpoint_save_name = model_save_dir + MOD_BASE_NAME + '_' + save_name + '.hdf5'
        add_callback_list.append(ModelCheckpoint(ModelCheckpoint_save_name, save_best_only=save_best_only))

    model = foo_make_model(model_2_load, dropout=dropout)
    # CREATE TRAINING BATCH GENERATOR \/
    my_training_batch_generator = image_tools.ImageBatchGenerator(batch_size, training_h5_list)
    # my_validation_batch_generator
    my_validation_batch_generator = image_tools.ImageBatchGenerator(batch_size, validation_h5_list)

    model.optimizer.learning_rate = learning_rate

    # Class imbalance weighting
    all_y = image_tools.get_h5_key_and_concatenate(training_h5_list, key_name=key_name)
    rebalance = class_weight.compute_class_weight('balanced', [0, 1], all_y.flatten())
    class_weights = {i: rebalance[i] for i in range(2)}

    callbacks = [keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)]
    for k in add_callback_list:
        callbacks.append(k)
    start = time.time()
    history = model.fit(my_training_batch_generator,
                        epochs=epochs,
                        validation_data=my_validation_batch_generator,
                        callbacks=callbacks,
                        class_weight=class_weights,
                        verbose=verbose_)
    run_time = time.time() - start
    print('total run time :' + str(round(run_time / 60)), ' minutes')
    info_dict = {'monitor': callbacks[0].monitor,
                 'patience': callbacks[0].patience,
                 'epochs': epochs,
                 'layers_re_trainable': [k.trainable for k in model.layers] * 1,
                 're_learning_rate': float(model.optimizer.learning_rate),
                 'model_2_load': model_2_load,
                 'training_h5_list': training_h5_list,
                 'validation_h5_list': validation_h5_list,
                 'dropout': dropout,
                 'run_time': run_time}
    return info_dict


###########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$###########
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


###########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$###########
#  for auto saving get the next number for saving in order
def foo_get_next_save_num(dir_str):
    num_names = [0]
    for k in utils.get_files(dir_str, '*mod_test_*'):
        try:
            num_names.append(int(k.split('.png')[-2][-3:]))
        except:
            pass
        try:
            num_names.append(int(k.split('.json')[-2][-3:]))
        except:
            pass
    save_num = str(np.max(num_names) + 1).zfill(4)
    return save_num


###########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$###########
# text for figure
def make_text_for_fig(training_info):
    tmp_str = ''
    for k in training_info.keys():
        if k in ['monitor', 'patience', 'epochs', 'layers_re_trainable', 're_learning_rate', 'model_2_load',
                 'training_h5_list', 'validation_h5_list', 'dropout', 'run_time']:
            a = copy.deepcopy(training_info[k])
            if k in ['model_2_load']:
                a = "\n  " + a.split(os.path.sep)[-1]
            elif k in ['training_h5_list', 'validation_h5_list']:
                for i, kk in enumerate(a):
                    a[i] = "\n  " + str(i + 1) + ') ' + kk.split(os.path.sep)[-1]
                a = ''.join(a)
            tmp_str = tmp_str + k + ' : ' + str(a) + '\n'
        tmp2 = tmp_str.split('\\n')
        tmp_str = '\n'.join(tmp2)
    return tmp_str


###########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$###########
# save the info file
def pack_training_info(training_info, save_and_plot_history_var):
    for k in ['L', 'all_logs', 'all_colors', 'logs_names', 'markers', 'matching_inds']:
        training_info[k] = eval('save_and_plot_history_var.' + k)
    for k in training_info.keys():
        if 'numpy.ndarray' in str(type(training_info[k])):
            training_info[k] = training_info[k].tolist()
    return training_info


###########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$###########
def plot_train_hist(training_info, all_inds, ylims=[None, None], text_str=None):
    _ = plt.figure(figsize=[15, 8])
    _ = plt.gcf().text(.95, .3, text_str, fontsize=20)
    c = np.tile(training_info['all_colors'][:training_info['L']], 3)
    leg_app = []
    for ind in all_inds:
        for k, kline in zip(np.asarray(training_info['matching_inds'])[:, ind],
                            np.tile(training_info['markers'][:3], 20)):
            x = np.asarray(training_info['all_logs'])[:, k]
            _ = plt.plot(x, linestyle=kline, color=c[k])
            leg_app.append(training_info['logs_names'][k])
        _ = plt.legend(leg_app)
    plt.ylim(ylims)


###########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$###########

def foo_save_and_plot(training_info, save_and_plot_history_1, save_loc):
    text_str = make_text_for_fig(training_info)
    training_info = pack_training_info(training_info, save_and_plot_history_1)
    save_num = foo_get_next_save_num(save_loc)

    # make figure and save
    plot_train_hist(training_info, [1, 2], [.9, 1], text_str)
    plt.savefig(save_loc + 'mod_test_fig_ACC_AUC_' + save_num + '.png', bbox_inches="tight")
    plot_train_hist(training_info, [0], [0, 0.25], text_str)
    plt.savefig(save_loc + 'mod_test_fig_LOSS_' + save_num + '.png', bbox_inches="tight")

    # save training info
    with open(save_loc + 'mod_test_data_' + save_num + '.json', 'w') as f:
        json.dump(training_info, f)
