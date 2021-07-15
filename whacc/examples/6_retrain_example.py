from whacc import utils
from whacc import image_tools
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.utils import class_weight
import numpy as np
import time
import os

split_h5_files = [
'/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/training.h5',
'/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/validation.h5'
]

all_models = utils.get_model_list('/Users/phil/Downloads/Colab data/model_iterations')
model_2_load = all_models[12]

# for timing \/
first_run_test_time = []
# for timing /\


# basic settings
epochs = 100
batch_size = 50
learning_rate = 10 ** -4
MOD_BASE_NAME = model_2_load.split(os.path.sep)[-1].split('.')[0]
# early stopping
monitor = 'val_loss'
patience = 10
start_tmp = time.time()

# ------
# CREATE TRAINING BATCH GENERATOR \/
my_training_batch_generator = image_tools.ImageBatchGenerator(batch_size, [split_h5_files[0]])
# my_validation_batch_generator
my_validation_batch_generator = image_tools.ImageBatchGenerator(batch_size, [split_h5_files[1]])
# CREATE TRAINING BATCH GENERATOR /\
# ------
# LOAD MODEL, MODEL SETTINGS \/
model = tf.keras.models.load_model(model_2_load)
# model = tf.keras.models.load_model('/content/temp_saves/savetest.hdf5')

model.optimizer.learning_rate = learning_rate  ###%%$^&$&&$
model.layers[0].trainable = False  # base
model.layers[1].trainable = False  #
model.layers[2].trainable = True  # class head

# callbacks = [keras.callbacks.EarlyStopping (monitor = 'val_loss', patience = 10 )] # Early stopping

# LOAD MODEL, MODEL SETTINGS /\
# ------
# RUN MODEL \/
start = time.time()
total_seconds = time.time() - start
print('total run time :' + str(round(total_seconds / 60)), ' minutes')
# Class imbalance weighting
all_y = utils.get_h5_key_and_concatenate(split_h5_files, key_name='labels')
rebalance = class_weight.compute_class_weight('balanced', [0, 1], all_y.flatten())
class_weights = {i: rebalance[i] for i in range(2)}

# Early stopping
callbacks = [keras.callbacks.EarlyStopping(monitor=monitor, patience=patience),
             ModelCheckpoint('/content/temp_saves/savetest.hdf5', save_best_only=True)]  ###%%$^&$&&$

history = model.fit(my_training_batch_generator, epochs=epochs,
                    validation_data=my_validation_batch_generator,
                    callbacks=callbacks,
                    class_weight=class_weights,
                    verbose=1)
# RUN MODEL /\
# ------
# save info for later reference \/
info_dict = {'monitor': callbacks[0].monitor,
             'patience': callbacks[0].patience,
             'epochs': epochs, 'total_seconds_to_retrain': total_seconds,
             'layers_re_trainable': [k.trainable for k in model.layers] * 1,
             're_learning_rate': model.optimizer.learning_rate}
# save info for later reference /\
#  predict for all regular H5s ** 8

first_run_test_time.append(time.time() - start_tmp)
# print(str((np.mean(first_run_test_time) * (L_TOTAL - len(first_run_test_time))) / 60) + ' minutes left ESTIMATED')
