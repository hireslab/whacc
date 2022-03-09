# get_ipython().system('pip install whacc -U')


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
import pytz
import json
from math import isclose, sqrt
from IPython import display

import pandas as pd
import time

# import lightgbm as lgb
# from datetime import datetime

import datetime
import h5py



def shift_with_frame_nums(data_frame_in, frame_nums, shift_by, add_name_str = None):
    if add_name_str is None:
        add_name_str = 'shifted_'+str(shift_by)
    data_frame = data_frame_in.copy()
    for i1, i2 in utils.loop_segments(frame_nums):
        data_frame[i1:i2] = data_frame_in[i1:i2].shift(shift_by)
    data_frame = data_frame.add_suffix('_'+add_name_str)
    return data_frame

def rolling_mean_with_frame_nums(data_frame_in, frame_nums, window, add_name_str=None):
    if add_name_str is None:
        add_name_str = 'rolling_mean_' + str(window)
    data_frame = data_frame_in.copy()
    for i1, i2 in utils.loop_segments(frame_nums):
        df_rolling = data_frame_in[i1:i2].rolling(window=window, min_periods=window)
        data_frame[i1:i2] = df_rolling.mean().shift(-(window // 2)).astype(np.float32)
    data_frame = data_frame.add_suffix('_' + add_name_str)
    return data_frame


def rolling_std_with_frame_nums(data_frame_in, frame_nums, window, add_name_str=None):
    if add_name_str is None:
        add_name_str = 'rolling_std_' + str(window)
    data_frame = data_frame_in.copy()
    for i1, i2 in utils.loop_segments(frame_nums):
        df_rolling = data_frame_in[i1:i2].rolling(window=window, min_periods=window)
        data_frame[i1:i2] = df_rolling.std().shift(-(window // 2)).astype(np.float32)
    data_frame = data_frame.add_suffix('_' + add_name_str)
    return data_frame


def concat_numpy_memmory_save(df__x, base_dir):
    np_list = utils.get_files(base_dir + os.sep, '*.npy')
    init_shape = list(df__x.shape)
    s = init_shape[1]
    init_shape[1] = s * len(np_list) + s
    final_x = np.zeros(init_shape, dtype=np.float32)

    final_x[:, :s] = df__x.astype('float32')
    for i, f in tqdm(enumerate(np_list)):
        ii = i + 1
        i1 = s * ii
        i2 = s * ii + s
        final_x[:, i1:i2] = np.load(f, allow_pickle=True).astype('float32')
    return final_x


def transform_x_data_np(df_in, frame_nums_in, base_dir):
    if os.path.isdir(base_dir):
        print("WARNING WARNING WARNING WARNING directory already exists please delete directory to redo, returning")
        return
    save_dir = base_dir + os.sep
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    st = time.time()
    cnt = 0
    for shift_by in tqdm([-2, -1, 1, 2]):
        x = shift_with_frame_nums(df_in, frame_nums_in, shift_by).to_numpy()
        np.save(save_dir + str(cnt) + '.npy', x)
        cnt += 1
    for roll_by in tqdm([3, 7]):
        x = rolling_mean_with_frame_nums(df_in, frame_nums_in, roll_by)
        np.save(save_dir + str(cnt) + '.npy', x)
        cnt += 1
        x = rolling_std_with_frame_nums(df_in, frame_nums_in, roll_by)
        np.save(save_dir + str(cnt) + '.npy', x)
        cnt += 1
    print(time.time() - st)


def make_data_chunks(h5, base_dir, do_transform_data=True, label_key='[0, 1]- (no touch, touch)'):
    time_length = 1  # do not change we shift time in pandas dataframe now
    batch_size = None
    feature_len = 2048
    label_index_to_lstm_len = None
    edge_value = np.nan
    remove_any_time_points_with_edges = False
    #     'labels'
    h5 = utils.make_list(h5)
    test_x = None
    for tmp_h5 in h5:

        # if label_key not in utils.print_h5_keys(tmp_h5, return_list=True, do_print=False):
        #     label_key='labels'
        G_test = image_tools.ImageBatchGenerator_feature_array(time_length, batch_size, tmp_h5, label_key=label_key,
                                                               feature_len=feature_len,
                                                               label_index_to_lstm_len=label_index_to_lstm_len,
                                                               edge_value=edge_value,
                                                               remove_any_time_points_with_edges=remove_any_time_points_with_edges)
        tmp_frame_nums_test = image_tools.get_h5_key_and_concatenate(tmp_h5, 'frame_nums')
        tmpx, tmpy = G_test.__getitem__(0)

        print(tmpx.shape, tmpy.shape)
        if test_x is not None:
            test_x = np.vstack((test_x, tmpx))
            test_y = np.concatenate((test_y, tmpy.flatten()))
            frame_nums_test = np.concatenate((frame_nums_test, tmp_frame_nums_test.flatten()))

        else:
            test_x = tmpx
            test_y = tmpy.flatten()
            frame_nums_test = tmp_frame_nums_test.flatten()

    df_test_x = pd.DataFrame(test_x)

    del test_x  # save RAM
    if do_transform_data:
        transform_x_data_np(df_test_x, frame_nums_test, base_dir)
    return df_test_x, test_y


####


def foo_rename(instr):
    if isinstance(instr, list):
        for i, k in enumerate(instr):
            instr[i] = foo_rename(k)
        return instr
    else:
        tmp1 = instr.split('My Drive')[-1]
        a = tmp1[0]
        a2 = os.sep
        x = a + 'Volumes' + a + 'GoogleDrive-114825029448473821206' + a + 'My Drive' + tmp1
        x = a2.join(x.split(a))
        return x


base_dir_all = r'G:\My Drive\colab_data2\model_testing_features_data\feature_data\DATA_FULL'
base_dir_all = foo_rename(base_dir_all)
base_dir_all
os.path.isdir(base_dir_all)

h5 = '/Users/phil/Desktop/AH0407_160613_JC1003_AAAC_3lag.h5'
utils.print_h5_keys(h5)
with h5py.File(h5, 'r') as h:
    tmp1 = h['images'][:20]
    print(h['model_name_used_as_feature_extractor'][0])

mod_name = 'MODEL_3_regular 80 border aug 0 to 9__ResNet50V2__3lag__regular__acc_test max__epoch 6__L_ind3__LABELS_-_2021_07_22_06_01_29'
"""ADD MODEL USED TO GET FEATURES FOR REPEATABILITY """
with h5py.File(h5, 'r+') as h:
    dname = 'model_name_used_as_feature_extractor_' + mod_name
    h.create_dataset(dname, data=0)

with h5py.File(h5, 'r+') as h:
    h.create_dataset_like('images')

#
def shift_with_frame_nums(data_frame_in, frame_nums, shift_by, add_name_str = None):
    if add_name_str is None:
        add_name_str = 'shifted_'+str(shift_by)
    data_frame = data_frame_in.copy()
    for i1, i2 in utils.loop_segments(frame_nums):
        data_frame[i1:i2] = data_frame_in[i1:i2].shift(shift_by)
    data_frame = data_frame.add_suffix('_'+add_name_str)
    return data_frame


h5_in = '/Users/phil/Desktop/AH0407_160613_JC1003_AAAC_3lag.h5'
shift_by = -1
add_name_str = None





class feature_maker():
    def __init__(self, h5_in, frame_nums = None, operational_key = 'images'):
        self.h5_in = h5_in
        if frame_nums is None:
            self.frame_nums = image_tools.get_h5_key_and_concatenate(h5_in, 'frame_nums')
        self.operational_key = operational_key
        self.set_operation_key()

        # self.xxxx = 11111
        # self.xxxx = 11111
        # self.xxxx = 11111
        # self.xxxx = 11111
        # self.xxxx = 11111
        # self.xxxx = 11111
    def set_operation_key(self, key_name = None):
        if key_name is not None:
            self.operational_key = key_name
        self.data = pd.DataFrame(image_tools.get_h5_key_and_concatenate(h5_in, self.operational_key))


    # def _rolling_(self, window, shift_from_center = 0, min_periods = None):
    #     if min_periods is None:
    #         min_periods = window
    #     add_name_list = ['FD__' + self.operational_key + 'rolling_',   '_W_' + str(window) + '_SFC_' + str(shift_from_center) + '_MP_' + str(min_periods)]
    #     data_frame = self.data.copy()
    #     return window, shift_from_center, min_periods, add_name_list, data_frame
    def rolling(self, window, operation, shift_from_center = 0, min_periods = None):
        if min_periods is None:
            min_periods = window
        add_name_list = ['FD__' + self.operational_key + '_rolling_',   '_W_' + str(window) + '_SFC_' + str(shift_from_center) + '_MP_' + str(min_periods)]
        data_frame = self.data.copy()
        add_name_str = operation.join(add_name_list)
        for i1, i2 in utils.loop_segments(self.frame_nums):
            df_rolling = self.data[i1:i2].rolling(window=window, min_periods=min_periods, center=True)
            tmp_func = eval('df_rolling.' + operation)
            data_frame[i1:i2] = tmp_func().shift(shift_from_center).astype(np.float32)
        return data_frame, add_name_str
    def shift(self, shift_by):
        add_name_str = 'FD__' + self.operational_key + '_shift_' + str(shift_by)
        data_frame = self.data.copy()
        for i1, i2 in utils.loop_segments(self.frame_nums):
            data_frame[i1:i2] = self.data[i1:i2].shift(shift_by).astype(np.float32)
        return data_frame, add_name_str

h5_in = '/Users/phil/Desktop/AH0407_160613_JC1003_AAAC_3lag.h5'
feature_data = image_tools.get_h5_key_and_concatenate(h5_in, 'images')
feature_data = pd.DataFrame(feature_data)


frame_nums = image_tools.get_h5_key_and_concatenate(h5_in, 'frame_nums')
frame_class = np.zeros(np.sum(frame_nums)).astype(int)
for i, (i1, i2) in enumerate(utils.loop_segments(frame_nums)):
    frame_class[i1:i2] = i
frame_class


feature_data['frame_nums'] =frame_class


grouped = feature_data.groupby('frame_# get_ipython().system('pip install whacc -U')


from tqdm.autonotebook import tqdm
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
import pytz
import json
from math import isclose, sqrt
from IPython import display

import pandas as pd
import time

# import lightgbm as lgb
# from datetime import datetime

import datetime
import h5py



def shift_with_frame_nums(data_frame_in, frame_nums, shift_by, add_name_str = None):
    if add_name_str is None:
        add_name_str = 'shifted_'+str(shift_by)
    data_frame = data_frame_in.copy()
    for i1, i2 in utils.loop_segments(frame_nums):
        data_frame[i1:i2] = data_frame_in[i1:i2].shift(shift_by)
    data_frame = data_frame.add_suffix('_'+add_name_str)
    return data_frame

def rolling_mean_with_frame_nums(data_frame_in, frame_nums, window, add_name_str=None):
    if add_name_str is None:
        add_name_str = 'rolling_mean_' + str(window)
    data_frame = data_frame_in.copy()
    for i1, i2 in utils.loop_segments(frame_nums):
        df_rolling = data_frame_in[i1:i2].rolling(window=window, min_periods=window)
        data_frame[i1:i2] = df_rolling.mean().shift(-(window // 2)).astype(np.float32)
    data_frame = data_frame.add_suffix('_' + add_name_str)
    return data_frame


def rolling_std_with_frame_nums(data_frame_in, frame_nums, window, add_name_str=None):
    if add_name_str is None:
        add_name_str = 'rolling_std_' + str(window)
    data_frame = data_frame_in.copy()
    for i1, i2 in utils.loop_segments(frame_nums):
        df_rolling = data_frame_in[i1:i2].rolling(window=window, min_periods=window)
        data_frame[i1:i2] = df_rolling.std().shift(-(window // 2)).astype(np.float32)
    data_frame = data_frame.add_suffix('_' + add_name_str)
    return data_frame


def concat_numpy_memmory_save(df__x, base_dir):
    np_list = utils.get_files(base_dir + os.sep, '*.npy')
    init_shape = list(df__x.shape)
    s = init_shape[1]
    init_shape[1] = s * len(np_list) + s
    final_x = np.zeros(init_shape, dtype=np.float32)

    final_x[:, :s] = df__x.astype('float32')
    for i, f in tqdm(enumerate(np_list)):
        ii = i + 1
        i1 = s * ii
        i2 = s * ii + s
        final_x[:, i1:i2] = np.load(f, allow_pickle=True).astype('float32')
    return final_x


def transform_x_data_np(df_in, frame_nums_in, base_dir):
    if os.path.isdir(base_dir):
        print("WARNING WARNING WARNING WARNING directory already exists please delete directory to redo, returning")
        return
    save_dir = base_dir + os.sep
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    st = time.time()
    cnt = 0
    for shift_by in tqdm([-2, -1, 1, 2]):
        x = shift_with_frame_nums(df_in, frame_nums_in, shift_by).to_numpy()
        np.save(save_dir + str(cnt) + '.npy', x)
        cnt += 1
    for roll_by in tqdm([3, 7]):
        x = rolling_mean_with_frame_nums(df_in, frame_nums_in, roll_by)
        np.save(save_dir + str(cnt) + '.npy', x)
        cnt += 1
        x = rolling_std_with_frame_nums(df_in, frame_nums_in, roll_by)
        np.save(save_dir + str(cnt) + '.npy', x)
        cnt += 1
    print(time.time() - st)


def make_data_chunks(h5, base_dir, do_transform_data=True, label_key='[0, 1]- (no touch, touch)'):
    time_length = 1  # do not change we shift time in pandas dataframe now
    batch_size = None
    feature_len = 2048
    label_index_to_lstm_len = None
    edge_value = np.nan
    remove_any_time_points_with_edges = False
    #     'labels'
    h5 = utils.make_list(h5)
    test_x = None
    for tmp_h5 in h5:

        # if label_key not in utils.print_h5_keys(tmp_h5, return_list=True, do_print=False):
        #     label_key='labels'
        G_test = image_tools.ImageBatchGenerator_feature_array(time_length, batch_size, tmp_h5, label_key=label_key,
                                                               feature_len=feature_len,
                                                               label_index_to_lstm_len=label_index_to_lstm_len,
                                                               edge_value=edge_value,
                                                               remove_any_time_points_with_edges=remove_any_time_points_with_edges)
        tmp_frame_nums_test = image_tools.get_h5_key_and_concatenate(tmp_h5, 'frame_nums')
        tmpx, tmpy = G_test.__getitem__(0)

        print(tmpx.shape, tmpy.shape)
        if test_x is not None:
            test_x = np.vstack((test_x, tmpx))
            test_y = np.concatenate((test_y, tmpy.flatten()))
            frame_nums_test = np.concatenate((frame_nums_test, tmp_frame_nums_test.flatten()))

        else:
            test_x = tmpx
            test_y = tmpy.flatten()
            frame_nums_test = tmp_frame_nums_test.flatten()

    df_test_x = pd.DataFrame(test_x)

    del test_x  # save RAM
    if do_transform_data:
        transform_x_data_np(df_test_x, frame_nums_test, base_dir)
    return df_test_x, test_y


####


def foo_rename(instr):
    if isinstance(instr, list):
        for i, k in enumerate(instr):
            instr[i] = foo_rename(k)
        return instr
    else:
        tmp1 = instr.split('My Drive')[-1]
        a = tmp1[0]
        a2 = os.sep
        x = a + 'Volumes' + a + 'GoogleDrive-114825029448473821206' + a + 'My Drive' + tmp1
        x = a2.join(x.split(a))
        return x


base_dir_all = r'G:\My Drive\colab_data2\model_testing_features_data\feature_data\DATA_FULL'
base_dir_all = foo_rename(base_dir_all)
base_dir_all
os.path.isdir(base_dir_all)

h5 = '/Users/phil/Desktop/AH0407_160613_JC1003_AAAC_3lag.h5'
utils.print_h5_keys(h5)
with h5py.File(h5, 'r') as h:
    tmp1 = h['images'][:20]
    print(h['model_name_used_as_feature_extractor'][0])

mod_name = 'MODEL_3_regular 80 border aug 0 to 9__ResNet50V2__3lag__regular__acc_test max__epoch 6__L_ind3__LABELS_-_2021_07_22_06_01_29'
"""ADD MODEL USED TO GET FEATURES FOR REPEATABILITY """
with h5py.File(h5, 'r+') as h:
    dname = 'model_name_used_as_feature_extractor_' + mod_name
    h.create_dataset(dname, data=0)

with h5py.File(h5, 'r+') as h:
    h.create_dataset_like('images')

#
def shift_with_frame_nums(data_frame_in, frame_nums, shift_by, add_name_str = None):
    if add_name_str is None:
        add_name_str = 'shifted_'+str(shift_by)
    data_frame = data_frame_in.copy()
    for i1, i2 in utils.loop_segments(frame_nums):
        data_frame[i1:i2] = data_frame_in[i1:i2].shift(shift_by)
    data_frame = data_frame.add_suffix('_'+add_name_str)
    return data_frame


h5_in = '/Users/phil/Desktop/AH0407_160613_JC1003_AAAC_3lag.h5'
shift_by = -1
add_name_str = None




"""
could use with h5py instead of get_and_concat and then create the class for each set of frame nums, basically make a 
wrapper class that wraps this class
"""

class feature_maker():
    def __init__(self, h5_in, frame_num_ind = None, frame_nums = None, operational_key = 'images'):
        self.h5_in = h5_in
        self.frame_num_ind = frame_num_ind
        if frame_nums is None:
            with h5py.File(self.h5_in, 'r') as h:
                if self.frame_num_ind is None:
                    self.frame_nums = copy.deepcopy(h['frame_nums'][:])
                else:
                    self.frame_nums = copy.deepcopy([h['frame_nums'][self.frame_num_ind]])
                    tmp_inds = np.asarray(utils.loop_segments(self.frame_nums, returnaslist=True))
                    self.data_inds = tmp_inds[:, self.frame_num_ind]
        self.operational_key = operational_key
        self.set_operation_key(operational_key)
    def set_operation_key(self, key_name):
        self.operational_key = key_name
        with h5py.File(self.h5_in, 'r') as h:
            if self.frame_num_ind is None:
                self.data = pd.DataFrame(copy.deepcopy(h[self.operational_key][:]))
            else:
                a = self.data_inds
                self.data = pd.DataFrame(copy.deepcopy(h[self.operational_key][a[0]:a[1]]))

    def rolling(self, window, operation, shift_from_center = 0, min_periods = None, kwargs = {}):
        if min_periods is None:
            min_periods = window
        add_name_list = ['FD__' + self.operational_key + '_rolling_',
                         '_W_' + str(window) + '_SFC_' + str(shift_from_center) + '_MP_' + str(min_periods)]
        data_frame = self.data.copy()
        add_name_str = operation.join(add_name_list)
        for i1, i2 in utils.loop_segments(self.frame_nums):
            df_rolling = self.data[i1:i2].rolling(window=window, min_periods=min_periods, center=True)
            tmp_func = eval('df_rolling.' + operation)
            data_frame[i1:i2] = tmp_func(**kwargs).shift(shift_from_center).astype(np.float32)
        return data_frame, add_name_str
    def shift(self, shift_by):
        add_name_str = 'FD__' + self.operational_key + '_shift_' + str(shift_by)
        data_frame = self.data.copy()
        for i1, i2 in utils.loop_segments(self.frame_nums):
            data_frame[i1:i2] = self.data[i1:i2].shift(shift_by).astype(np.float32)
        return data_frame, add_name_str

    def operate(self, operation, add_name_str = '', kwargs = {}):
        add_name_str = 'FD__' + self.operational_key + '_'+operation+'_' + add_name_str
        data_frame = self.data.copy()
        for i1, i2 in utils.loop_segments(self.frame_nums):
            data_frame[i1:i2] = eval('self.data[i1:i2].' + operation + '(**kwargs).astype(np.float32)')
        return data_frame, add_name_str


h5_in = '/Users/phil/Desktop/AH0407_160613_JC1003_AAAC_3lag.h5'

FM = feature_maker(h5_in, frame_num_ind = 0)

# tmp1, tmp2 = FM.shift(2)
# tmp1 = pd.DataFrame(np.random.rand(10, 5))
kwargs = {'periods':-1}
data_frame, add_name_str = FM.operate('shift', kwargs = kwargs)

kargs = {'min_periods':0, 'center':True}
kargs = {'periods':2}
tmp1, tmp2 = FM.rolling(7, 'mean')



#
#
# h5_in = '/Users/phil/Desktop/AH0407_160613_JC1003_AAAC_3lag.h5'
# feature_data = image_tools.get_h5_key_and_concatenate(h5_in, 'images')
# feature_data = pd.DataFrame(feature_data)
#
#
# frame_nums = image_tools.get_h5_key_and_concatenate(h5_in, 'frame_nums')
# frame_class = np.zeros(np.sum(frame_nums)).astype(int)
# for i, (i1, i2) in enumerate(utils.loop_segments(frame_nums)):
#     frame_class[i1:i2] = i
# frame_class
#
#
# feature_data['frame_nums'] =frame_class
#
#
# grouped = feature_data.groupby('frame_nums')
#
# tmp1 = grouped.rolling(window = 3, min_periods = 3, center=True).mean()
#
#
# FM = feature_maker(h5_in)
# # tmp1, tmp2 = FM.rolling(10, 'std')
#
# tmp1, tmp2 = FM.shift(2)
#
#
#
# tmp1 = pd.DataFrame(np.random.rand(10, 5))
# kargs = {'min_periods':3, 'center':True}
# kargs = {}
# # kargs = (min_periods=3, center=True)
# # kargs = None
# df_rolling = tmp1.rolling(window = 3, **kargs)
#
# # df_rolling = tmp1.rolling(window=3, min_periods=3, center=True)
# df_rolling.mean().shift(0).shift(0).astype(np.float32)#
#
#
#
# if add_name_str is None:
#     add_name_str = 'shifted_'+str(shift_by)
#     add_name_str = 'feature_data_' + add_name_str
#
# key_exists = utils.h5_key_exists(h5_in, add_name_str)
#
#
#
# assert key_exists, "key name '"+ add_name_str + "' exists, if you want to overwrite it, delete it and re run this function"
# with h5py.File(h5_in, 'r+') as h:# can always change to save on each iteration to save ram but RAM should be fine
#     data_frame_in = pd.DataFrame(h['images'][:])
#     data_frame_out = shift_with_frame_nums(data_frame_in, h['frame_nums'], shift_by, add_name_str = add_name_str)
#     h.create_dataset_like(add_name_str, h['images'], data = data_frame_out.to_numpy())
#
#
#
# def shift_with_frame_nums(h5_in, shift_by, frame_nums=None, add_name_str=None, data_frame_in=None):
#     with h5py.File(h5_in, 'r+') as h:
#         if data_frame_in is None:
#             data_frame_in = h['images'][:]
#         if frame_nums is None:# this allows user to insert their own frame nums if needed
#             frame_nums = h['frame_nums'][:]
#         if add_name_str is None:
#             add_name_str = 'shifted_' + str(shift_by)
#         data_frame = data_frame_in.copy()
#         for i1, i2 in utils.loop_segments(frame_nums):
#             data_frame[i1:i2] = data_frame_in[i1:i2].shift(shift_by)
#         data_frame = data_frame.add_suffix('_' + add_name_str)
#     # return data_frame
#
#
# # def change_to_local_dir(bd):
# #     bd.split('Drive')[]
# #     bd2 = ''.join(bd.split('gdrive/My Drive'))
# #     return bd2
#
# tmp1 = Path(base_dir_all)
# tmp1 / 'asdf.txt'
#
# time_length = 1  # do not change we shift time in pandas dataframe now
# feature_len = 2048
# batch_size = 1000
# tmp_h5 = ''
# label_key = '[0, 1]- (no touch, touch)'
# G_test = image_tools.ImageBatchGenerator_feature_array(time_length, batch_size, tmp_h5,
#                                                        label_key=label_key, feature_len=feature_len)
#
# ####
# # In[35]:
#
#
# label_key = '[0, 1]- (no touch, touch)'
# label_key = 'labels'
# label_key in utils.print_h5_keys(tmp1, return_list=True, do_print=False)
#
# # In[79]:
#
#
# base_dir_all = r'G:\My Drive\colab_data2\model_testing_features_data\feature_data\DATA_FULL'
# base_dir_all = r'G:\My Drive\Colab data\testing_TL_10_20_30\data_ANM234232_140120_AH1030_AAAA_a\3lag\ANM234232_140120_AH1030_AAAA_a_3lag___30'
# all_h5s = utils.get_h5s(base_dir_all)
# for h5 in all_h5s:
#     base_dir = os.path.dirname(h5) + os.sep + 'transformed_data'
#     df_tmp_x, tmp_y = make_data_chunks(h5, base_dir)
#
# # In[57]:
#
# utils.np_stats(y_pred)
# all_h5s[-3], h5
#
# # In[61]:
#
#
# # righthere
# alt_labels_h5s = r'G:\My Drive\colab_data2\model_testing\all_data\final_predictions\ALT_LABELS_FINAL_PRED'
# base_dir_all = r'G:\My Drive\colab_data2\model_testing_features_data\feature_data\DATA_FULL'
#
# model_save_name_full = 'MODEL_FULL_TRAIN_rollSTD_3_7_rollMEAN_3_7_shift_neg2_to_2_val_with_real_val_PM_JC'
# # need to finish evaling above model so that num mod in h5 are equal
# # model_save_name_full = 'MODEL_FULL_TRAIN_rollSTD_3_7_rollMEAN_3_7_shift_neg2_to_2_val_with_real_val_ANDREW'
#
# model_save_name_full = 'model_5____' + '3lag__' + 'regular_labels__' + model_save_name_full + '_LIGHT_GBM'
# model_save_name_full
# all_h5s = utils.get_h5s(base_dir_all)
# alt_labels_h5s = utils.get_h5s(alt_labels_h5s)
# for h5, alt_labels_h5 in tqdm(zip(all_h5s, alt_labels_h5s)):
#     bn1 = os.path.basename(h5)
#     bn2 = os.path.basename(alt_labels_h5)
#     assert bn1[:16] == bn2[:16], 'files dont match'
#     base_dir = os.path.dirname(h5) + os.sep + 'transformed_data'
#     df_tmp_x, tmp_y = make_data_chunks(h5, base_dir)
#     df_tmp_x = concat_numpy_memmory_save(df_tmp_x, base_dir)
#     pred = bst.predict(df_tmp_x)
#     utils.np_stats(pred)
#
#     # is pred an int or a float in the first one it is an int64 which is weird
#     with h5py.File(alt_labels_h5, 'r+') as hf:
#         try:
#             hf.create_dataset(model_save_name_full, data=pred)
#         except:
#             del hf[model_save_name_full]
#             time.sleep(.1)
#             hf.create_dataset(model_save_name_full, data=pred)
#
# # In[32]:
#
#
# tmp1 = r'G:\My Drive\Colab data\testing_TL_10_20_30\data_ANM234232_140120_AH1030_AAAA_a\3lag\ANM234232_140120_AH1030_AAAA_a_3lag___30\train.h5'
# utils.print_h5_keys(tmp1)
# L = image_tools.get_h5_key_and_concatenate(tmp1, 'labels')
#
# img = image_tools.get_h5_key_and_concatenate(tmp1, 'images')
# utils.np_stats(L)
# img.shape, L.shape
#
# # In[41]:
#
#
# h5 = [r'D:\model_testing_features_data\feature_data\regular_80_border\ALL_RETRAIN_H5_data\3lag\train_3lag.h5',
#       r'G:\My Drive\colab_data2\model_testing_features_data\feature_data\testing_TL_10_20_30\data_ANM234232_140120_AH1030_AAAA_a\3lag\ANM234232_140120_AH1030_AAAA_a_3lag___30\train.h5']
#
# #       r'D:\model_testing_features_data\feature_data\DATA_FULL_in_range_only\data_AH0667_170317_JC1241_AAAA\3lag\AH0667_170317_JC1241_AAAA_3lag.h5']
# base_dir = r'D:\temp_save\train_x_arrays_with_A'
# df_train_x, train_y = make_data_chunks(h5, base_dir)
#
# h5 = [r'D:\model_testing_features_data\feature_data\regular_80_border\ALL_RETRAIN_H5_data\3lag\val_3lag.h5',
#       r'G:\My Drive\colab_data2\model_testing_features_data\feature_data\testing_TL_10_20_30\data_ANM234232_140120_AH1030_AAAA_a\3lag\ANM234232_140120_AH1030_AAAA_a_3lag___30\val.h5']
#
# #       r'D:\model_testing_features_data\feature_data\DATA_FULL_in_range_only\data_AH0698_170601_PM0121_AAAA\3lag\AH0698_170601_PM0121_AAAA_3lag.h5']
# base_dir = r'D:\temp_save\val_x_arrays_with_A'
# df_val_x, val_y = make_data_chunks(h5, base_dir)
#
# h5 = r'D:\model_testing_features_data\feature_data\regular_80_border\DATA_FULL\3lag\holy_test_set_10_percent_3lag.h5'
# base_dir = r'D:\temp_save\test_x_arrays'
# df_test_x, test_y = make_data_chunks(h5, base_dir)
#
# # In[34]:
#
#
# h5 = r'D:\model_testing_features_data\feature_data\8_testing_h5s\data_ANM234232_140118_AH1026_AAAA\3lag\ANM234232_140118_AH1026_AAAA_3lag.h5'
# base_dir = r'D:\temp_save\andrew_x_arrays'
# df_andrew_x, andrew_y = make_data_chunks(h5, base_dir)
#
# # In[4]:
#
#
# # utils.print_h5_keys(h5)
# # frame_nums_andrew = image_tools.get_h5_key_and_concatenate(h5, 'frame_nums')
# # frame_nums_andrew
#
#
# # ### load the data from file list
# #
#
# # In[42]:
#
#
# base_dir = r'D:\temp_save\train_x_arrays_with_A'
# df_train_x = concat_numpy_memmory_save(df_train_x, base_dir)
#
# base_dir = r'D:\temp_save\val_x_arrays_with_A'
# df_val_x = concat_numpy_memmory_save(df_val_x, base_dir)
#
# base_dir = r'D:\temp_save\test_x_arrays'
# df_test_x = concat_numpy_memmory_save(df_test_x, base_dir)
#
# # In[35]:
#
#
# base_dir = r'D:\temp_save\andrew_x_arrays'
# df_andrew_x = concat_numpy_memmory_save(df_andrew_x, base_dir)
#
#
# # In[43]:
#
#
# def rm_nan_rows(a, a2=None):
#     indx = ~np.isnan(a).any(axis=1)
#     if a2 is None:
#         return a[indx, :]
#     else:
#         return a[indx, :], a2[indx]
#
#
# # In[44]:
#
#
# df_train_x, train_y = rm_nan_rows(df_train_x, train_y)
# df_test_x, test_y, = rm_nan_rows(df_test_x, test_y)
# df_val_x, val_y = rm_nan_rows(df_val_x, val_y)
#
# # In[32]:
#
#
# df_andrew_x, andrew_y = rm_nan_rows(df_andrew_x, andrew_y)
#
# # #set up light GBM model
#
# # ###format data to lgb format
#
# # In[45]:
#
#
# train_data = lgb.Dataset(df_train_x, label=train_y)
# test_data = lgb.Dataset(df_test_x, label=test_y)
# validation_data = lgb.Dataset(df_val_x, label=val_y)
#
# # ###define params and metrics
#
# # In[4]:
#
#
# # param = {'objective': 'binary'}
# # param['metric'] = ['auc', 'binary_logloss']
# # param['device'] ='cpu'
# # param['max_bin'] ='63'
# # param['num_leaves'] =255
# # param['n_estimators'] = 400
# # param['early_stopping_rounds'] = 50
# # param['verbose'] = 1
#
# # param['num_round'] = 5000
#
#
# # In[46]:
#
#
# param = {'num_leaves': 255, 'objective': 'binary'}
# # param['metric'] = 'auc'
# param['metric'] = ['auc']
# param['early_stopping_rounds'] = 20
# param['device'] = 'cpu'
# param['verbose'] = 1
# param['n_estimators'] = 400
# param['num_round'] = 160
#
# # ###train
# #
#
# # In[47]:
#
#
# bst = lgb.LGBMClassifier(**param)
# # bst.get_params()
#
#
# # In[48]:
#
#
# st = time.time()
# bst.fit(df_train_x, train_y, eval_metric=['auc'], eval_set=[(df_val_x, val_y), (df_test_x, test_y)])
# str(datetime.timedelta(seconds=time.time() - st))
#
# # In[21]:
#
#
# # bst.save_model(r'D:\models\MODEL_FULL_TRAIN_rollSTD_3_7_rollMEAN_3_7_shift_neg2_to_2_val_with_real_val_PM_JC.txt')
#
#
# # In[49]:
#
#
# bst.booster_.save_model(
#     r'D:\models\MODEL_FULL_TRAIN_rollSTD_3_7_rollMEAN_3_7_shift_neg2_to_2_val_with_real_val_ANDREW.txt')
#
# # In[22]:
#
#
# utils.get_class_info(bst)
#
# # In[ ]:
#
#
# #
# bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data], callbacks=callbacks)
#
# bst = lgb.LGBMClassifier(n_jobs=-1, n_estimators=400, **params)
#
# bst.fit(X_train, y_train, eval_metric='auc', eval_set=[(X_test, y_test)], early_stopping_rounds=50,
#         categorical_feature=categorical_features, verbose=5)
#
# # In[ ]:
#
#
# num_round = 2000
# bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
#
# # In[ ]:
#
#
# # num_round = 5000
#
# callbacks = [lgb.early_stopping(50, first_metric_only=False, verbose=True, min_delta=0.0)]
#
# # In[ ]:
#
#
# # num_round = 3000
# # bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
#
#
# # In[ ]:
#
#
# # visualize
# import matplotlib.pyplot as plt
# import matplotlib.style as style
# import seaborn as sns
# from matplotlib import pyplot
# from matplotlib.ticker import ScalarFormatter
#
# sns.set_context("talk")
# style.use('fivethirtyeight')
#
# fi = pd.DataFrame()
# fi['features'] = X.columns.values.tolist()
# fi['importance'] = model.feature_importance(importance_type="gain")
#
# sns.barplot(x='importance', y='features', data=fi.sort_values(by='importance', ascending=True))
#
# # In[ ]:
#
#
# bst.save_model('/content/gdrive/My Drive/colab_data2/model_FULL_TRAIN_rollSTD_3_7_rollMEAN_3_7_shift_neg2_to_2.txt')  #
#
# # In[ ]:
#
#
# # utils.get_class_info(bst.params)
# bst.params
#
#
# # In[16]:
#
#
# def smooth_it(data, kernel_size=7):
#     kernel = np.ones(kernel_size) / kernel_size
#     return np.convolve(data, kernel, mode='same')
#
#
# # In[33]:
#
#
# yhat = bst.predict(df_andrew_x)
# np.mean(1 * (smooth_it(yhat, 7) > .5) == andrew_y)
#
# # In[17]:
#
#
# yhat = bst.predict(df_val_x)
# np.mean(1 * (smooth_it(yhat, 7) > .5) == val_y)
#
# # In[18]:
#
#
# yhat = bst.predict(df_train_x)
# np.mean(1 * (smooth_it(yhat, 7) > .5) == train_y)
#
# # In[19]:
#
#
# yhat = bst.predict(df_test_x)
# np.mean(1 * (smooth_it(yhat, 7) > .5) == test_y)
#
# # In[ ]:
#
#
# test_y
#
# # In[102]:
#
#
# utils.np_stats(yhat)
#
# # In[61]:
#
#
# plt.figure(figsize=(10, 10))
# real_y = andrew_y
#
# i1 += 4000
# # i1 = 0
# i2 = i1 + 4000
# yhat2 = 1 * (yhat > 0.5)
#
# # plt.plot(yhat[i1:i2])
# tmpy = smooth_it(yhat[i1:i2], kernel_size=3)
# plt.plot(tmpy, '--')
# tmpy = 1 * (tmpy > 0.5)
# plt.plot(tmpy)
# plt.plot(andrew_y[i1:i2] - 1.1)
#
# plt.plot((andrew_y[i1:i2] - yhat2[i1:i2] - 1.1) / 2, '.')
# plt.title(str(i1) + ' to ' + str(i2))
#
# # In[71]:
#
#
# plt.figure(figsize=(20, 10))
# frame_segs = utils.loop_segments(frame_nums_andrew, returnaslist=True)
#
# # for i1, i2 in utils.loop_segments(frame_nums_andrew):
#
# # cnt = -1
# cnt += 1
# i1, i2 = frame_segs[0][cnt], frame_segs[1][cnt]
#
# yhat2 = 1 * (yhat > 0.5)
#
# # plt.plot(yhat[i1:i2])
# tmpy = smooth_it(yhat[i1:i2], kernel_size=3)
# plt.plot(tmpy, '--')
# tmpy = 1 * (tmpy > 0.5)
# plt.plot(tmpy)
# plt.plot(andrew_y[i1:i2] - 1.1)
#
# plt.plot((andrew_y[i1:i2] - yhat2[i1:i2] - 1.1) / 2, '.')
# plt.title(str(i1) + ' to ' + str(i2))
#
# # In[42]:
#
#
# frame_segs[0][cnt], frame_segs[1][cnt]
#
# # #load model
#
# # In[60]:
#
#
# get_ipython().run_cell_magic('time', '',
#                              "model_file = r'D:\\models\\MODEL_FULL_TRAIN_rollSTD_3_7_rollMEAN_3_7_shift_neg2_to_2_val_with_real_val_ANDREW.txt'\nmodel_file = r'D:\\models\\MODEL_FULL_TRAIN_rollSTD_3_7_rollMEAN_3_7_shift_neg2_to_2_val_with_real_val_PM_JC.txt'\n\n# bst = lgb.load(filename = model_file)\n\nbst = lgb.Booster(model_file=model_file)")
#
# # In[13]:
#
#
# utils.get_class_info(bst)
#
# # In[17]:
#
#
# get_ipython().run_cell_magic('time', '',
#                              'x = df_test_x[:4000, :]\ny_pred = bst.predict(x)\nutils.np_stats(y_pred)\nplt.plot(y_pred)')
#
# # In[12]:
#
#
# get_ipython().run_cell_magic('time', '', 'x = df_test_x[:10, :]\ny_pred = bst.predict_proba(x)')
#
# # #make and save predictions
#
# # In[ ]:
#
#
# h5src = '/content/gdrive/My Drive/colab_data2/model_testing_features_data/feature_data/8_testing_h5s/'
#
# h5 = utils.get_h5s(h5src)
#
# time_length = 1
# batch_size = None
# label_key = '[0, 1]- (no touch, touch)'
# feature_len = 2048
# label_index_to_lstm_len = None
# edge_value = np.nan
# remove_any_time_points_with_edges = True
#
# G_test = image_tools.ImageBatchGenerator_feature_array(time_length, batch_size, h5, label_key=label_key,
#                                                        feature_len=feature_len,
#                                                        label_index_to_lstm_len=label_index_to_lstm_len,
#                                                        edge_value=edge_value,
#                                                        remove_any_time_points_with_edges=remove_any_time_points_with_edges)
# frame_nums_test = image_tools.get_h5_key_and_concatenate(h5, 'frame_nums')
#
# # In[ ]:
#
#
# test_x, test_y = G_test.__getitem__(0)
#
# # In[ ]:
#
#
# df_test_x = pd.DataFrame(test_x)
# del test_x  # save RAM
#
# # In[ ]:
#
#
# df_test_x = transform_x_data(df_test_x, frame_nums_test)
#
# # In[ ]:
#
#
# with open('df_test_x_2.pkl', 'wb') as f:
#     pickle.dump(df_test_x, f)
#
# # In[ ]:
#
#
# with open('df_test_x_2.pkl', 'rb') as f:
#     df_test_x = pickle.load(f)
#
# # In[ ]:
#
#
# # df_test_x = df_test_x.to_numpy()
#
#
# # In[ ]:
#
#
# yhat = bst.predict(df_test_x[:1000])
#
# # In[ ]:
#
#
# df_test_x[:1000].shape
#
# # In[ ]:
#
#
# from whacc import utils, image_tools, transfer_learning, analysis, error_analysis
#
# # #_____
#
# # In[ ]:
#
#
# # In[21]:
#
#
# get_ipython().system('pip install xgboost -U')
#
# import xgboost
#
# print("xgboost", xgboost.__version__)
# from xgboost import XGBClassifier
#
# # utils.get_class_info(xgboost)
#
#
# # In[ ]:
#
#
# clf = xgb.XGBClassifier(use_label_encoder=False, max_depth=200, n_estimators=400, subsample=1, learning_rate=0.09,
#                         reg_lambda=0.1, reg_alpha=0.1, gamma=1)
#
# # In[ ]:
#
#
# param = dict()
# param['gpu_id'] = 0
# param['tree_method'] = 'gpu_hist'
#
# param = {
#     'objective': 'binary:logistic',
#     'tree_method': 'binary:logistic',
#     'gpu_id': 0,
#     'use_label_encoder': False
# }
# model = XGBClassifier(**param)
# # train_data = lgb.Dataset(df_train_x, label=train_y)
# # validation_data = lgb.Dataset(df_test_x, label=test_y)
# model.fit(df_train_x, train_y)
#
# # In[ ]:
#
#
# grid = {'max_depth': 10}
# clf = XGBClassifier()
# clf.max_depth
# clf.set_params(**grid)
# XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
#               gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,
#               min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
#               objective='binary:logistic', reg_alpha=0, reg_lambda=1,
#               scale_pos_weight=1, seed=0, silent=True, subsample=1)
# clf.max_depth
#
# # In[ ]:
#
#
# xgbmodel = XGBClassifier(objective='binary:logistic', use_label_encoder=False)
#
# # In[ ]:
#
#
# x, y = G_train.__getitem__(0)
#
# xgbmodel.fit(x, y)
#
# xgbmodel.get_params()
#
# # In[ ]:
#
#
# time_length = 7
# batch_size = int(92213 / 10)
# print(batch_size)
# label_key = '[0, 1]- (no touch, touch)'
# feature_len = 2048
# label_index_to_lstm_len = 3
# edge_value = np.nanf
# remove_any_time_points_with_edges = True
#
# h5 = '/content/data/ALL_RETRAIN_H5_data____temp/ALL_RETRAIN_H5_data/3lag/train_3lag.h5'
# G_train = image_tools.ImageBatchGenerator_feature_array(time_length, batch_size, h5, label_key=label_key,
#                                                         feature_len=feature_len,
#                                                         label_index_to_lstm_len=label_index_to_lstm_len,
#                                                         edge_value=edge_value,
#                                                         remove_any_time_points_with_edges=remove_any_time_points_with_edges)
#
# # In[ ]:
#
#
# model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, par)
#
# model.fit(x, y)
#
# # #_____
#
# # ##build model
#
# # In[ ]:
#
#
# model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, n_estimators=500, subsample=.9,
#                       max_depth=10)
#
# # In[ ]:
#
#
# model.get_params()
#
# # In[ ]:
#
#
# del model
#
# # In[ ]:
#
#
# # ##fit model
#
# # In[ ]:
#
#
# import time
#
# # In[ ]:
#
#
# st = time.time()
# x, y = G_train.__getitem__(0)
# model.fit(x, y)
# print(time.time() - st)
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# train_x.size  # about .5 gb on ram
#
# # In[ ]:
#
#
# xgbPredictor = xgboost.XGBRegressor(**self.xgb_params)
# xgbPredictor.fit(Xs, ys)
#
# # This hack should only be used if tree_method == gpu_hist or gpu_exact
# if self.xgb_params['tree_method'][:3] == 'gpu':
#     with tempfile.TemporaryFile() as dump_file:
#         pickle.dump(xgbPredictor, dump_file)
#         dump_file.seek(0)
#         self.predictor_ = pickle.load(dump_file)
# else:
#     self.predictor_ = xgbPredictor
#
# # #full batch all at once
#
# # In[ ]:
#
#
# # time_length = 7
# # batch_size = int(92213/1)
# # print(batch_size)
# # label_key='[0, 1]- (no touch, touch)'
# # feature_len=2048
# # label_index_to_lstm_len=3
# # edge_value= -1
# # remove_any_time_points_with_edges=True
#
# # h5 = '/content/data/ALL_RETRAIN_H5_data____temp/ALL_RETRAIN_H5_data/3lag/train_3lag.h5'
# # G_train = image_tools.ImageBatchGenerator_feature_array(time_length, batch_size, h5, label_key=label_key, feature_len=feature_len,
# #                                       label_index_to_lstm_len=label_index_to_lstm_len, edge_value=edge_value,
# #                                       remove_any_time_points_with_edges = remove_any_time_points_with_edges)
#
#
# # In[ ]:
#
#
# parameters = {'base_score': 0.23,  # i think this should be set to mean of the labels it is the bias value
#               'booster': 'gbtree',
#               'colsample_bylevel': 1,
#               # "colsample_bylevel" is the fraction of features (randomly selected) that will be used in each node to train each tree.
#               'colsample_bynode': 1,
#               'colsample_bytree': 1,
#               # "colsample_bytree" is the fraction of features (randomly selected) that will be used to train each tree.
#               'enable_categorical': False,
#               'gamma': 0,
#               'gpu_id': -1,
#               'importance_type': None,
#               'interaction_constraints': '',
#               'learning_rate': [.2, .1, .05, .01],
#               'max_delta_step': 0,
#               'max_depth': [8, 12],
#               'min_child_weight': 1,
#               'monotone_constraints': '()',
#               'n_estimators': [100, 250, 500],
#               'n_jobs': -1,  # use all core I think 8
#               'num_parallel_tree': 1,
#               'objective': 'binary:logistic',
#               'predictor': 'auto',
#               'random_state': 0,
#               'reg_alpha': 0,
#               'reg_lambda': 1,
#               'scale_pos_weight': 1,
#               'single_precision_histogram': True,  # supposed to make it faster as well
#               'subsample': 0.9,
#               # "subsample" is the fraction of the training samples (randomly selected) that will be used to train each tree.
#               #  this seems like this should not be 1 it should be lower to work better ......
#               'tree_method': 'hist',  # was 'exact' but 'hist' is supposed ot be faster
#               'use_label_encoder': False,
#               'validate_parameters': 1,
#               'verbosity': 1}
# parameters = {
#     'learning_rate': [.2, .1, .05, .01],
#     'max_depth': [8, 12],
#     'n_estimators': [100, 250, 500]}
# for k in parameters:
#     parameters[k] = utils.make_list(parameters[k], suppress_warning=True)
# # parameters
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # !lscpu
#
#
# # In[ ]:
#
#
# # # model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, single_precision_histogram=True)
# # from sklearn.model_selection import GridSearchCV
# # # parameters = {
# # #     'max_depth': range (2, 10, 1),
# # #     'n_estimators': range(60, 220, 40),
# # #     'learning_rate': [0.1, 0.01, 0.05]}
#
# # model = XGBClassifier(objective= 'binary:logistic')
#
# # grid_search = GridSearchCV(
# #   estimator=model,
# #   param_grid=parameters,
# #   scoring = 'roc_auc',
# #   n_jobs = 10,
# #   cv = 10,
# #   verbose=True)
#
# # grid_search.fit(x[:20], y[:20])
#
#
# # In[ ]:
#
#
# from sklearn.model_selection import GridSearchCV
#
# model = XGBClassifier(objective='binary:logistic')
# kfold = 5
# gs = GridSearchCV(estimator=model,
#                   param_grid=parameters,
#                   scoring='roc_auc',
#                   n_jobs=-1,
#                   cv=kfold,
#                   verbose=3,
#                   return_train_score=True)
#
# # In[ ]:
#
#
# train_x, train_y
# test_x, test_y
#
# # In[ ]:
#
#
# # grid_result.best_estimator_.model.model.history.history
# from sklearn import preprocessing
#
# train_x = preprocessing.scale(train_x)
# test_x = preprocessing.scale(test_x)
# # train_x = preprocessing.scale(train_x)
# # train_x = preprocessing.scale(train_x)
#
#
# # In[ ]:
#
#
# train_x[:10, :5]
#
# # In[ ]:
#
#
# grid_result = gs.fit(train_x[:10, :5],
#                      train_y[:10],
#                      verbose=3)
#
# # In[ ]:
#
#
# grid_result = gs.fit([train_x[:10, :5]],
#                      train_y[:10],
#                      verbose=3)
#
# # In[ ]:
#
#
# # grid_result = gs.fit(train_x[:100, :5],
# #                     train_y[:100],
# #                     verbose=1,
# #                     validation_data=(test_x[:100, 5], test_y[:100]))
#
#
# # In[ ]:
#
#
# import joblib
#
# try:
#     joblib.dump(gs, '/content/gdrive/My Drive/XGBoost_gris_search/XGBoost_grid_search_001.pkl')
# except:
#     joblib.dump(grid_result, '/content/gdrive/My Drive/XGBoost_gris_search/XGBoost_grid_search_001.pkl')
# joblib.dump(grid_result, '/content/gdrive/My Drive/XGBoost_gris_search/XGBoost_grid_search_001.pkl')
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# clf = XGBClassifier()
# clf.max_depth
# clf.set_params(**grid)
#
# clf.get_params()
# history = clf.fit(train_x[:10, :5], train_y[:10],
#
#                   # In[ ]:
#
#                   grid={'max_depth': 10}
# clf = XGBClassifier()
# clf.max_depth
# clf.set_params(**grid)
# XGBClassifier(base_score=0.5,
#               colsample_bylevel=1,
#               colsample_bytree=1,
#               gamma=0,
#               learning_rate=0.1,
#               max_delta_step=0,
#               max_depth=10,
#               min_child_weight=1,
#               missing=None,
#               n_estimators=500,
#               nthread=-1,
#               objective='binary:logistic',
#               reg_alpha=0,
#               reg_lambda=1,
#               scale_pos_weight=1,
#               seed=0,
#               silent=True,
#               subsample=1)
# clf.n_estimators
# # clf.fit(x[:20], y[:20])
# history = clf.fit(train_x[:10, :5], train_y[:10])
#
# # In[ ]:
#
#
# grid = {'max_depth': 10}
# clf = XGBClassifier()
# clf.max_depth
# clf.set_params(**grid)
# XGBClassifier(base_score=0.5,
#               colsample_bylevel=1,
#               colsample_bytree=1,
#               gamma=0,
#               learning_rate=0.1,
#               max_delta_step=0,
#               max_depth=10,
#               min_child_weight=1,
#               missing=None,
#               n_estimators=500,
#               nthread=-1,
#               objective='binary:logistic',
#               reg_alpha=0,
#               reg_lambda=1,
#               scale_pos_weight=1,
#               seed=0,
#               silent=True,
#               subsample=1)
# clf.n_estimators
# # clf.fit(x[:20], y[:20])
# history = clf.fit(train_x[:10, :5], train_y[:10])
#
# # In[ ]:
#
#
# import time
#
# st = time.time()
#
# model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, single_precision_histogram=True,
#                       n_estimators=500, subsample=.9, max_depth=10)
# x, y = G_train.__getitem__(0)
# model.fit(x, y)
#
# print(time.time() - st)
#
# # In[ ]:
#
#
# import time
#
# st = time.time()
#
# model_small_test = XGBClassifier(objective='binary:logistic', use_label_encoder=False, single_precision_histogram=True,
#                                  n_estimators=500, subsample=.9, max_depth=10)
# x, y = G_train.__getitem__(0)
# model_small_test.fit(x[:10, :5], y[:10])
#
# print(time.time() - st)
#
# # In[ ]:
#
#
# x[:10, :10]
#
# # In[ ]:
#
#
# from sklearn import preprocessing
#
# x = preprocessing.scale(x)
#
# # In[ ]:
#
#
# num_data_points = 10000
# model_small_test = XGBClassifier(objective='binary:logistic', use_label_encoder=False, single_precision_histogram=True,
#                                  n_estimators=500, subsample=.9, max_depth=10)
# st = time.time()
# model_small_test.fit(x[:num_data_points, :], y[:num_data_points])
# print(time.time() - st)
#
# # In[ ]:
#
#
# """
# before scaling all 2048 features
# 100  --> 5.22572660446167
# 1000 --> 11.855197429656982
# 10000--> 55.457772970199585
#
# after scaling all 2048 features
# 100  --> 3.963913917541504
# 1000 --> 9.200135469436646
# 10000--> 48.088958740234375
# """
#
# # In[ ]:
#
#
# xtmp = [100, 1000, 10000]
# plt.plot(xtmp, [5.22572660446167,
#                 11.855197429656982,
#                 55.457772970199585])
# plt.plot(xtmp, [3.963913917541504,
#                 9.200135469436646,
#                 48.088958740234375])
#
# # In[ ]:
#
#
# import pickle
#
# file_name = "/content/gdrive/My Drive/XGBoost_gris_search/xgb_reg.pkl"
#
# # save
# pickle.dump(model, open(file_name, "wb"))
#
# # In[ ]:
#
#
# model = pickle.load(open(file_name, "rb"))
#
# # In[ ]:
#
#
# from xgboost import plot_importance, plot_tree
#
# # In[ ]:
#
#
# # plt.plot(x[:, 163], x[:, 171], '.')
# tmpx = x[:10, :5]
# R1 = np.corrcoef(x)
# R1
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# plt.rcParams["figure.figsize"] = (30, 30)
# _ = plot_importance(model, height=0.9, max_num_features=200)
#
# # In[ ]:
#
#
# from IPython.core.pylabtools import figsize
#
# plot_importance(figsize=(10, 10))
#
# # In[ ]:
#
#
# real = np.asarray([])
# pred = np.asarray([])
# for kk in tqdm(range(G_val.__len__())):
#     x, y = G_val.__getitem__(kk)
# real = np.append(real, y)
# y_hat = model.predict(x)
# pred = np.append(pred, y_hat)
# np.mean(pred == real)
#
# # In[ ]:
#
#
# model
#
# # In[ ]:
#
#
# real = np.asarray([])
# pred = np.asarray([])
# for kk in tqdm(range(G_test.__len__())):
#     x, y = G_test.__getitem__(kk)
# real = np.append(real, y)
# y_hat = model.predict(x)
# pred = np.append(pred, y_hat)
# np.mean(pred == real)
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# real = np.asarray([])
# pred = np.asarray([])
# for kk in tqdm(range(G_train.__len__())):
#     x, y = G_train.__getitem__(kk)
# real = np.append(real, y)
# y_hat = model.predict(x)
# pred = np.append(pred, y_hat)
# np.mean(pred == real)
#
# # #__
# #
#
# #
#
# # In[ ]:
#
#
# utils.get_class_info(xgbmodel)
#
# # In[ ]:
#
#
# from xgboost import XGBClassifier
#
# xgbmodel = XGBClassifier(objective='binary:logistic')
#
# for k in range(G_train.__len__()):
#     x, y = G_train.__getitem__(k)
# xgbmodel.fit(x, y)
# for kk in range(G_val.__len__()):
#     x, y = G_val.__getitem__(kk)
# xgbmodel.score(x, y)
#
# # In[ ]:
#
#
# send_text('OOOOHHHHH SHIIIITTTTT ITS DOONNNEEEEE')
#
# # In[ ]:
#
#
# from xgboost import XGBClassifier
#
# utils.get_class_info(XGBClassifier)
# XGBClassifier()
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # #plot the features sorted to get an idea
#
# # In[ ]:
#
#
# h5 = '/content/data/ALL_RETRAIN_H5_data____temp/DATA_FULL/3lag/holy_test_set_10_percent_3lag.h5'
# # h5 = '/content/data/ALL_RETRAIN_H5_data____temp/ALL_RETRAIN_H5_data/3lag/train_3lag.h5'
# a = image_tools.ImageBatchGenerator_feature_array(7, 4000, h5, label_key='[0, 1]- (no touch, touch)', feature_len=2048,
#                                                   label_index_to_lstm_len=3, edge_value=-1,
#                                                   remove_any_time_points_with_edges=True)
# a.__len__()
#
# # In[ ]:
#
#
# x, y = a.__getitem__(9)
#
# # In[ ]:
#
#
# x.shape
#
# # In[ ]:
#
#
# # add_to = -2000
# add_to += 100
# plt.figure(figsize=(5, 700))
# tmp1 = int(2048 / 4)
# tmp1 = 2048 * 3
# # plt.imshow(x[2200+add_to:2350+add_to, ::tmp1].T)
# x2 = x[2200 + add_to:2350 + add_to, tmp1:tmp1 + 2000].T
# # sort_inds = np.argsort(np.mean(x2[:, 20:28], axis = 1))
# x3 = np.zeros_like(x2)
# for i, k in enumerate(sort_inds):
#     x3[i, :] = x2[k, :]
#
# y2 = y[2200 + add_to:2350 + add_to]
# y2 = y2 * np.max(x3)
# y2 = np.tile(y2, (10, 1))
# x3 = np.vstack((y2, x3, y2))
# plt.imshow(x3)
# # plt.axis('off')
# # plt.axis('tight')
# plt.show()
# # plt.figure(figsize=(5, 5))
# # time.sleep(3)
#
#
# plt.plot(y[2200 + add_to:2350 + add_to])
# plt.axis('off')
# plt.axis('tight')
# plt.ylim([-.1, 1.1])
# nums')
#
# grouped.rolling(10).mean()
#
#
# FM = feature_maker(h5_in)
# # tmp1, tmp2 = FM.rolling(10, 'std')
#
# tmp1, tmp2 = FM.shift(2)
#
#
#
#
#
#
# tmp1 = pd.DataFrame.rolling(window = 10)
#
#
#
#
# """
# pass function (or fucntion string)
# have frame nums and data self contained
# pass in k args
#
#
#
# """
#
#
# kargs = {'min_periods':3, 'center':True}
# # kargs = (min_periods=3, center=True)
# # kargs = None
# df_rolling = tmp1.rolling(window = 3, **kargs)
#
# # df_rolling = tmp1.rolling(window=3, min_periods=3, center=True)
# df_rolling.mean().shift(0).shift(0).astype(np.float32)#
#
#
#
# if add_name_str is None:
#     add_name_str = 'shifted_'+str(shift_by)
#     add_name_str = 'feature_data_' + add_name_str
#
# key_exists = utils.h5_key_exists(h5_in, add_name_str)
#
#
#
# assert key_exists, "key name '"+ add_name_str + "' exists, if you want to overwrite it, delete it and re run this function"
# with h5py.File(h5_in, 'r+') as h:# can always change to save on each iteration to save ram but RAM should be fine
#     data_frame_in = pd.DataFrame(h['images'][:])
#     data_frame_out = shift_with_frame_nums(data_frame_in, h['frame_nums'], shift_by, add_name_str = add_name_str)
#     h.create_dataset_like(add_name_str, h['images'], data = data_frame_out.to_numpy())
#
#
#
# def shift_with_frame_nums(h5_in, shift_by, frame_nums=None, add_name_str=None, data_frame_in=None):
#     with h5py.File(h5_in, 'r+') as h:
#         if data_frame_in is None:
#             data_frame_in = h['images'][:]
#         if frame_nums is None:# this allows user to insert their own frame nums if needed
#             frame_nums = h['frame_nums'][:]
#         if add_name_str is None:
#             add_name_str = 'shifted_' + str(shift_by)
#         data_frame = data_frame_in.copy()
#         for i1, i2 in utils.loop_segments(frame_nums):
#             data_frame[i1:i2] = data_frame_in[i1:i2].shift(shift_by)
#         data_frame = data_frame.add_suffix('_' + add_name_str)
#     # return data_frame
#
#
# # def change_to_local_dir(bd):
# #     bd.split('Drive')[]
# #     bd2 = ''.join(bd.split('gdrive/My Drive'))
# #     return bd2
#
# tmp1 = Path(base_dir_all)
# tmp1 / 'asdf.txt'
#
# time_length = 1  # do not change we shift time in pandas dataframe now
# feature_len = 2048
# batch_size = 1000
# tmp_h5 = ''
# label_key = '[0, 1]- (no touch, touch)'
# G_test = image_tools.ImageBatchGenerator_feature_array(time_length, batch_size, tmp_h5,
#                                                        label_key=label_key, feature_len=feature_len)
#
# ####
# # In[35]:
#
#
# label_key = '[0, 1]- (no touch, touch)'
# label_key = 'labels'
# label_key in utils.print_h5_keys(tmp1, return_list=True, do_print=False)
#
# # In[79]:
#
#
# base_dir_all = r'G:\My Drive\colab_data2\model_testing_features_data\feature_data\DATA_FULL'
# base_dir_all = r'G:\My Drive\Colab data\testing_TL_10_20_30\data_ANM234232_140120_AH1030_AAAA_a\3lag\ANM234232_140120_AH1030_AAAA_a_3lag___30'
# all_h5s = utils.get_h5s(base_dir_all)
# for h5 in all_h5s:
#     base_dir = os.path.dirname(h5) + os.sep + 'transformed_data'
#     df_tmp_x, tmp_y = make_data_chunks(h5, base_dir)
#
# # In[57]:
#
# utils.np_stats(y_pred)
# all_h5s[-3], h5
#
# # In[61]:
#
#
# # righthere
# alt_labels_h5s = r'G:\My Drive\colab_data2\model_testing\all_data\final_predictions\ALT_LABELS_FINAL_PRED'
# base_dir_all = r'G:\My Drive\colab_data2\model_testing_features_data\feature_data\DATA_FULL'
#
# model_save_name_full = 'MODEL_FULL_TRAIN_rollSTD_3_7_rollMEAN_3_7_shift_neg2_to_2_val_with_real_val_PM_JC'
# # need to finish evaling above model so that num mod in h5 are equal
# # model_save_name_full = 'MODEL_FULL_TRAIN_rollSTD_3_7_rollMEAN_3_7_shift_neg2_to_2_val_with_real_val_ANDREW'
#
# model_save_name_full = 'model_5____' + '3lag__' + 'regular_labels__' + model_save_name_full + '_LIGHT_GBM'
# model_save_name_full
# all_h5s = utils.get_h5s(base_dir_all)
# alt_labels_h5s = utils.get_h5s(alt_labels_h5s)
# for h5, alt_labels_h5 in tqdm(zip(all_h5s, alt_labels_h5s)):
#     bn1 = os.path.basename(h5)
#     bn2 = os.path.basename(alt_labels_h5)
#     assert bn1[:16] == bn2[:16], 'files dont match'
#     base_dir = os.path.dirname(h5) + os.sep + 'transformed_data'
#     df_tmp_x, tmp_y = make_data_chunks(h5, base_dir)
#     df_tmp_x = concat_numpy_memmory_save(df_tmp_x, base_dir)
#     pred = bst.predict(df_tmp_x)
#     utils.np_stats(pred)
#
#     # is pred an int or a float in the first one it is an int64 which is weird
#     with h5py.File(alt_labels_h5, 'r+') as hf:
#         try:
#             hf.create_dataset(model_save_name_full, data=pred)
#         except:
#             del hf[model_save_name_full]
#             time.sleep(.1)
#             hf.create_dataset(model_save_name_full, data=pred)
#
# # In[32]:
#
#
# tmp1 = r'G:\My Drive\Colab data\testing_TL_10_20_30\data_ANM234232_140120_AH1030_AAAA_a\3lag\ANM234232_140120_AH1030_AAAA_a_3lag___30\train.h5'
# utils.print_h5_keys(tmp1)
# L = image_tools.get_h5_key_and_concatenate(tmp1, 'labels')
#
# img = image_tools.get_h5_key_and_concatenate(tmp1, 'images')
# utils.np_stats(L)
# img.shape, L.shape
#
# # In[41]:
#
#
# h5 = [r'D:\model_testing_features_data\feature_data\regular_80_border\ALL_RETRAIN_H5_data\3lag\train_3lag.h5',
#       r'G:\My Drive\colab_data2\model_testing_features_data\feature_data\testing_TL_10_20_30\data_ANM234232_140120_AH1030_AAAA_a\3lag\ANM234232_140120_AH1030_AAAA_a_3lag___30\train.h5']
#
# #       r'D:\model_testing_features_data\feature_data\DATA_FULL_in_range_only\data_AH0667_170317_JC1241_AAAA\3lag\AH0667_170317_JC1241_AAAA_3lag.h5']
# base_dir = r'D:\temp_save\train_x_arrays_with_A'
# df_train_x, train_y = make_data_chunks(h5, base_dir)
#
# h5 = [r'D:\model_testing_features_data\feature_data\regular_80_border\ALL_RETRAIN_H5_data\3lag\val_3lag.h5',
#       r'G:\My Drive\colab_data2\model_testing_features_data\feature_data\testing_TL_10_20_30\data_ANM234232_140120_AH1030_AAAA_a\3lag\ANM234232_140120_AH1030_AAAA_a_3lag___30\val.h5']
#
# #       r'D:\model_testing_features_data\feature_data\DATA_FULL_in_range_only\data_AH0698_170601_PM0121_AAAA\3lag\AH0698_170601_PM0121_AAAA_3lag.h5']
# base_dir = r'D:\temp_save\val_x_arrays_with_A'
# df_val_x, val_y = make_data_chunks(h5, base_dir)
#
# h5 = r'D:\model_testing_features_data\feature_data\regular_80_border\DATA_FULL\3lag\holy_test_set_10_percent_3lag.h5'
# base_dir = r'D:\temp_save\test_x_arrays'
# df_test_x, test_y = make_data_chunks(h5, base_dir)
#
# # In[34]:
#
#
# h5 = r'D:\model_testing_features_data\feature_data\8_testing_h5s\data_ANM234232_140118_AH1026_AAAA\3lag\ANM234232_140118_AH1026_AAAA_3lag.h5'
# base_dir = r'D:\temp_save\andrew_x_arrays'
# df_andrew_x, andrew_y = make_data_chunks(h5, base_dir)
#
# # In[4]:
#
#
# # utils.print_h5_keys(h5)
# # frame_nums_andrew = image_tools.get_h5_key_and_concatenate(h5, 'frame_nums')
# # frame_nums_andrew
#
#
# # ### load the data from file list
# #
#
# # In[42]:
#
#
# base_dir = r'D:\temp_save\train_x_arrays_with_A'
# df_train_x = concat_numpy_memmory_save(df_train_x, base_dir)
#
# base_dir = r'D:\temp_save\val_x_arrays_with_A'
# df_val_x = concat_numpy_memmory_save(df_val_x, base_dir)
#
# base_dir = r'D:\temp_save\test_x_arrays'
# df_test_x = concat_numpy_memmory_save(df_test_x, base_dir)
#
# # In[35]:
#
#
# base_dir = r'D:\temp_save\andrew_x_arrays'
# df_andrew_x = concat_numpy_memmory_save(df_andrew_x, base_dir)
#
#
# # In[43]:
#
#
# def rm_nan_rows(a, a2=None):
#     indx = ~np.isnan(a).any(axis=1)
#     if a2 is None:
#         return a[indx, :]
#     else:
#         return a[indx, :], a2[indx]
#
#
# # In[44]:
#
#
# df_train_x, train_y = rm_nan_rows(df_train_x, train_y)
# df_test_x, test_y, = rm_nan_rows(df_test_x, test_y)
# df_val_x, val_y = rm_nan_rows(df_val_x, val_y)
#
# # In[32]:
#
#
# df_andrew_x, andrew_y = rm_nan_rows(df_andrew_x, andrew_y)
#
# # #set up light GBM model
#
# # ###format data to lgb format
#
# # In[45]:
#
#
# train_data = lgb.Dataset(df_train_x, label=train_y)
# test_data = lgb.Dataset(df_test_x, label=test_y)
# validation_data = lgb.Dataset(df_val_x, label=val_y)
#
# # ###define params and metrics
#
# # In[4]:
#
#
# # param = {'objective': 'binary'}
# # param['metric'] = ['auc', 'binary_logloss']
# # param['device'] ='cpu'
# # param['max_bin'] ='63'
# # param['num_leaves'] =255
# # param['n_estimators'] = 400
# # param['early_stopping_rounds'] = 50
# # param['verbose'] = 1
#
# # param['num_round'] = 5000
#
#
# # In[46]:
#
#
# param = {'num_leaves': 255, 'objective': 'binary'}
# # param['metric'] = 'auc'
# param['metric'] = ['auc']
# param['early_stopping_rounds'] = 20
# param['device'] = 'cpu'
# param['verbose'] = 1
# param['n_estimators'] = 400
# param['num_round'] = 160
#
# # ###train
# #
#
# # In[47]:
#
#
# bst = lgb.LGBMClassifier(**param)
# # bst.get_params()
#
#
# # In[48]:
#
#
# st = time.time()
# bst.fit(df_train_x, train_y, eval_metric=['auc'], eval_set=[(df_val_x, val_y), (df_test_x, test_y)])
# str(datetime.timedelta(seconds=time.time() - st))
#
# # In[21]:
#
#
# # bst.save_model(r'D:\models\MODEL_FULL_TRAIN_rollSTD_3_7_rollMEAN_3_7_shift_neg2_to_2_val_with_real_val_PM_JC.txt')
#
#
# # In[49]:
#
#
# bst.booster_.save_model(
#     r'D:\models\MODEL_FULL_TRAIN_rollSTD_3_7_rollMEAN_3_7_shift_neg2_to_2_val_with_real_val_ANDREW.txt')
#
# # In[22]:
#
#
# utils.get_class_info(bst)
#
# # In[ ]:
#
#
# #
# bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data], callbacks=callbacks)
#
# bst = lgb.LGBMClassifier(n_jobs=-1, n_estimators=400, **params)
#
# bst.fit(X_train, y_train, eval_metric='auc', eval_set=[(X_test, y_test)], early_stopping_rounds=50,
#         categorical_feature=categorical_features, verbose=5)
#
# # In[ ]:
#
#
# num_round = 2000
# bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
#
# # In[ ]:
#
#
# # num_round = 5000
#
# callbacks = [lgb.early_stopping(50, first_metric_only=False, verbose=True, min_delta=0.0)]
#
# # In[ ]:
#
#
# # num_round = 3000
# # bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
#
#
# # In[ ]:
#
#
# # visualize
# import matplotlib.pyplot as plt
# import matplotlib.style as style
# import seaborn as sns
# from matplotlib import pyplot
# from matplotlib.ticker import ScalarFormatter
#
# sns.set_context("talk")
# style.use('fivethirtyeight')
#
# fi = pd.DataFrame()
# fi['features'] = X.columns.values.tolist()
# fi['importance'] = model.feature_importance(importance_type="gain")
#
# sns.barplot(x='importance', y='features', data=fi.sort_values(by='importance', ascending=True))
#
# # In[ ]:
#
#
# bst.save_model('/content/gdrive/My Drive/colab_data2/model_FULL_TRAIN_rollSTD_3_7_rollMEAN_3_7_shift_neg2_to_2.txt')  #
#
# # In[ ]:
#
#
# # utils.get_class_info(bst.params)
# bst.params
#
#
# # In[16]:
#
#
# def smooth_it(data, kernel_size=7):
#     kernel = np.ones(kernel_size) / kernel_size
#     return np.convolve(data, kernel, mode='same')
#
#
# # In[33]:
#
#
# yhat = bst.predict(df_andrew_x)
# np.mean(1 * (smooth_it(yhat, 7) > .5) == andrew_y)
#
# # In[17]:
#
#
# yhat = bst.predict(df_val_x)
# np.mean(1 * (smooth_it(yhat, 7) > .5) == val_y)
#
# # In[18]:
#
#
# yhat = bst.predict(df_train_x)
# np.mean(1 * (smooth_it(yhat, 7) > .5) == train_y)
#
# # In[19]:
#
#
# yhat = bst.predict(df_test_x)
# np.mean(1 * (smooth_it(yhat, 7) > .5) == test_y)
#
# # In[ ]:
#
#
# test_y
#
# # In[102]:
#
#
# utils.np_stats(yhat)
#
# # In[61]:
#
#
# plt.figure(figsize=(10, 10))
# real_y = andrew_y
#
# i1 += 4000
# # i1 = 0
# i2 = i1 + 4000
# yhat2 = 1 * (yhat > 0.5)
#
# # plt.plot(yhat[i1:i2])
# tmpy = smooth_it(yhat[i1:i2], kernel_size=3)
# plt.plot(tmpy, '--')
# tmpy = 1 * (tmpy > 0.5)
# plt.plot(tmpy)
# plt.plot(andrew_y[i1:i2] - 1.1)
#
# plt.plot((andrew_y[i1:i2] - yhat2[i1:i2] - 1.1) / 2, '.')
# plt.title(str(i1) + ' to ' + str(i2))
#
# # In[71]:
#
#
# plt.figure(figsize=(20, 10))
# frame_segs = utils.loop_segments(frame_nums_andrew, returnaslist=True)
#
# # for i1, i2 in utils.loop_segments(frame_nums_andrew):
#
# # cnt = -1
# cnt += 1
# i1, i2 = frame_segs[0][cnt], frame_segs[1][cnt]
#
# yhat2 = 1 * (yhat > 0.5)
#
# # plt.plot(yhat[i1:i2])
# tmpy = smooth_it(yhat[i1:i2], kernel_size=3)
# plt.plot(tmpy, '--')
# tmpy = 1 * (tmpy > 0.5)
# plt.plot(tmpy)
# plt.plot(andrew_y[i1:i2] - 1.1)
#
# plt.plot((andrew_y[i1:i2] - yhat2[i1:i2] - 1.1) / 2, '.')
# plt.title(str(i1) + ' to ' + str(i2))
#
# # In[42]:
#
#
# frame_segs[0][cnt], frame_segs[1][cnt]
#
# # #load model
#
# # In[60]:
#
#
# get_ipython().run_cell_magic('time', '',
#                              "model_file = r'D:\\models\\MODEL_FULL_TRAIN_rollSTD_3_7_rollMEAN_3_7_shift_neg2_to_2_val_with_real_val_ANDREW.txt'\nmodel_file = r'D:\\models\\MODEL_FULL_TRAIN_rollSTD_3_7_rollMEAN_3_7_shift_neg2_to_2_val_with_real_val_PM_JC.txt'\n\n# bst = lgb.load(filename = model_file)\n\nbst = lgb.Booster(model_file=model_file)")
#
# # In[13]:
#
#
# utils.get_class_info(bst)
#
# # In[17]:
#
#
# get_ipython().run_cell_magic('time', '',
#                              'x = df_test_x[:4000, :]\ny_pred = bst.predict(x)\nutils.np_stats(y_pred)\nplt.plot(y_pred)')
#
# # In[12]:
#
#
# get_ipython().run_cell_magic('time', '', 'x = df_test_x[:10, :]\ny_pred = bst.predict_proba(x)')
#
# # #make and save predictions
#
# # In[ ]:
#
#
# h5src = '/content/gdrive/My Drive/colab_data2/model_testing_features_data/feature_data/8_testing_h5s/'
#
# h5 = utils.get_h5s(h5src)
#
# time_length = 1
# batch_size = None
# label_key = '[0, 1]- (no touch, touch)'
# feature_len = 2048
# label_index_to_lstm_len = None
# edge_value = np.nan
# remove_any_time_points_with_edges = True
#
# G_test = image_tools.ImageBatchGenerator_feature_array(time_length, batch_size, h5, label_key=label_key,
#                                                        feature_len=feature_len,
#                                                        label_index_to_lstm_len=label_index_to_lstm_len,
#                                                        edge_value=edge_value,
#                                                        remove_any_time_points_with_edges=remove_any_time_points_with_edges)
# frame_nums_test = image_tools.get_h5_key_and_concatenate(h5, 'frame_nums')
#
# # In[ ]:
#
#
# test_x, test_y = G_test.__getitem__(0)
#
# # In[ ]:
#
#
# df_test_x = pd.DataFrame(test_x)
# del test_x  # save RAM
#
# # In[ ]:
#
#
# df_test_x = transform_x_data(df_test_x, frame_nums_test)
#
# # In[ ]:
#
#
# with open('df_test_x_2.pkl', 'wb') as f:
#     pickle.dump(df_test_x, f)
#
# # In[ ]:
#
#
# with open('df_test_x_2.pkl', 'rb') as f:
#     df_test_x = pickle.load(f)
#
# # In[ ]:
#
#
# # df_test_x = df_test_x.to_numpy()
#
#
# # In[ ]:
#
#
# yhat = bst.predict(df_test_x[:1000])
#
# # In[ ]:
#
#
# df_test_x[:1000].shape
#
# # In[ ]:
#
#
# from whacc import utils, image_tools, transfer_learning, analysis, error_analysis
#
# # #_____
#
# # In[ ]:
#
#
# # In[21]:
#
#
# get_ipython().system('pip install xgboost -U')
#
# import xgboost
#
# print("xgboost", xgboost.__version__)
# from xgboost import XGBClassifier
#
# # utils.get_class_info(xgboost)
#
#
# # In[ ]:
#
#
# clf = xgb.XGBClassifier(use_label_encoder=False, max_depth=200, n_estimators=400, subsample=1, learning_rate=0.09,
#                         reg_lambda=0.1, reg_alpha=0.1, gamma=1)
#
# # In[ ]:
#
#
# param = dict()
# param['gpu_id'] = 0
# param['tree_method'] = 'gpu_hist'
#
# param = {
#     'objective': 'binary:logistic',
#     'tree_method': 'binary:logistic',
#     'gpu_id': 0,
#     'use_label_encoder': False
# }
# model = XGBClassifier(**param)
# # train_data = lgb.Dataset(df_train_x, label=train_y)
# # validation_data = lgb.Dataset(df_test_x, label=test_y)
# model.fit(df_train_x, train_y)
#
# # In[ ]:
#
#
# grid = {'max_depth': 10}
# clf = XGBClassifier()
# clf.max_depth
# clf.set_params(**grid)
# XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
#               gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,
#               min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
#               objective='binary:logistic', reg_alpha=0, reg_lambda=1,
#               scale_pos_weight=1, seed=0, silent=True, subsample=1)
# clf.max_depth
#
# # In[ ]:
#
#
# xgbmodel = XGBClassifier(objective='binary:logistic', use_label_encoder=False)
#
# # In[ ]:
#
#
# x, y = G_train.__getitem__(0)
#
# xgbmodel.fit(x, y)
#
# xgbmodel.get_params()
#
# # In[ ]:
#
#
# time_length = 7
# batch_size = int(92213 / 10)
# print(batch_size)
# label_key = '[0, 1]- (no touch, touch)'
# feature_len = 2048
# label_index_to_lstm_len = 3
# edge_value = np.nanf
# remove_any_time_points_with_edges = True
#
# h5 = '/content/data/ALL_RETRAIN_H5_data____temp/ALL_RETRAIN_H5_data/3lag/train_3lag.h5'
# G_train = image_tools.ImageBatchGenerator_feature_array(time_length, batch_size, h5, label_key=label_key,
#                                                         feature_len=feature_len,
#                                                         label_index_to_lstm_len=label_index_to_lstm_len,
#                                                         edge_value=edge_value,
#                                                         remove_any_time_points_with_edges=remove_any_time_points_with_edges)
#
# # In[ ]:
#
#
# model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, par)
#
# model.fit(x, y)
#
# # #_____
#
# # ##build model
#
# # In[ ]:
#
#
# model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, n_estimators=500, subsample=.9,
#                       max_depth=10)
#
# # In[ ]:
#
#
# model.get_params()
#
# # In[ ]:
#
#
# del model
#
# # In[ ]:
#
#
# # ##fit model
#
# # In[ ]:
#
#
# import time
#
# # In[ ]:
#
#
# st = time.time()
# x, y = G_train.__getitem__(0)
# model.fit(x, y)
# print(time.time() - st)
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# train_x.size  # about .5 gb on ram
#
# # In[ ]:
#
#
# xgbPredictor = xgboost.XGBRegressor(**self.xgb_params)
# xgbPredictor.fit(Xs, ys)
#
# # This hack should only be used if tree_method == gpu_hist or gpu_exact
# if self.xgb_params['tree_method'][:3] == 'gpu':
#     with tempfile.TemporaryFile() as dump_file:
#         pickle.dump(xgbPredictor, dump_file)
#         dump_file.seek(0)
#         self.predictor_ = pickle.load(dump_file)
# else:
#     self.predictor_ = xgbPredictor
#
# # #full batch all at once
#
# # In[ ]:
#
#
# # time_length = 7
# # batch_size = int(92213/1)
# # print(batch_size)
# # label_key='[0, 1]- (no touch, touch)'
# # feature_len=2048
# # label_index_to_lstm_len=3
# # edge_value= -1
# # remove_any_time_points_with_edges=True
#
# # h5 = '/content/data/ALL_RETRAIN_H5_data____temp/ALL_RETRAIN_H5_data/3lag/train_3lag.h5'
# # G_train = image_tools.ImageBatchGenerator_feature_array(time_length, batch_size, h5, label_key=label_key, feature_len=feature_len,
# #                                       label_index_to_lstm_len=label_index_to_lstm_len, edge_value=edge_value,
# #                                       remove_any_time_points_with_edges = remove_any_time_points_with_edges)
#
#
# # In[ ]:
#
#
# parameters = {'base_score': 0.23,  # i think this should be set to mean of the labels it is the bias value
#               'booster': 'gbtree',
#               'colsample_bylevel': 1,
#               # "colsample_bylevel" is the fraction of features (randomly selected) that will be used in each node to train each tree.
#               'colsample_bynode': 1,
#               'colsample_bytree': 1,
#               # "colsample_bytree" is the fraction of features (randomly selected) that will be used to train each tree.
#               'enable_categorical': False,
#               'gamma': 0,
#               'gpu_id': -1,
#               'importance_type': None,
#               'interaction_constraints': '',
#               'learning_rate': [.2, .1, .05, .01],
#               'max_delta_step': 0,
#               'max_depth': [8, 12],
#               'min_child_weight': 1,
#               'monotone_constraints': '()',
#               'n_estimators': [100, 250, 500],
#               'n_jobs': -1,  # use all core I think 8
#               'num_parallel_tree': 1,
#               'objective': 'binary:logistic',
#               'predictor': 'auto',
#               'random_state': 0,
#               'reg_alpha': 0,
#               'reg_lambda': 1,
#               'scale_pos_weight': 1,
#               'single_precision_histogram': True,  # supposed to make it faster as well
#               'subsample': 0.9,
#               # "subsample" is the fraction of the training samples (randomly selected) that will be used to train each tree.
#               #  this seems like this should not be 1 it should be lower to work better ......
#               'tree_method': 'hist',  # was 'exact' but 'hist' is supposed ot be faster
#               'use_label_encoder': False,
#               'validate_parameters': 1,
#               'verbosity': 1}
# parameters = {
#     'learning_rate': [.2, .1, .05, .01],
#     'max_depth': [8, 12],
#     'n_estimators': [100, 250, 500]}
# for k in parameters:
#     parameters[k] = utils.make_list(parameters[k], suppress_warning=True)
# # parameters
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # !lscpu
#
#
# # In[ ]:
#
#
# # # model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, single_precision_histogram=True)
# # from sklearn.model_selection import GridSearchCV
# # # parameters = {
# # #     'max_depth': range (2, 10, 1),
# # #     'n_estimators': range(60, 220, 40),
# # #     'learning_rate': [0.1, 0.01, 0.05]}
#
# # model = XGBClassifier(objective= 'binary:logistic')
#
# # grid_search = GridSearchCV(
# #   estimator=model,
# #   param_grid=parameters,
# #   scoring = 'roc_auc',
# #   n_jobs = 10,
# #   cv = 10,
# #   verbose=True)
#
# # grid_search.fit(x[:20], y[:20])
#
#
# # In[ ]:
#
#
# from sklearn.model_selection import GridSearchCV
#
# model = XGBClassifier(objective='binary:logistic')
# kfold = 5
# gs = GridSearchCV(estimator=model,
#                   param_grid=parameters,
#                   scoring='roc_auc',
#                   n_jobs=-1,
#                   cv=kfold,
#                   verbose=3,
#                   return_train_score=True)
#
# # In[ ]:
#
#
# train_x, train_y
# test_x, test_y
#
# # In[ ]:
#
#
# # grid_result.best_estimator_.model.model.history.history
# from sklearn import preprocessing
#
# train_x = preprocessing.scale(train_x)
# test_x = preprocessing.scale(test_x)
# # train_x = preprocessing.scale(train_x)
# # train_x = preprocessing.scale(train_x)
#
#
# # In[ ]:
#
#
# train_x[:10, :5]
#
# # In[ ]:
#
#
# grid_result = gs.fit(train_x[:10, :5],
#                      train_y[:10],
#                      verbose=3)
#
# # In[ ]:
#
#
# grid_result = gs.fit([train_x[:10, :5]],
#                      train_y[:10],
#                      verbose=3)
#
# # In[ ]:
#
#
# # grid_result = gs.fit(train_x[:100, :5],
# #                     train_y[:100],
# #                     verbose=1,
# #                     validation_data=(test_x[:100, 5], test_y[:100]))
#
#
# # In[ ]:
#
#
# import joblib
#
# try:
#     joblib.dump(gs, '/content/gdrive/My Drive/XGBoost_gris_search/XGBoost_grid_search_001.pkl')
# except:
#     joblib.dump(grid_result, '/content/gdrive/My Drive/XGBoost_gris_search/XGBoost_grid_search_001.pkl')
# joblib.dump(grid_result, '/content/gdrive/My Drive/XGBoost_gris_search/XGBoost_grid_search_001.pkl')
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# clf = XGBClassifier()
# clf.max_depth
# clf.set_params(**grid)
#
# clf.get_params()
# history = clf.fit(train_x[:10, :5], train_y[:10],
#
#                   # In[ ]:
#
#                   grid={'max_depth': 10}
# clf = XGBClassifier()
# clf.max_depth
# clf.set_params(**grid)
# XGBClassifier(base_score=0.5,
#               colsample_bylevel=1,
#               colsample_bytree=1,
#               gamma=0,
#               learning_rate=0.1,
#               max_delta_step=0,
#               max_depth=10,
#               min_child_weight=1,
#               missing=None,
#               n_estimators=500,
#               nthread=-1,
#               objective='binary:logistic',
#               reg_alpha=0,
#               reg_lambda=1,
#               scale_pos_weight=1,
#               seed=0,
#               silent=True,
#               subsample=1)
# clf.n_estimators
# # clf.fit(x[:20], y[:20])
# history = clf.fit(train_x[:10, :5], train_y[:10])
#
# # In[ ]:
#
#
# grid = {'max_depth': 10}
# clf = XGBClassifier()
# clf.max_depth
# clf.set_params(**grid)
# XGBClassifier(base_score=0.5,
#               colsample_bylevel=1,
#               colsample_bytree=1,
#               gamma=0,
#               learning_rate=0.1,
#               max_delta_step=0,
#               max_depth=10,
#               min_child_weight=1,
#               missing=None,
#               n_estimators=500,
#               nthread=-1,
#               objective='binary:logistic',
#               reg_alpha=0,
#               reg_lambda=1,
#               scale_pos_weight=1,
#               seed=0,
#               silent=True,
#               subsample=1)
# clf.n_estimators
# # clf.fit(x[:20], y[:20])
# history = clf.fit(train_x[:10, :5], train_y[:10])
#
# # In[ ]:
#
#
# import time
#
# st = time.time()
#
# model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, single_precision_histogram=True,
#                       n_estimators=500, subsample=.9, max_depth=10)
# x, y = G_train.__getitem__(0)
# model.fit(x, y)
#
# print(time.time() - st)
#
# # In[ ]:
#
#
# import time
#
# st = time.time()
#
# model_small_test = XGBClassifier(objective='binary:logistic', use_label_encoder=False, single_precision_histogram=True,
#                                  n_estimators=500, subsample=.9, max_depth=10)
# x, y = G_train.__getitem__(0)
# model_small_test.fit(x[:10, :5], y[:10])
#
# print(time.time() - st)
#
# # In[ ]:
#
#
# x[:10, :10]
#
# # In[ ]:
#
#
# from sklearn import preprocessing
#
# x = preprocessing.scale(x)
#
# # In[ ]:
#
#
# num_data_points = 10000
# model_small_test = XGBClassifier(objective='binary:logistic', use_label_encoder=False, single_precision_histogram=True,
#                                  n_estimators=500, subsample=.9, max_depth=10)
# st = time.time()
# model_small_test.fit(x[:num_data_points, :], y[:num_data_points])
# print(time.time() - st)
#
# # In[ ]:
#
#
# """
# before scaling all 2048 features
# 100  --> 5.22572660446167
# 1000 --> 11.855197429656982
# 10000--> 55.457772970199585
#
# after scaling all 2048 features
# 100  --> 3.963913917541504
# 1000 --> 9.200135469436646
# 10000--> 48.088958740234375
# """
#
# # In[ ]:
#
#
# xtmp = [100, 1000, 10000]
# plt.plot(xtmp, [5.22572660446167,
#                 11.855197429656982,
#                 55.457772970199585])
# plt.plot(xtmp, [3.963913917541504,
#                 9.200135469436646,
#                 48.088958740234375])
#
# # In[ ]:
#
#
# import pickle
#
# file_name = "/content/gdrive/My Drive/XGBoost_gris_search/xgb_reg.pkl"
#
# # save
# pickle.dump(model, open(file_name, "wb"))
#
# # In[ ]:
#
#
# model = pickle.load(open(file_name, "rb"))
#
# # In[ ]:
#
#
# from xgboost import plot_importance, plot_tree
#
# # In[ ]:
#
#
# # plt.plot(x[:, 163], x[:, 171], '.')
# tmpx = x[:10, :5]
# R1 = np.corrcoef(x)
# R1
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# plt.rcParams["figure.figsize"] = (30, 30)
# _ = plot_importance(model, height=0.9, max_num_features=200)
#
# # In[ ]:
#
#
# from IPython.core.pylabtools import figsize
#
# plot_importance(figsize=(10, 10))
#
# # In[ ]:
#
#
# real = np.asarray([])
# pred = np.asarray([])
# for kk in tqdm(range(G_val.__len__())):
#     x, y = G_val.__getitem__(kk)
# real = np.append(real, y)
# y_hat = model.predict(x)
# pred = np.append(pred, y_hat)
# np.mean(pred == real)
#
# # In[ ]:
#
#
# model
#
# # In[ ]:
#
#
# real = np.asarray([])
# pred = np.asarray([])
# for kk in tqdm(range(G_test.__len__())):
#     x, y = G_test.__getitem__(kk)
# real = np.append(real, y)
# y_hat = model.predict(x)
# pred = np.append(pred, y_hat)
# np.mean(pred == real)
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# real = np.asarray([])
# pred = np.asarray([])
# for kk in tqdm(range(G_train.__len__())):
#     x, y = G_train.__getitem__(kk)
# real = np.append(real, y)
# y_hat = model.predict(x)
# pred = np.append(pred, y_hat)
# np.mean(pred == real)
#
# # #__
# #
#
# #
#
# # In[ ]:
#
#
# utils.get_class_info(xgbmodel)
#
# # In[ ]:
#
#
# from xgboost import XGBClassifier
#
# xgbmodel = XGBClassifier(objective='binary:logistic')
#
# for k in range(G_train.__len__()):
#     x, y = G_train.__getitem__(k)
# xgbmodel.fit(x, y)
# for kk in range(G_val.__len__()):
#     x, y = G_val.__getitem__(kk)
# xgbmodel.score(x, y)
#
# # In[ ]:
#
#
# send_text('OOOOHHHHH SHIIIITTTTT ITS DOONNNEEEEE')
#
# # In[ ]:
#
#
# from xgboost import XGBClassifier
#
# utils.get_class_info(XGBClassifier)
# XGBClassifier()
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # #plot the features sorted to get an idea
#
# # In[ ]:
#
#
# h5 = '/content/data/ALL_RETRAIN_H5_data____temp/DATA_FULL/3lag/holy_test_set_10_percent_3lag.h5'
# # h5 = '/content/data/ALL_RETRAIN_H5_data____temp/ALL_RETRAIN_H5_data/3lag/train_3lag.h5'
# a = image_tools.ImageBatchGenerator_feature_array(7, 4000, h5, label_key='[0, 1]- (no touch, touch)', feature_len=2048,
#                                                   label_index_to_lstm_len=3, edge_value=-1,
#                                                   remove_any_time_points_with_edges=True)
# a.__len__()
#
# # In[ ]:
#
#
# x, y = a.__getitem__(9)
#
# # In[ ]:
#
#
# x.shape
#
# # In[ ]:
#
#
# # add_to = -2000
# add_to += 100
# plt.figure(figsize=(5, 700))
# tmp1 = int(2048 / 4)
# tmp1 = 2048 * 3
# # plt.imshow(x[2200+add_to:2350+add_to, ::tmp1].T)
# x2 = x[2200 + add_to:2350 + add_to, tmp1:tmp1 + 2000].T
# # sort_inds = np.argsort(np.mean(x2[:, 20:28], axis = 1))
# x3 = np.zeros_like(x2)
# for i, k in enumerate(sort_inds):
#     x3[i, :] = x2[k, :]
#
# y2 = y[2200 + add_to:2350 + add_to]
# y2 = y2 * np.max(x3)
# y2 = np.tile(y2, (10, 1))
# x3 = np.vstack((y2, x3, y2))
# plt.imshow(x3)
# # plt.axis('off')
# # plt.axis('tight')
# plt.show()
# # plt.figure(figsize=(5, 5))
# # time.sleep(3)
#
#
# plt.plot(y[2200 + add_to:2350 + add_to])
# plt.axis('off')
# plt.axis('tight')
# plt.ylim([-.1, 1.1])
