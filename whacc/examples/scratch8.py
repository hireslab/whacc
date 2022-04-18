from whacc import utils, image_tools

import copy
import numpy as np
import pandas as pd
import h5py

from tqdm.autonotebook import tqdm


class feature_maker():
    def __init__(self, h5_in, frame_num_ind=None, frame_nums=None, operational_key='FD__original', disable_tqdm=False,
                 delete_if_exists=False, index_features_delete_the_rest=None):
        """

        Parameters
        ----------
        h5_in : h5 string pointing to the h4 file with the data to transform
        frame_num_ind : int referencing the frame num  ind to transform, note: if save_it is one ALL data is converted
        automatically and saved in h5. frame_num_ind only works when you call with
        frame_nums : default None, auto looks for key 'frame_nums' in h5 file or you can insert your own
        operational_key : the data array key to be transformed
        disable_tqdm : default False, when save_it is True it will show a loading bar with the progress unless set to true
        delete_if_exists : default False, when calling a function with save_it as True, you can choose to overwrite that
        data by setting this value to True

        Returns
        -------
        feature maker class

        Examples
        ________

        h5_in = '/Users/phil/Desktop/AH0407_160613_JC1003_AAAC_3lag.h5'

        FM = feature_maker(h5_in, frame_num_ind=2, delete_if_exists = True) #set frame_num_ind to an int so that we can quickly look at the transformation
        data, data_name = FM.shift(5, save_it=False) # shift it 5 to the right, fills in with nan as needed, look at just the 5th frame num set (5th video)
        # to see how it looks
        print('the below name will be used to save the data in the the h5 when calling functions with save_it=True')
        print(data_name)
        FM.shift(5, save_it=True) # now lets save it

        data, data_name_rolling_mean_100 = FM.rolling(100, 'mean', save_it=True)  #lets do the rolling mean and save it

        data, data_name = FM.operate('diff', kwargs={'periods': 1}, save_it=False) # lets perform a diff operation
        print(data_name)
        print(data)
        FM.operate('diff', kwargs={'periods': -1}, save_it=True) # now lets save it and save it

        # now lets change the operational key so we can transform some data we just transformed, specifically the rolling mean 10
        FM.set_operation_key(data_name_rolling_mean_100)

        data, data_name_diff_100_mean = FM.operate('diff', kwargs={'periods': -50}, save_it=False) # lets check how the name will change
        print(data_name_diff_100_mean)
        print("notice the FD__ twice, this means the data has been transformed twice")
        print('also notice that how the data was transformed can be seen because it is split up by ____ (4 underscores)')

        data, data_name_diff_100_mean = FM.operate('diff', kwargs={'periods': -50}, save_it=True) # save it

        a = utils.print_h5_keys(h5_in, 1, 1)
        key_name_list = ['FD__original', 'FD__FD__FD__original_rolling_mean_W_100_SFC_0_MP_100_____diff_periods_-50____', 'FD__FD__original_rolling_mean_W_100_SFC_0_MP_100____']
        with h5py.File(h5_in, 'r+') as h:
            for k in key_name_list:
                plt.plot(h[k][:8000, 0])
        """
        if utils.h5_key_exists(h5_in, 'has_data_been_randomized'):
            assert image_tools.get_h5_key_and_concatenate(h5_in,
                                                          'has_data_been_randomized').tolist() is False, """this data has been randomized, it is not fit to perform temporal operations on, search for key 'has_data_been_randomized' for more info"""

        self._frame_num_ind_save_ = None
        self.disable_tqdm = disable_tqdm
        self.h5_in = h5_in
        self.frame_num_ind = frame_num_ind
        self.delete_if_exists = delete_if_exists
        self.operational_key = operational_key
        if frame_nums is None:
            print('extracting frame_nums from h5 file, ideally you should just put that in yourself though')
            frame_nums = image_tools.get_h5_key_and_concatenate(h5_in, 'frame_nums')
        self.all_frame_nums = frame_nums
        self.len_frame_nums = len(self.all_frame_nums)
        self.set_data_and_frame(frame_num_ind)
        index_features_delete_the_rest

    def set_data_inds(self, ind):  # frame nums used ot extract below in 'set_operation_key'
        tmp_inds = np.asarray(utils.loop_segments(self.all_frame_nums, returnaslist=True))
        if ind is None:
            self.data_inds = [tmp_inds[0][0], tmp_inds[-1][-1]]
        else:
            self.data_inds = tmp_inds[:, ind]

    def set_operation_key(self, key_name=None):
        if key_name is not None:
            self.operational_key = key_name
        with h5py.File(self.h5_in, 'r') as h:
            self.full_data_shape = np.asarray(h[self.operational_key].shape)
            if self.frame_num_ind is None:
                self.data = pd.DataFrame(copy.deepcopy(h[self.operational_key][:]))
            else:
                a = self.data_inds  # just the current frame numbers
                self.data = pd.DataFrame(copy.deepcopy(h[self.operational_key][a[0]:a[1]]))

    def init_h5_data_key(self, data_key, delete_if_exists=False):
        key_exists = utils.h5_key_exists(self.h5_in, data_key)
        assert not (
                key_exists and not delete_if_exists), "key exists, if you want to overwrite set 'delete_if_exists' = True"
        with h5py.File(self.h5_in, 'r+') as x:
            if key_exists and delete_if_exists:
                print('deleting key to overwrite it')
                del x[data_key]
            x.create_dataset_like(data_key, x[self.operational_key])

    def rolling(self, window, operation, shift_from_center=0, min_periods=None, save_it=False, kwargs={}):
        """

        Parameters
        ----------
        window : window size int
        operation : a string with the operation e.g. 'mean' or 'std' see pandas docs for rolling
        shift_from_center : default 0 but can shift as needed
        min_periods : default is equal to win length, only allows operation when we have this many data points. so deals with
        the edges
        save_it : bool, False, loop through frame nums and save to h5 file
        kwargs : dict of args that can be applied to 'operation' function

        Returns
        -------

        """
        if min_periods is None:
            min_periods = window
        add_name_list = ['FD__' + self.operational_key + '_rolling_',
                         '_W_' + str(window) + '_SFC_' + str(shift_from_center) + '_MP_' + str(min_periods) + '____']
        add_name_str = operation.join(add_name_list)
        if save_it:
            self._save_it_(self.rolling, window, operation, shift_from_center, min_periods, False, kwargs)
            return None, add_name_str
        data_frame = self.data.copy()
        for i1, i2 in utils.loop_segments(self.frame_nums):
            df_rolling = self.data[i1:i2].rolling(window=window, min_periods=min_periods, center=True)
            tmp_func = eval('df_rolling.' + operation)
            data_frame[i1:i2] = tmp_func(**kwargs).shift(shift_from_center).astype(np.float32)
        return data_frame, add_name_str

    def shift(self, shift_by, save_it=False):
        """

        Parameters
        ----------
        shift_by : amount to shift by
        save_it :  bool, False, loop through frame nums and save to h5 file

        Returns
        -------

        """
        add_name_str = 'FD__' + self.operational_key + '_shift_' + str(shift_by) + '____'
        if save_it:
            self._save_it_(self.shift, shift_by)
            return None, add_name_str

        data_frame = self.data.copy()
        for i1, i2 in utils.loop_segments(self.frame_nums):
            data_frame[i1:i2] = self.data[i1:i2].shift(shift_by).astype(np.float32)
        return data_frame, add_name_str

    def operate(self, operation, save_it=False, extra_name_str='', kwargs={}):
        """

        Parameters
        ----------
        operation : a string with the operation e.g. 'mean' or 'std' or 'diff' see pandas operation on  dataframes
        save_it :  bool, False, loop through frame nums and save to h5 file
        extra_name_str : add to key name string if desired
        kwargs : dict of args that can be applied to 'operation' function see pandas operation on  dataframes

        Returns
        -------

        """
        add_name_str_tmp = self.dict_to_string_name(kwargs)
        add_name_str = 'FD__' + self.operational_key + '_' + operation + '_' + add_name_str_tmp + extra_name_str + '____'
        if save_it:
            self._save_it_(self.operate, operation, False, extra_name_str, kwargs)
            return None, add_name_str
        data_frame = self.data.copy()
        for i1, i2 in utils.loop_segments(self.frame_nums):
            data_frame[i1:i2] = eval('self.data[i1:i2].' + operation + '(**kwargs).astype(np.float32)')
        return data_frame, add_name_str

    @staticmethod
    def dict_to_string_name(in_dict):
        """
        used to transform dict into a string name for naming h5 keys
        Parameters
        ----------
        in_dict : dict

        Returns
        -------
        string
        """
        str_list = []
        for k in in_dict:
            str_list.append(k)
            str_list.append(str(in_dict[k]))
        return '_'.join(str_list)

    def set_data_and_frame(self, ind):
        self.frame_num_ind = ind
        if ind is None:
            self.frame_nums = copy.deepcopy(self.all_frame_nums)
        else:
            self.frame_nums = [self.all_frame_nums[ind]]
        self.set_data_inds(ind)
        self.set_operation_key()

    # tqdm(sorted(random_frame_inds[i]), disable=disable_TQDM, total=total_frames, initial=cnt1)
    def _save_it_(self, temp_funct, *args):
        self._frame_num_ind_save_ = copy.deepcopy(self.frame_num_ind)
        with h5py.File(self.h5_in, 'r+') as h:
            for k in tqdm(range(self.len_frame_nums), disable=self.disable_tqdm):
                self.set_data_and_frame(k)
                data, key_name = temp_funct(*args)
                if k == 0:
                    self.init_h5_data_key(key_name, delete_if_exists=self.delete_if_exists)
                    print('making key, ' + key_name)
                a = self.data_inds
                h[key_name][a[0]:a[1]] = data
        self.set_data_and_frame(self._frame_num_ind_save_)  # set data back to what it was when user set it

    def total_rolling_operation(self, data_in, win, operation_function, shift_from_center=0):
        """
        NOTE: for making feature data proper key names for saving is 'FD_TOTAL_' folowed by operation e.g. 'FD_TOTAL_nanstd'
        Parameters
        ----------
        data_in : 2D matrix
        win : window size
        operation_function : function to be applies to each window e.g. np.nanmean, note DON'T include parentheses
        shift_from_center : num units shift from center

        Returns
        -------
        data_out: output data
        is_nan_inds: bool array indexing where nans were
        """
        assert win % 2 == 1, 'window must be odd'
        mid = win // 2
        pad = np.zeros([win, data_in.shape[1]]) * np.nan

        L_pad = pad[:mid - shift_from_center]
        R_pad = pad[:mid + shift_from_center]

        data_in = np.vstack([L_pad, data_in, R_pad])

        is_nan_inds = []
        data_out = []
        for k in range(data_in.shape[0] - win + 1):
            x = data_in[k:(k + win)]
            data_out.append(operation_function(x))
            is_nan_inds.append(np.any(np.isnan(x)))
        return np.asarray(data_out), np.asarray(is_nan_inds)

    def total_rolling_operation_h5_wrapper(self, window, operation, key_to_operate_on, mod_key_name=None,
                                           save_it=False,
                                           shift_from_center=0):
        if save_it:
            assert mod_key_name is not None, """if save_it is True, 'mod_key_name' must not be None e.g. 'FD_TOTAL_std_1_of_'"""
        all_data = []
        with h5py.File(self.h5_in, 'r') as h:
            frame_nums = h['frame_nums'][:]
            for i, (i1, i2) in enumerate(utils.loop_segments(frame_nums)):
                data_out, is_nan_inds = self.total_rolling_operation(h[key_to_operate_on][i1:i2, :], window, operation,
                                                                     shift_from_center=shift_from_center)
                all_data.append(data_out)
        all_data = np.hstack(all_data)
        mod_key_name = mod_key_name + key_to_operate_on
        if save_it:
            utils.overwrite_h5_key(self.h5_in, mod_key_name, all_data)
        return all_data


from tqdm.auto import tqdm
import numpy as np

########################################################################################################################
########################################################################################################################
h5_feature_data = '/Users/phil/Desktop/pipeline_test/AH0407x160609_3lag_feature_data.h5'
FM = feature_maker(h5_feature_data, operational_key='FD__original', delete_if_exists=True)

for periods in tqdm([-5]):
    data, key_name = FM.shift(periods, save_it=True)
########################################################################################################################
########################################################################################################################
########################################################################################################################
# from whacc.feature_maker import feature_maker, total_rolling_operation_h5_wrapper


for periods in tqdm([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]):
    data, key_name = FM.shift(periods, save_it=True)

for smooth_it_by in tqdm([3, 7, 11, 15, 21, 41, 61]):
    data, key_name = FM.rolling(smooth_it_by, 'mean', save_it=True)

for periods in tqdm([-50, -20, -10, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 10, 20, 50]):
    data, key_name = FM.operate('diff', kwargs={'periods': periods}, save_it=True)

for smooth_it_by in tqdm([3, 7, 11, 15, 21, 41, 61]):
    data, key_name = FM.rolling(smooth_it_by, 'std', save_it=True)

win = 1
# key_to_operate_on = 'FD__original'
op = np.std
mod_key_name = 'FD_TOTAL_std_' + str(win) + '_of_'
all_keys = utils.lister_it(utils.print_h5_keys(FM.h5_in, 1, 0), 'FD__', 'FD_TOTAL')
for key_to_operate_on in tqdm(all_keys):
    data_out = FM.total_rolling_operation_h5_wrapper(win, op, key_to_operate_on, mod_key_name=mod_key_name,
                                                     save_it=True)

utils.get_selected_features(greater_than_or_equal_to=4)

inds = fd_dict['features_used_of_10'] >= 4
tmp1 = fd_dict['full_feature_names_and_neuron_nums'][inds]
import numpy as np

tmp2 = np.unique(fd_dict['full_neuron_nums'][inds])

for k in tmp1:
    print(k)

"""
'FD_TOTAL_std_1_of_original_diff_periods_3',
'FD_TOTAL_std_1_of_original_rolling_mean_W_3_SFC_0_MP_3',
'FD_TOTAL_std_1_of_original_rolling_mean_W_7_SFC_0_MP_7',
'FD_TOTAL_std_1_of_original_rolling_mean_W_11_SFC_0_MP_11',
'FD_TOTAL_std_1_of_original_shift_3', 'FD_TOTAL_std_1_of_original'],

do those first then take any of the completed data from them that are used as single features 

then save the TOTAL operations and singles, 

then go through all the singles

"""

len(utils.lister_it(tmp1, '_diff_'))

h52 = '/Users/phil/Desktop/pipeline_test/AH0407x160609_3lag_feature_data.h5'
utils.print


def total_rolling_sliding_window_view(data_in, win, operation_function, shift_from_center=0):
    assert win % 2 == 1, 'window must be odd'
    mid = win // 2
    pad = np.zeros([win, data_in.shape[1]]) * np.nan

    L_pad = pad[:mid - shift_from_center]
    R_pad = pad[:mid + shift_from_center]

    data_in = np.vstack([L_pad, data_in, R_pad])
    w = data_in.shape[1]
    data_in = np.lib.stride_tricks.sliding_window_view(data_in, (win, w))
    data_in = np.reshape(data_in, [-1, win * w])
    data_out = operation_function(data_in, axis=1)
    return np.asarray(data_out)
