from whacc.feature_maker import convert_h5_to_feature_h5, standard_feature_generation, load_selected_features
from whacc.pole_tracker import PoleTracking
from whacc import model_maker
import shutil

import numpy as np
from pathlib import Path
import os
import sys
import glob
from natsort import os_sorted
import scipy.io as spio
import h5py
import matplotlib.pyplot as plt
import pandas as pd

import copy
import time
from whacc import image_tools
import whacc
import platform
import subprocess
from scipy.signal import medfilt, medfilt2d
import pickle
from tqdm.autonotebook import tqdm

from datetime import timedelta, datetime
import pytz
import warnings

def get_time_string(time_zone_string = 'America/Los_Angeles'):
    tz = pytz.timezone(time_zone_string)
    loc_dt = pytz.utc.localize(datetime.utcnow())
    current_time = loc_dt.astimezone(tz)
    todays_version = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    return todays_version


def batch_process_videos_on_colab(bd, local_temp_dir, video_batch_size=2):
    time_str = get_time_string()
    bd_base_name = os.path.basename(os.path.normpath(bd))
    # load model once in the beginning
    RESNET_MODEL = model_maker.load_final_model()  # NOTE: there will be a warning since this likely isn't optimized for GPU, this is fine
    time_dict = {'num_files': [], 'time_copy_to_local': [], 'time_all': [], 'time_to_track': [], 'time_to_3lag': [],
                 'time_to_features': [], 'time_to_all_features': [], 'time_to_cut_features': []}
    time_df = None
    while True:  # keep going until there are no more files to process

        grab_file_list = True
        while grab_file_list:  # continuously look for files to run
            # get files that tell us which mp4s to process
            list_of_file_dicts = np.asarray(get_files(bd, '*file_list_for_batch_processing.pkl'))
            # sort it by the newest first since we we edit it each time (becoming the newest file)
            # this ensures we finished one set completely first
            inds = np.flip(np.argsort([os.path.getctime(k) for k in list_of_file_dicts]))
            list_of_file_dicts = list_of_file_dicts[inds]
            if len(list_of_file_dicts) == 0:
                print('FINISHED PROCESSING')
                return time_df
                # assert False, "FINISHED PROCESSING no more files to process"
            # load file dictionary
            file_dict = load_obj(list_of_file_dicts[0])
            # get base directory for current videos we are processing
            mp4_bd = os.path.dirname(list_of_file_dicts[0])
            # copy folder structure for the finished mp4s and predictions to go to
            copy_folder_structure(bd, os.path.normpath(bd) + '_FINISHED')
            # check if all the files have already been processes
            if np.all(file_dict['is_processed']==True):
                x = list_of_file_dicts[0]  # copy instruction file with list of mp4s to final directory we are finished
                shutil.move(x, x.replace(bd_base_name, bd_base_name + '_FINISHED'))
                x = os.path.dirname(x) + os.sep + 'template_img.png'
                shutil.move(x, x.replace(bd_base_name, bd_base_name + '_FINISHED'))
            else:
                grab_file_list = False  # ready to run data
        start = time.time()
        time_list = [start]
        # overwrite local folder to copy files to
        if os.path.exists(local_temp_dir):
            shutil.rmtree(local_temp_dir)
        Path(local_temp_dir).mkdir(parents=True, exist_ok=True)
        # copy over mp4s and template image to local directory
        x = os.sep + 'template_img.png'
        template_dir = local_temp_dir + x
        shutil.copy(mp4_bd + x, template_dir)
        process_these_videos = np.where(file_dict['is_processed'] == False)[0][:video_batch_size]
        for i in process_these_videos:
            x = os.sep + os.path.basename(file_dict['mp4_names'][i])
            shutil.copy(mp4_bd + x, local_temp_dir + x)
        time_list.append(time.time())  #

        # track the mp4s for the pole images
        PT = PoleTracking(video_directory=local_temp_dir, template_png_full_name=template_dir)
        PT.track_all_and_save()
        time_list.append(time.time())
        # convert the images to '3lag' images
        #  h5_in = '/Users/phil/Desktop/temp_dir/AH0407x160609.h5'
        h5_in = PT.full_h5_name
        h5_3lag = h5_in.replace('.h5', '_3lag.h5')
        image_tools.convert_to_3lag(h5_in, h5_3lag)
        time_list.append(time.time())

        # convert to feature data
        # h5_3lag = '/Users/phil/Desktop/temp_dir/AH0407x160609_3lag.h5'
        h5_feature_data = h5_3lag.replace('.h5', '_feature_data.h5')
        in_gen = image_tools.ImageBatchGenerator(500, h5_3lag, label_key='labels')
        convert_h5_to_feature_h5(RESNET_MODEL, in_gen, h5_feature_data)
        time_list.append(time.time())

        # delete 3lag don't it need anymore
        os.remove(h5_3lag)
        # h5_feature_data = '/Users/phil/Desktop/temp_dir/AH0407x160609_3lag_feature_data.h5'
        # generate all the modified features (41*2048)+41 = 84,009
        standard_feature_generation(h5_feature_data)
        time_list.append(time.time())
        all_x = load_selected_features(h5_feature_data)
        # delete the big o' file
        file_nums = str(process_these_videos[0] + 1) + '_to_' + str(process_these_videos[-1] + 1) + '_of_' + str(
            len(file_dict['is_processed']))
        h5_final = h5_in.replace('.h5', '_final_to_combine_' + file_nums + '.h5')
        print(h5_final)
        with h5py.File(h5_final, 'w') as h:
            h['final_3095_features'] = all_x
        copy_over_all_non_image_keys(h5_in, h5_final)
        # then if you want you can copy the images too maybe just save as some sort of mp4 IDK
        time_list.append(time.time())
        os.remove(h5_feature_data)
        x = os.path.dirname(list_of_file_dicts[0]) + os.sep
        dst = x.replace(bd_base_name, bd_base_name + '_FINISHED') + os.path.basename(h5_final)
        shutil.copy(h5_final, dst)

        for k in process_these_videos:  # save the dict file so that we know the video has been processed
            file_dict['is_processed'][k] = True
        save_obj(file_dict, list_of_file_dicts[0])

        # move the mp4s to the alt final dir
        for i in process_these_videos:
            x = mp4_bd + os.sep + os.path.basename(file_dict['mp4_names'][i])
            final_mp4_dir = x.replace(bd_base_name, bd_base_name + '_FINISHED_MP4s')
            Path(os.path.dirname(final_mp4_dir)).mkdir(parents=True, exist_ok=True)
            shutil.move(x, final_mp4_dir)
        time_list.append(time.time())

        time_array = np.diff(time_list)
        df_name_list = ['copy mp4s to local', 'track the pole', 'convert to 3lag', 'create feature data (CNN)',
                        'engineer all features', 'make final h5 3095', 'copy  h5 and mp4s to final destination',
                        'number of files']

        len_space = ' ' * max(len(k) + 4 for k in df_name_list)
        print('operation                                 total     per file')
        to_print = ''.join([(k2 + len_space)[:len(len_space)] + str(timedelta(seconds=k)).split(".")[
            0] + '   ' + str(timedelta(seconds=k3)).split(".")[0] + '\n' for k, k2, k3 in
                            zip(time_array, df_name_list, time_array / len(process_these_videos))])
        print(to_print)
        print('TOTAL                                     total     per file')
        print(len_space + str(timedelta(seconds=int(np.sum(time_array)))).split(".")[0] + '   ' +
              str(timedelta(seconds=np.sum(time_array) / len(process_these_videos))).split(".")[0])

        time_array = np.concatenate((time_array, [len(process_these_videos)]))
        if time_df is None:
            time_df = pd.DataFrame(time_array[None, :], columns=df_name_list)
        else:
            tmp_df = pd.DataFrame(time_array[None, :], columns=df_name_list)
            time_df = time_df.append(tmp_df, ignore_index=True)
        save_obj(time_df, os.path.normpath(bd) + '_FINISHED' + os.sep + 'time_df_'+time_str+'.pkl')


# def batch_process_videos_on_colab(bd, local_temp_dir, video_batch_size=30):
#     bd_base_name = os.path.basename(os.path.normpath(bd))
#     # load model once in the beginning
#     RESNET_MODEL = model_maker.load_final_model()  # NOTE: there will be a warning since this likely isn't optimized for GPU, this is fine
#     fd = load_feature_data()  # load feature data info
#     while True:  # once a   ll files are
#         time_list = []
#         start = time.time()
#         grab_file_list = True
#         while grab_file_list:  # continuously look for files to run
#             # get files that tell us which mp4s to process
#             list_of_file_dicts = np.asarray(get_files(bd, '*file_list_for_batch_processing.pkl'))
#             # sort it by the newest first since we we edit it each time (becoming the newest file)
#             # this ensures we finished one set completely first
#             inds = np.flip(np.argsort([os.path.getctime(k) for k in list_of_file_dicts]))
#             list_of_file_dicts = list_of_file_dicts[inds]
#             if len(list_of_file_dicts) == 0:
#                 print('FINISHED PROCESSING')
#                 assert False, "FINISHED PROCESSING no more files to process"
#             # load file dictionary
#             file_dict = load_obj(list_of_file_dicts[0])
#             # get base directory for current videos we are processing
#             mp4_bd = os.path.dirname(list_of_file_dicts[0])
#             # copy folder structure for the finished mp4s and predictions to go to
#             copy_folder_structure(bd, os.path.normpath(bd) + '_FINISHED')
#             # check if all the files have already been processes
#             if np.all(file_dict['is_processed'] == True):
#                 x = list_of_file_dicts[
#                     0]  # copy the instruction file with list of mp4s to final directory we are finished
#                 shutil.move(x, x.replace(bd_base_name, bd_base_name + '_FINISHED'))
#                 x = os.path.dirname(x) + os.sep + 'template_img.png'
#                 shutil.move(x, x.replace(bd_base_name, bd_base_name + '_FINISHED'))
#             else:
#                 grab_file_list = False  # ready to run data
#
#         # overwrite local folder to copy files to
#         if os.path.exists(local_temp_dir):
#             shutil.rmtree(local_temp_dir)
#         Path(local_temp_dir).mkdir(parents=True, exist_ok=True)
#         # copy over mp4s and template image to local directory
#         x = os.sep + 'template_img.png'
#         template_dir = local_temp_dir + x
#         shutil.copy(mp4_bd + x, template_dir)
#         process_these_videos = np.where(file_dict['is_processed'] == False)[0][:video_batch_size]
#         for i in process_these_videos:
#             x = os.sep + os.path.basename(file_dict['mp4_names'][i])
#             shutil.copy(mp4_bd + x, local_temp_dir + x)
#
#         # track the mp4s for the pole images
#         PT = PoleTracking(video_directory=local_temp_dir, template_png_full_name=template_dir)
#         PT.track_all_and_save()
#
#         # convert the images to '3lag' images
#         #  h5_in = '/Users/phil/Desktop/temp_dir/AH0407x160609.h5'
#         h5_in = PT.full_h5_name
#         h5_3lag = h5_in.replace('.h5', '_3lag.h5')
#         image_tools.convert_to_3lag(h5_in, h5_3lag)
#
#         # convert to feature data
#         # h5_3lag = '/Users/phil/Desktop/temp_dir/AH0407x160609_3lag.h5'
#         h5_feature_data = h5_3lag.replace('.h5', '_feature_data.h5')
#         in_gen = image_tools.ImageBatchGenerator(500, h5_3lag, label_key='labels')
#         convert_h5_to_feature_h5(RESNET_MODEL, in_gen, h5_feature_data)
#
#         # delete 3lag don't it need anymore
#         os.remove(h5_3lag)
#         # h5_feature_data = '/Users/phil/Desktop/temp_dir/AH0407x160609_3lag_feature_data.h5'
#         # generate all the modified features (41*2048)+41 = 84,009
#         standard_feature_generation(h5_feature_data)
#         all_x = load_selected_features(h5_feature_data)
#         # delete the big o' file
#         file_nums = str(process_these_videos[0] + 1) + '_to_' + str(process_these_videos[-1] + 1) + '_of_' + str(
#             len(file_dict['is_processed']))
#         h5_final = h5_in.replace('.h5', '_final_to_combine_' + file_nums + '.h5')
#         print(h5_final)
#         with h5py.File(h5_final, 'w') as h:
#             h['final_3095_features'] = all_x
#         copy_over_all_non_image_keys(h5_in, h5_final)
#         # then if you want you can copy the images too maybe just save as some sort of mp4 IDK
#         os.remove(h5_feature_data)
#         x = os.path.dirname(list_of_file_dicts[0]) + os.sep
#         dst = x.replace(bd_base_name, bd_base_name + '_FINISHED') + os.path.basename(h5_final)
#         shutil.copy(h5_final, dst)
#
#         for k in process_these_videos:  # save the dict file so that we know the video has been processed
#             file_dict['is_processed'][k] = True
#         save_obj(file_dict, list_of_file_dicts[0])
#
#         # move the mp4s to the final dir
#         for i in process_these_videos:
#             x = mp4_bd + os.sep + os.path.basename(file_dict['mp4_names'][i])
#             shutil.move(x, x.replace(bd_base_name, bd_base_name + '_FINISHED'))
#         time_list.append(time.time() - start)


def auto_combine_final_h5s(bd, delete_extra_files=True):
    """

    Parameters
    ----------
    bd : just put your base directory, it will automatically load the pkl file and check if all the videos are processed
    if that is the case it will combine them and by default
    delete_extra_files : delete the files after combining the final one.
    Returns
    -------

    """
    finished_sessions = get_files(bd, '*file_list_for_batch_processing.pkl')
    for f in finished_sessions:
        file_dict = load_obj(f)
        if np.all(file_dict['is_processed'] == True):
            h5_file_list_to_combine = os_sorted(get_files(os.path.dirname(f), '*_final_to_combine_*'))
            if len(h5_file_list_to_combine) > 0:
                combine_final_h5s(h5_file_list_to_combine, delete_extra_files=delete_extra_files)


def combine_final_h5s(h5_file_list_to_combine, delete_extra_files=False):
    keys = ['file_name_nums', 'final_3095_features', 'frame_nums', 'full_file_names', 'in_range', 'labels',
            'locations_x_y', 'max_val_stack']
    fn = h5_file_list_to_combine[0].split('final')[0] + 'final_combined.h5'
    trial_nums_and_frame_nums = []
    for k in h5_file_list_to_combine:
        trial_nums_and_frame_nums.append(image_tools.get_h5_key_and_concatenate(k, 'trial_nums_and_frame_nums'))
    trial_nums_and_frame_nums = np.hstack(trial_nums_and_frame_nums)

    with h5py.File(fn, 'w') as h:
        for k in keys:
            out = image_tools.get_h5_key_and_concatenate(h5_file_list_to_combine, k)
            h[k] = out
        h['template_img'] = image_tools.get_h5_key_and_concatenate(h5_file_list_to_combine[0], 'template_img')
        h['multiplier'] = image_tools.get_h5_key_and_concatenate(h5_file_list_to_combine[0], 'multiplier')
        h['trial_nums_and_frame_nums'] = trial_nums_and_frame_nums
    if delete_extra_files:
        for k in h5_file_list_to_combine:
            os.remove(k)


def make_mp4_list_dict(video_directory, overwrite=False):
    fn = video_directory + os.sep + 'file_list_for_batch_processing.pkl'
    if os.path.isfile(fn):
        warnings.warn("warning file already exists! if you overwrite a partially processed directory, you will " \
                          "experience issues like overwrite errors, and you'll lose your progress. if you are sure you want to overwrite " \
                          "make sure to delete the corresponding '_FINISHED' directory, and if necessary move the mp4s back "\
                          "to the processing folder and set overwrite = True")
        warnings.warn("this above message is in reference to the following directory...\n"+video_directory)
        print(video_directory)

        # assert overwrite, "warning file already exists! if you overwrite a partially processed directory, you will " \
        #                   "experience issues like overwrite errors, and you'll lose your progress if you are sure you want to overwrite " \
        #                   "make sure to delete the corresponding '_FINISHED' directory  and set overwrite = True"
    tmpd = dict()
    tmpd['original_mp4_directory'] = video_directory
    tmpd['mp4_names'] = os_sorted(glob.glob(video_directory + '/*.mp4'))
    tmpd['is_processed'] = np.full(np.shape(tmpd['mp4_names']), False)
    tmpd['NOTES'] = """you can put any notes here directly from the text file if you want to"""
    save_obj(tmpd, fn)


def _check_pkl(name):
    if name[-4:] != '.pkl':
        return name + '.pkl'
    return name


def save_obj(obj, name):
    with open(_check_pkl(name), 'wb') as f:
        # pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(obj, f, protocol=4)


def load_obj(name):
    with open(_check_pkl(name), 'rb') as f:
        return pickle.load(f)


def get_whacc_path():
    path = os.path.dirname(whacc.__file__)
    return path


def load_feature_data():
    x = get_whacc_path() + "/whacc_data/feature_data/"
    d = load_obj(x + 'feature_data_dict.pkl')
    x = d['features_used_of_10'][:]
    d['features_used_of_10_bool'] = [True if k in x else False for k in range(2048 * 41 + 41)]
    return d


# def load_top_feature_selection_out_of_ten():
#     fn = get_whacc_path() + "/whacc_data/features_used_in_light_GBM_mods_out_of_10.npy"
#     return np.load(fn)

def get_selected_features(greater_than_or_equal_to=4):
    '''
    Parameters
    ----------
    greater_than_or_equal_to : 0 means select all features, 10 means only the features that were use in EVERY test
    light GBM model. Note: the save Light GBM (model) is trained on greater_than_or_equal_to = 4, so you can change this
    but you will need to retrain the light GBM (model).
    Returns keep_features_index : index to the giant '84,009' features, note greater_than_or_equal_to = 4 return 3095
    features
    -------
    '''

    fd = load_feature_data()
    features_out_of_10 = fd['features_used_of_10']
    keep_features_index = np.where(features_out_of_10 >= greater_than_or_equal_to)[0]
    return keep_features_index


def isnotebook():
    try:
        c = str(get_ipython().__class__)
        shell = get_ipython().__class__.__name__
        if 'colab' in c:
            return True
        elif shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


# if tqdm_import_helper():
#     from tqdm.notebook import tqdm
# else:
#     from tqdm import tqdm


def four_class_labels_from_binary(x):
    a = np.asarray(x)
    b = np.asarray([0] + list(np.diff(a)))
    c = a + b
    c[c == -1] = 3
    return c


def print_h5_keys(h5file, return_list=False, do_print=True):
    with h5py.File(h5file, 'r') as h:
        x = copy.deepcopy(list(h.keys()))
        if do_print:
            print_list_with_inds(x)
        if return_list:
            return x


def copy_h5_key_to_another_h5(h5_to_copy_from, h5_to_copy_to, label_string_to_copy_from, label_string_to_copy_to=None):
    if label_string_to_copy_to is None:
        label_string_to_copy_to = label_string_to_copy_from
    with h5py.File(h5_to_copy_from, 'r') as h:
        with h5py.File(h5_to_copy_to, 'r+') as h2:
            try:
                h2[label_string_to_copy_to][:] = h[label_string_to_copy_from][:]
            except:
                h2.create_dataset(label_string_to_copy_to, shape=np.shape(h[label_string_to_copy_from][:]),
                                  data=h[label_string_to_copy_from][:])


def lister_it(in_list, keep_strings='', remove_string=None, return_bool_index=False):
    if len(in_list) == 0:
        print("in_list was empty, returning in_list")
        return in_list

    def index_list_of_strings(in_list2, cmp_string):
        return np.asarray([cmp_string in string for string in in_list2])

    if isinstance(keep_strings, str): keep_strings = [keep_strings]
    if isinstance(remove_string, str): remove_string = [remove_string]

    keep_i = np.asarray([False] * len(in_list))
    for k in keep_strings:
        keep_i = np.vstack((keep_i, index_list_of_strings(in_list, k)))
    keep_i = np.sum(keep_i, axis=0) > 0

    remove_i = np.asarray([True] * len(in_list))
    if remove_string is not None:
        for k in remove_string:
            remove_i = np.vstack((remove_i, np.invert(index_list_of_strings(in_list, k))))
        remove_i = np.product(remove_i, axis=0) > 0

    inds = keep_i * remove_i  # np.invert(remove_i)
    if inds.size <= 0:
        return []
    else:
        out = np.asarray(in_list)[inds]
        if return_bool_index:
            return out, inds
    return out


# def lister_it(in_list, keep_strings=None, remove_string=None):
#     """
#
#     Parameters
#     ----------
#     in_list : list
#     keep_strings : list
#     remove_string : list
#
#     Returns
#     -------
#
#     """
#     if isinstance(keep_strings, str):
#         keep_strings = [keep_strings]
#     if isinstance(remove_string, str):
#         remove_string = [remove_string]
#
#     if keep_strings is None:
#         new_list = copy.deepcopy(in_list)
#     else:
#         new_list = []
#         for L in in_list:
#             for k in keep_strings:
#                 if k in L:
#                     new_list.append(L)
#
#     if remove_string is None:
#         new_list_2 = copy.deepcopy(in_list)
#     else:
#         new_list_2 = []
#         for L in new_list:
#             for k in remove_string:
#                 if k not in L:
#                     new_list_2.append(L)
#     final_list = intersect_lists([new_list_2, new_list])
#     return final_list


def plot_pole_tracking_max_vals(h5_file):
    with h5py.File(h5_file, 'r') as hf:
        for i, k in enumerate(hf['max_val_stack'][:]):
            plt.plot()


# def get_class_info(c, sort_by_type=True, include_underscore_vars=False, return_name_and_type=False, end_prev_len=40):
#     names = []
#     type_to_print = []
#     for k in dir(c):
#         if include_underscore_vars is False and k[0] != '_':
#             tmp1 = str(type(eval('c.' + k)))
#             type_to_print.append(tmp1.split("""'""")[-2])
#             names.append(k)
#         elif include_underscore_vars:
#             tmp1 = str(type(eval('c.' + k)))
#             type_to_print.append(tmp1.split("""'""")[-2])
#             names.append(k)
#     len_space = ' ' * max(len(k) for k in names)
#     len_space_type = ' ' * max(len(k) for k in type_to_print)
#     if sort_by_type:
#         ind_array = np.argsort(type_to_print)
#     else:
#         ind_array = np.argsort(names)
#
#     for i in ind_array:
#         k1 = names[i]
#         k2 = type_to_print[i]
#         # k3 = str(c[names[i]])
#         k3 = str(eval('c.' + names[i]))
#         k1 = (k1 + len_space)[:len(len_space)]
#         k2 = (k2 + len_space_type)[:len(len_space_type)]
#         if len(k3) > end_prev_len:
#             k3 = '...' + k3[-end_prev_len:]
#         else:
#             k3 = '> ' + k3[-end_prev_len:]
#
#         print(k1 + ' type->   ' + k2 + '  ' + k3)
#     if return_name_and_type:
#         return names, type_to_print

def get_class_info2(c, sort_by=None, include_underscore_vars=False, return_name_and_type=False, end_prev_len=40):
    def get_len_or_shape(x_in):
        which_one = None
        try:
            len_or_shape_out = str(len(x_in))
            which_one = 'length'
            if type(x_in).__module__ == np.__name__:
                len_or_shape_out = str(x_in.shape)
                which_one = 'shape '
        except:
            if which_one is None:
                len_or_shape_out = 'None'
                which_one = 'None  '
        return len_or_shape_out, which_one

    names = []
    len_or_shape = []
    len_or_shape_which_one = []
    type_to_print = []

    for k in dir(c):
        if include_underscore_vars is False and k[0] != '_':

            tmp1 = str(type(eval('c.' + k)))
            type_to_print.append(tmp1.split("""'""")[-2])
            names.append(k)
            a, b = get_len_or_shape(eval('c.' + names[-1]))
            len_or_shape.append(a)
            len_or_shape_which_one.append(b)
        elif include_underscore_vars:
            tmp1 = str(type(eval('c.' + k)))
            type_to_print.append(tmp1.split("""'""")[-2])
            names.append(k)
            a, b = get_len_or_shape(eval('c.' + names[-1]))
            len_or_shape.append(a)
            len_or_shape_which_one.append(b)
    len_space = ' ' * max(len(k) for k in names)
    len_space_type = ' ' * max(len(k) for k in type_to_print)
    len_space_shape = ' ' * max(len(k) for k in len_or_shape)
    if sort_by is None:
        ind_array = np.arange(len(names))
    elif 'type' in sort_by.lower():
        ind_array = np.argsort(type_to_print)
    elif 'len' in sort_by.lower() or 'shape' in sort_by.lower():
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        tmp1 = np.asarray([eval(k) for k in len_or_shape])
        tmp1[tmp1 == None] = np.nan
        tmp1 = [np.max(iii) for iii in tmp1]
        ind_array = np.argsort(tmp1)
    elif 'name' in sort_by.lower():
        ind_array = np.argsort(names)
    else:
        ind_array = np.arange(len(names))

    for i in ind_array:
        k1 = names[i]
        k2 = type_to_print[i]
        k5 = len_or_shape[i]
        x = eval('c.' + names[i])
        k3 = str(x)
        k1 = (k1 + len_space)[:len(len_space)]
        k2 = (k2 + len_space_type)[:len(len_space_type)]
        k5 = (k5 + len_space_shape)[:len(len_space_shape)]
        if len(k3) > end_prev_len:
            k3 = '...' + k3[-end_prev_len:]
        else:
            k3 = '> ' + k3[-end_prev_len:]
        print(k1 + ' type->   ' + k2 + '  ' + len_or_shape_which_one[i] + '->   ' + k5 + '  ' + k3)
    if return_name_and_type:
        return names, type_to_print


def get_class_info(c, sort_by_type=True, include_underscore_vars=False, return_name_and_type=False, end_prev_len=40):
    def get_len_or_shape(x_in):
        which_one = None
        try:
            len_or_shape_out = str(len(x_in))
            which_one = 'length'
            if type(x_in).__module__ == np.__name__:
                len_or_shape_out = str(x_in.shape)
                which_one = 'shape '
        except:
            if which_one is None:
                len_or_shape_out = 'None'
                which_one = 'None  '
        return len_or_shape_out, which_one

    names = []
    len_or_shape = []
    len_or_shape_which_one = []
    type_to_print = []

    for k in dir(c):
        if include_underscore_vars is False and k[0] != '_':

            tmp1 = str(type(eval('c.' + k)))
            type_to_print.append(tmp1.split("""'""")[-2])
            names.append(k)
            a, b = get_len_or_shape(eval('c.' + names[-1]))
            len_or_shape.append(a)
            len_or_shape_which_one.append(b)
        elif include_underscore_vars:
            tmp1 = str(type(eval('c.' + k)))
            type_to_print.append(tmp1.split("""'""")[-2])
            names.append(k)
            a, b = get_len_or_shape(eval('c.' + names[-1]))
            len_or_shape.append(a)
            len_or_shape_which_one.append(b)
    len_space = ' ' * max(len(k) for k in names)
    len_space_type = ' ' * max(len(k) for k in type_to_print)
    len_space_shape = ' ' * max(len(k) for k in len_or_shape)
    if sort_by_type:
        ind_array = np.argsort(type_to_print)
    else:
        ind_array = np.argsort(names)

    for i in ind_array:
        k1 = names[i]
        k2 = type_to_print[i]
        k5 = len_or_shape[i]
        x = eval('c.' + names[i])
        k3 = str(x)
        k1 = (k1 + len_space)[:len(len_space)]
        k2 = (k2 + len_space_type)[:len(len_space_type)]
        k5 = (k5 + len_space_shape)[:len(len_space_shape)]
        if len(k3) > end_prev_len:
            k3 = '...' + k3[-end_prev_len:]
        else:
            k3 = '> ' + k3[-end_prev_len:]
        print(k1 + ' type->   ' + k2 + '  ' + len_or_shape_which_one[i] + '->   ' + k5 + '  ' + k3)
    if return_name_and_type:
        return names, type_to_print


def get_dict_info(c, sort_by_type=True, include_underscore_vars=False, return_name_and_type=False, end_prev_len=40):
    names = []
    type_to_print = []
    for k in c.keys():
        if include_underscore_vars is False and str(k)[0] != '_':
            tmp1 = str(type(c[k]))
            type_to_print.append(tmp1.split("""'""")[-2])
            names.append(str(k))
        elif include_underscore_vars:
            tmp1 = str(type(c[k]))
            type_to_print.append(tmp1.split("""'""")[-2])
            names.append(str(k))
    len_space = ' ' * max(len(k) for k in names)
    len_space_type = ' ' * max(len(k) for k in type_to_print)
    if sort_by_type:
        ind_array = np.argsort(type_to_print)
    else:
        ind_array = np.argsort(names)

    for i in ind_array:
        k1 = names[i]
        k2 = type_to_print[i]
        try:
            k3 = str(c[names[i]])
        except:
            k3 = str(c[float(names[i])])
        k1 = (k1 + len_space)[:len(len_space)]
        k2 = (k2 + len_space_type)[:len(len_space_type)]
        if len(k3) > end_prev_len:
            k3 = '...' + k3[-end_prev_len:]
        else:
            k3 = '> ' + k3[-end_prev_len:]

        print(k1 + ' type->   ' + k2 + '  ' + k3)
    if return_name_and_type:
        return names, type_to_print


def group_consecutives(vals, step=1):
    """

    Parameters
    ----------
    vals :
        
    step :
         (Default value = 1)

    Returns
    -------

    """
    run = []
    run_ind = []
    result = run
    result_ind = run_ind
    expect = None
    for k, v in enumerate(vals):
        if v == expect:
            if not (np.isnan(v)):
                # print(v)
                # print(expect)
                run.append(v)
                run_ind.append(k)
        else:
            if not (np.isnan(v)):
                run = [v]
                run_ind = [k]
                result.append(run)
                result_ind.append(run_ind)
        expect = v + step
    # print(result)
    if result == []:
        pass
    elif result[0] == []:
        result = result[1:]
        result_ind = result_ind[1:]
    return result, result_ind


def get_h5s(base_dir, print_h5_list=True):
    """

    Parameters
    ----------
    base_dir :
        

    Returns
    -------

    """
    H5_file_list = []
    for path in Path(base_dir + os.path.sep).rglob('*.h5'):
        H5_file_list.append(str(path.parent) + os.path.sep + path.name)
    H5_file_list.sort()
    if print_h5_list:
        print_list_with_inds(H5_file_list)
    return H5_file_list


def check_if_file_lists_match(H5_list_LAB, H5_list_IMG):
    """

    Parameters
    ----------
    H5_list_LAB :
        
    H5_list_IMG :
        

    Returns
    -------

    """
    for h5_LAB, h5_IMG in zip(H5_list_LAB, H5_list_IMG):
        try:
            assert h5_IMG.split(os.path.sep)[-1] in h5_LAB
        except:
            print('DO NOT CONTINUE --- some files do not match on your lists try again')
            assert (1 == 0)
    print('yay they all match!')


def print_list_with_inds(list_in):
    """

    Parameters
    ----------
    list_in :
        

    Returns
    -------

    """
    _ = [print(str(i) + ' ' + k.split(os.path.sep)[-1]) for i, k in enumerate(list_in)]


def get_model_list(model_save_dir):
    """

    Parameters
    ----------
    model_save_dir :
        

    Returns
    -------

    """
    print('These are all the models to choose from...')
    model_2_load_all = glob.glob(model_save_dir + '/*.ckpt')
    print_list_with_inds(model_2_load_all)
    return model_2_load_all


def recursive_dir_finder(base_path, search_term):
    """enter base directory and search term to find all the directories in base directory
      with files matching the search_term. output a sorted list of directories.
      e.g. -> recursive_dir_finder('/content/mydropbox/', '*.mp4')

    Parameters
    ----------
    base_path :
        
    search_term :
        

    Returns
    -------

    """
    matching_folders = []
    for root, dirs, files in os.walk(base_path):
        if glob.glob(root + '/' + search_term):
            matching_folders.append(root)
    try:
        matching_folders = os_sorted(matching_folders)
    except:
        matching_folders = sorted(matching_folders)
    return matching_folders


def get_model_list(model_save_dir):
    """

    Parameters
    ----------
    model_save_dir :
        

    Returns
    -------

    """
    print('These are all the models to choose from...')
    model_2_load_all = sorted(glob.glob(model_save_dir + '/*.ckpt'))
    # print(*model_2_load_all, sep = '\n')
    _ = [print(str(i) + ' ' + k.split(os.path.sep)[-1]) for i, k in enumerate(model_2_load_all)]
    return model_2_load_all


def get_files(base_dir, search_term=''):
    """
base_dir = '/content/gdrive/My Drive/LIGHT_GBM/FEATURE_DATA/'
num_folders_deep = 1
file_list = []
for i, path in enumerate(Path(base_dir + os.sep).rglob('')):
  x = str(path.parent) + os.path.sep + path.name
  if i ==0:
    file_list.append(x)
    cnt = len(x.split(os.sep))
  if (len(x.split(os.sep))-cnt)<=num_folders_deep:
    file_list.append(x)
list(set(file_list))

    Parameters
    ----------
    base_dir :
        
    search_term :

    Returns
    -------

    """
    file_list = []
    for path in Path(base_dir + os.sep).rglob(search_term):
        ##### can I edit this with default depth of one and only look x num folders deep to prevent long searchs in main folders?
        file_list.append(str(path.parent) + os.path.sep + path.name)
    file_list.sort()
    return file_list


'''
these below 3 function used to load mat files into dict easily was found and copied directly from 
https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
and contributed by user -mergen 

to the best of my knowledge code found on stackoverflow is under the creative commons license and as such is legal to 
use in my package. contact phillip.maire@gmail.com if you have any questions. 
'''


def loadmat(filename):
    """this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    Parameters
    ----------
    filename :
        

    Returns
    -------

    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    """checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries

    Parameters
    ----------
    dict :
        

    Returns
    -------

    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """A recursive function which constructs from matobjects nested dictionaries

    Parameters
    ----------
    matobj :
        

    Returns
    -------

    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def get_inds_of_inds(a, return_unique_list=False):
    a2 = []
    for k in a:
        a2.append(list(np.where([k == kk for kk in a])[0]))
    try:
        inds_of_inds = list(np.unique(a2, axis=0))
        for i, k in enumerate(inds_of_inds):
            inds_of_inds[i] = list(k)
    except:
        inds_of_inds = list(np.unique(a2))
    if return_unique_list:
        return inds_of_inds, pd.unique(a)
    else:
        return inds_of_inds


def inds_around_inds(x, N):
    """

    Parameters
    ----------
    x : array
    N : window size

    Returns
    -------
    returns indices of arrays where array >0 with borders of ((N - 1) / 2), so x = [0, 0, 0, 1, 0, 0, 0] and N = 3
    returns [2, 3, 4]
    """
    assert N / 2 != round(N / 2), 'N must be an odd number so that there are equal number of points on each side'
    cumsum = np.cumsum(np.insert(x, 0, 0))
    a = (cumsum[N:] - cumsum[:-N]) / float(N)
    a = np.where(a > 0)[0] + ((N - 1) / 2)
    return a.astype('int')


def loop_segments(frame_num_array, returnaslist=False):
    """

    Parameters
    ----------
    frame_num_array :
    num of frames in each trial in a list
    Returns
    -------
    2 lists with the proper index for pulling those trials out one by one in a for loop
    Examples
    ________
    a3 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    frame_num_array = [4, 5]
    for i1, i2 in loop_segments(frame_num_array):
        print(a3[i1:i2])

    >>>[0, 1, 2, 3]
    >>>[4, 5, 6, 7, 8]
    """
    frame_num_array = list(frame_num_array)
    frame_num_array = [0] + frame_num_array
    frame_num_array = np.cumsum(frame_num_array)
    frame_num_array = frame_num_array.astype(int)
    if returnaslist:
        return [list(frame_num_array[:-1]), list(frame_num_array[1:])]
    else:
        return zip(list(frame_num_array[:-1]), list(frame_num_array[1:]))


##_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*##
##_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*##
##_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*##
##_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*##
##_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*##
##_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*##
##_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*##
##_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*##
##_*_*_*_ below programs users will likely not use_*_*_*_*_*_*##
##_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*##
##_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*##
##_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*##
##_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*##
##_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*##
##_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*##
##_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*##
##_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*##


def _get_human_contacts_(all_h5s):
    """
    just used to get array of contacts, not meant to be used long term
    Parameters
    ----------
    all_h5s :

    Returns
    -------

    """
    h_cont = []
    a = [k.split(os.path.sep)[-1].split('_', maxsplit=1)[-1] for k in all_h5s]
    inds_of_inds, list_of_uniq_files = get_inds_of_inds(a, True)
    for i, k in enumerate(inds_of_inds):
        tmp1 = np.array([])
        for ii, kk in enumerate(k):
            with h5py.File(all_h5s[kk], 'r') as h:
                tmp1 = np.vstack([tmp1, h['labels'][:]]) if tmp1.size else h['labels'][:]
        h_cont.append(tmp1)
    return h_cont, list_of_uniq_files


def create_master_dataset(h5c, all_h5s_imgs, h_cont, borders=80, max_pack_val=100):
    """
    Parameters
    ----------
    h5c : h5 creator class
    all_h5s_imgs : list of h5s with images
    h_cont : a tensor of human contacts people by frames trial H5 files (down right deep)
    borders : for touch 0000011100000 it will find the 111 in it and get all the bordering areas around it. points are
    unique so 0000011100000 and 0000010100000 will return the same index
    max_pack_val : speeds up the process by transferring data in chunks of this max size instead of building them all up in memory
    it's a max instead of a set value because it can be if the len(IMAGES)%max_pack_val is equal to 0 or 1 it will crash, so I calculate it
    so that it wont crash.

    Returns
    -------create_master_dataset
    saves an H5 file
    Examples
    ________

    all_h5s = utils.get_h5s('/content/gdrive/My Drive/Colab data/curation_for_auto_curator/finished_contacts/')
    all_h5s_imgs = utils.get_h5s('/content/gdrive/My Drive/Colab data/curation_for_auto_curator/H5_data/')
    h_cont = utils._get_human_contacts_(all_h5s)
    h5c = image_tools.h5_iterative_creator('/content/gdrive/My Drive/Colab data/curation_for_auto_curator/test_____.h5',
                                           overwrite_if_file_exists = True,
                                           color_channel = False)
    utils.create_master_dataset(h5c, all_h5s_imgs, h_cont, borders = 80, max_pack_val = 100)
    """
    frame_nums = []
    for k, k2 in zip(all_h5s_imgs, h_cont):
        if len(k2.shape) == 1:
            k2 = np.vstack((k2, k2))
        with h5py.File(k, 'r') as h:
            max_human_label = np.max(k2, axis=0)
            mean_human_label = np.mean(k2, axis=0)
            mean_human_label = (mean_human_label > 0.5) * 1
            b = inds_around_inds(max_human_label, borders * 2 + 1)
            tmp1, _ = group_consecutives(b)
            for tmp2 in tmp1:
                frame_nums.append(len(tmp2))

            pack_every_x = [k for k in np.flip(range(3, max_pack_val + 1)) if len(b) % k >= 2]
            assert pack_every_x, ['chosen H5 file has value of ', len(b), ' and max_pack_val is ', max_pack_val,
                                  ' increase max_pack_val to prevent this error']
            pack_every_x = np.max(pack_every_x)

            # np.max([k for k in np.flip(range(3, 100)) if len(b) % k >= 2])
            new_imgs = np.array([])
            new_labels = np.array([])
            cntr = 0
            for k3 in tqdm(b):
                cntr += 1
                if new_imgs.size:
                    new_imgs = np.concatenate((new_imgs, h['images'][k3][None, :, :, 0]), axis=0)
                    new_labels = np.append(new_labels, max_human_label[k3])
                else:
                    new_imgs = h['images'][k3][None, :, :, 0]
                    new_labels = mean_human_label[k3]
                if cntr >= pack_every_x:  # this makes it ~ 100X faster than stacking up in memory
                    h5c.add_to_h5(new_imgs, new_labels)
                    new_imgs = np.array([])
                    new_labels = np.array([])
                    cntr = 0
            h5c.add_to_h5(new_imgs, new_labels)

    with h5py.File(h5c.h5_full_file_name, 'r+') as h:
        h.create_dataset('frame_nums', shape=np.shape(frame_nums), data=frame_nums)
        h.create_dataset('inds_extracted', shape=np.shape(b), data=b)


def get_time_it(txt):
    """
  Example
  -------
  import re
  import matplotlib.pyplot as plt
  for k in range(8):
    S = 'test '*10**k
    s2 = 'test'
    %time [m.start() for m in re.finditer(S2, S)]

  # then copy it into a string like below

  txt = '''
  CPU times: user 0 ns, sys: 9.05 ms, total: 9.05 ms
  Wall time: 9.01 ms
  CPU times: user 12 µs, sys: 1 µs, total: 13 µs
  Wall time: 15.7 µs
  CPU times: user 48 µs, sys: 3 µs, total: 51 µs
  Wall time: 56.3 µs
  CPU times: user 281 µs, sys: 0 ns, total: 281 µs
  Wall time: 287 µs
  CPU times: user 2.42 ms, sys: 0 ns, total: 2.42 ms
  Wall time: 2.43 ms
  CPU times: user 21.8 ms, sys: 22 µs, total: 21.8 ms
  Wall time: 21.2 ms
  CPU times: user 198 ms, sys: 21.5 ms, total: 219 ms
  Wall time: 214 ms
  CPU times: user 1.83 s, sys: 191 ms, total: 2.02 s
  Wall time: 2.02 s
  '''
  data = get_time_it(txt)
  ax = plt.plot(data[1:])
  plt.yscale('log')
  """
    vars = [k.split('\n')[0] for k in txt.split('Wall time: ')[1:]]
    a = dict()
    a['s'] = 10 ** 0
    a['ms'] = 10 ** -3
    a['µs'] = 10 ** -6
    data = []
    for k in vars:
        units = k.split(' ')[-1]
        data.append(float(k.split(' ')[0]) * a[units])
    return data


def save_what_is_left_of_your_h5_file(H5_file, do_del_and_rename=0):
    tst_cor = []
    with h5py.File(H5_file, 'a') as hf:
        for k in hf.keys():
            if hf.get(k):
                tst_cor.append(0)
            else:
                tst_cor.append(1)
        if any(tst_cor):
            print('Corrupt file found, creating new file')
            H5_fileTMP = H5_file + 'TMP'
            with h5py.File(H5_file + 'TMP', 'w') as hf2:
                for k in hf.keys():
                    if hf.get(k):
                        print('Adding key ' + k + ' to new temp H5 file...')
                        hf2.create_dataset(k, data=hf[k])
                    else:
                        print('***Key ' + k + ' was corrupt, skipping this key...')
                hf2.close()
                hf.close()
            if do_del_and_rename:
                print('Deleting corrupt H5 file')
                os.remove(H5_file)
                print('renaming new h5 file ')
                os.rename(H5_fileTMP, H5_file)
        else:
            print('File is NOT corrupt!')
    print('FINISHED')


def stack_lag_h5_maker(f, f2, buffer=2, shift_to_the_right_by=0):
    """

    Parameters
    ----------
    f : h5 file with SINGLE FRAMES this is meant to be a test program. if used long term I will change this part
    f2 :
    buffer :
    shift_to_the_right_by :

    Returns
    -------

    """
    h5c = image_tools.h5_iterative_creator(f2, overwrite_if_file_exists=True)
    with h5py.File(f, 'r') as h:
        x = h['frame_nums'][:]
        for ii, (k1, k2) in enumerate(tqdm(loop_segments(x), total=len(x))):
            x2 = h['images'][k1:k2]
            if len(x2.shape) == 4:
                x2 = x2[:, :, :, 0]  # only want one 'color' channel
            new_imgs = image_tools.stack_imgs_lag(x2, buffer=2, shift_to_the_right_by=0)
            h5c.add_to_h5(new_imgs, np.ones(new_imgs.shape[0]) * -1)

    # copy over the other info from the OG h5 file
    copy_h5_key_to_another_h5(f, f2, 'labels', 'labels')
    copy_h5_key_to_another_h5(f, f2, 'frame_nums', 'frame_nums')


def diff_lag_h5_maker(f3):
    """
    need to use the stack_lag_h5_maker first and then send a copy of that into this one again these program are only a temp
    solution, if we use these methods for the main model then I will make using them more fluid and not depend on one another
    Parameters
    ----------
    f3 : the file from stack_lag_h5_maker output

    Returns
    -------
    """
    # change color channel 0 and 1 to diff images from color channel 3 so color channels 0, 1, and 2 are 0-2, 1-2, and 2
    with h5py.File(f3, 'r+') as h:
        for i in tqdm(range(h['images'].shape[0])):
            k = copy.deepcopy(h['images'][i])
            for img_i in range(2):
                k = k.astype(float)
                a = k[:, :, img_i] - k[:, :, -1]
                a = ((a + 255) / 2).astype(np.uint8)
                h['images'][i, :, :, img_i] = a


def expand_single_frame_to_3_color_h5(f, f2):
    h5c = image_tools.h5_iterative_creator(f2, overwrite_if_file_exists=True)
    with h5py.File(f, 'r') as h:
        x = h['frame_nums'][:]
        for ii, (k1, k2) in enumerate(tqdm(loop_segments(x), total=len(x))):
            new_imgs = h['images'][k1:k2]
            new_imgs = np.tile(new_imgs[:, :, :, None], 3)
            h5c.add_to_h5(new_imgs, np.ones(new_imgs.shape[0]) * -1)

    # copy over the other info from the OG h5 file
    copy_over_all_non_image_keys(f, f2)
    # copy_h5_key_to_another_h5(f, f2, 'labels', 'labels')
    # copy_h5_key_to_another_h5(f, f2, 'frame_nums', 'frame_nums')


def copy_over_all_non_image_keys(f, f2):
    """

    Parameters
    ----------
    f : source
    f2 : destination

    Returns
    -------

    """
    k_names = print_h5_keys(f, return_list=True, do_print=False)
    k_names = lister_it(k_names, remove_string='MODEL_')
    k_names = lister_it(k_names, remove_string='images')
    with h5py.File(f, 'r') as h:
        with h5py.File(f2, 'r+') as h2:
            for kn in k_names:
                try:
                    h2.create_dataset(kn, data=h[kn])
                except:
                    del h2[kn]
                    h2.create_dataset(kn, data=h[kn])
            # try:
            #     copy_h5_key_to_another_h5(f, f2, kn, kn)
            # except:
            #     del h[kn]
            #     # time.sleep(2)
            #     copy_h5_key_to_another_h5(f, f2, kn, kn)
    if 'frame_nums' not in k_names and 'trial_nums_and_frame_nums' in k_names:
        tnfn = image_tools.get_h5_key_and_concatenate([f], 'trial_nums_and_frame_nums')
        with h5py.File(f2, 'r+') as h:
            try:
                h.create_dataset('frame_nums', data=(tnfn[1, :]).astype(int))
            except:
                del h['frame_nums']
                time.sleep(2)
                h.create_dataset('frame_nums', data=(tnfn[1, :]).astype(int))


def force_write_to_h5(h5_file, data, data_name):
    with h5py.File(h5_file, 'r+') as h:
        try:
            h.create_dataset(data_name, data=data)
        except:
            del h[data_name]
            h.create_dataset(data_name, data=data)


def reduce_to_single_frame_from_color(f, f2):
    h5c = image_tools.h5_iterative_creator(f2, overwrite_if_file_exists=True, color_channel=False)
    with h5py.File(f, 'r') as h:
        try:
            x = h['trial_nums_and_frame_nums'][1, :]
        except:
            x = h['frame_nums'][:]
        for ii, (k1, k2) in enumerate(tqdm(loop_segments(x), total=len(x))):
            new_imgs = h['images'][k1:k2][..., -1]
            # new_imgs = np.tile(new_imgs[:, :, :, None], 3)
            h5c.add_to_h5(new_imgs, np.ones(new_imgs.shape[0]) * -1)

    # copy over the other info from the OG h5 file
    copy_over_all_non_image_keys(f, f2)
    # copy_h5_key_to_another_h5(f, f2, 'labels', 'labels')
    # copy_h5_key_to_another_h5(f, f2, 'frame_nums', 'frame_nums')


def make_all_H5_types(base_dir_all_h5s):
    def last_folder(f):
        tmp1 = str(Path(f).parent.absolute())
        return str(Path(tmp1).parent.absolute()) + os.sep

    for f in get_h5s(base_dir_all_h5s):
        base_f = last_folder(f)
        basename = os.path.basename(f)[:-3] + '_regular.h5'
        basedir = base_f + 'regular' + os.sep
        if not os.path.isdir(basedir):
            os.mkdir(basedir)
        f2 = basedir + basename
        expand_single_frame_to_3_color_h5(f, f2)

        basename = os.path.basename(f)[:-3] + '_3lag.h5'
        basedir = base_f + '3lag' + os.sep
        if not os.path.isdir(basedir):
            os.mkdir(basedir)
        f2 = basedir + basename

        stack_lag_h5_maker(f, f2, buffer=2, shift_to_the_right_by=0)

        basename = os.path.basename(f2)[:-3] + '_diff.h5'
        basedir = base_f + '3lag_diff' + os.sep
        if not os.path.isdir(basedir):
            os.mkdir(basedir)
        f3 = basedir + basename
        shutil.copy(f2, f3)
        diff_lag_h5_maker(f3)


def get_all_label_types_from_array(array):
    all_labels = []
    x1 = copy.deepcopy(array)  # [0, 1]- (no touch, touch)
    all_labels.append(x1)

    x2 = four_class_labels_from_binary(x1)  # [0, 1, 2, 3]- (no touch, touch, onset, offset)
    all_labels.append(x2)

    x3 = copy.deepcopy(x2)
    x3[x3 != 2] = 0
    x3[x3 == 2] = 1  # [0, 1]- (not onset, onset)
    all_labels.append(x3)

    x4 = copy.deepcopy(x2)  # [0, 1]- (not offset, offset)
    x4[x4 != 3] = 0
    x4[x4 == 3] = 1
    all_labels.append(x4)

    x5 = copy.deepcopy(x2)  # [0, 1, 2]- (no event, onset, offset)
    x5[x5 == 1] = 0
    x5[x5 == 2] = 1
    x5[x5 == 3] = 2
    all_labels.append(x5)

    x6 = copy.deepcopy(x2)  # [0, 1, 2]- (no event, onset, offset)
    onset_inds = x6[:-1] == 2
    bool_inds_one_after_onset = np.append(False, onset_inds)
    offset_inds = x6[:-1] == 3
    bool_inds_one_after_offset = np.append(False, offset_inds)
    offset_inds = x6 == 3
    x6[bool_inds_one_after_onset] = 3
    x6[offset_inds] = 4
    x6[bool_inds_one_after_offset] = 5
    all_labels.append(x6)

    x7 = copy.deepcopy(x6)
    x7[x7 == 2] = 0
    x7[x7 == 5] = 0
    x7[x7 == 3] = 2
    x7[x7 == 4] = 3
    all_labels.append(x7)

    resort = [5, 1, 4, 0, 3, 2, 6]
    # resort = range(len(resort))
    a_final = []
    for i in resort:
        a_final.append(all_labels[i])

    return np.asarray(a_final)


def make_alt_labels_h5s(base_dir_all_h5s):
    for f in get_h5s(base_dir_all_h5s):
        basename = '_ALT_LABELS.'.join(os.path.basename(f).split('.'))
        basedir = os.sep.join(f.split(os.sep)[:-2]) + os.sep + 'ALT_LABELS' + os.sep
        if not os.path.isdir(basedir):
            os.mkdir(basedir)
        new_h5_name = basedir + basename

        with h5py.File(f, 'r') as h:

            x1 = copy.deepcopy(h['labels'][:])  # [0, 1]- (no touch, touch)

            x2 = four_class_labels_from_binary(x1)  # [0, 1, 2, 3]- (no touch, touch, onset, offset)

            x3 = copy.deepcopy(x2)
            x3[x3 != 2] = 0
            x3[x3 == 2] = 1  # [0, 1]- (not onset, onset)

            x4 = copy.deepcopy(x2)  # [0, 1]- (not offset, offset)
            x4[x4 != 3] = 0
            x4[x4 == 3] = 1

            x5 = copy.deepcopy(x2)  # [0, 1, 2]- (no event, onset, offset)
            x5[x5 == 1] = 0
            x5[x5 == 2] = 1
            x5[x5 == 3] = 2

            x6 = copy.deepcopy(x2)  # [0, 1, 2]- (no event, onset, offset)
            onset_inds = x6[:-1] == 2
            bool_inds_one_after_onset = np.append(False, onset_inds)
            offset_inds = x6[:-1] == 3
            bool_inds_one_after_offset = np.append(False, offset_inds)
            offset_inds = x6 == 3
            x6[bool_inds_one_after_onset] = 3
            x6[offset_inds] = 4
            x6[bool_inds_one_after_offset] = 5

            x7 = copy.deepcopy(x6)
            x7[x7 == 2] = 0
            x7[x7 == 5] = 0
            x7[x7 == 3] = 2
            x7[x7 == 4] = 3

        with h5py.File(new_h5_name, 'w') as h:
            h.create_dataset('[0, 1]- (no touch, touch)', shape=np.shape(x1), data=x1)
            h.create_dataset('[0, 1, 2, 3]- (no touch, touch, onset, offset', shape=np.shape(x2), data=x2)
            h.create_dataset('[0, 1]- (not onset, onset)', shape=np.shape(x3), data=x3)
            h.create_dataset('[0, 1]- (not offset, offset)', shape=np.shape(x4), data=x4)
            h.create_dataset('[0, 1, 2]- (no event, onset, offset)', shape=np.shape(x5), data=x5)
            h.create_dataset('[0, 1, 2, 3, 4, 5]- (no touch, touch, onset, one after onset, offset, one after offset)',
                             shape=np.shape(x6), data=x6)
            h.create_dataset('[0, 1, 2, 3]- (no touch, touch, one after onset, offset)', shape=np.shape(x7), data=x7)


def intersect_lists(d):
    return list(set(d[0]).intersection(*d))


def get_in_range(H5_list, pole_up_add=200, pole_down_add=0, write_to_h5=True, return_in_range=False):
    all_in_range = []
    for k in H5_list:
        with h5py.File(k, 'r+') as hf:
            new_in_range = np.zeros_like(hf['in_range'][:])
            fn = hf['trial_nums_and_frame_nums'][1, :]
            for i, (i1, i2) in enumerate(loop_segments(fn)):
                x = hf['pole_times'][:, i] + i1
                x1 = x[0] + pole_up_add
                x2 = x[1] + pole_down_add
                x2 = min([x2, i2])
                new_in_range[x1:x2] = 1
            if write_to_h5:
                hf['in_range'][:] = new_in_range
            if return_in_range:
                all_in_range.append(new_in_range)
    if return_in_range:
        return all_in_range


def define_in_range(h5_file, pole_up_set_time=0, pole_down_add_to_trigger=0, write_to_h5=True, return_in_range=False):
    with h5py.File(h5_file, 'r+') as hf:
        new_in_range = np.zeros_like(hf['in_range'][:])
        fn = hf['trial_nums_and_frame_nums'][1, :]
        for i, (i1, i2) in enumerate(loop_segments(fn)):
            x = hf['pole_times'][:, i] + i1
            x1 = i1 + pole_up_set_time
            x2 = x[1] + pole_down_add_to_trigger
            x2 = min([x2, i2])
            new_in_range[x1:x2] = 1
        if write_to_h5:
            hf['in_range'][:] = new_in_range
        if return_in_range:
            return new_in_range


def add_to_h5(h5_file, key, values, overwrite_if_exists=False):
    all_keys = print_h5_keys(h5_file, return_list=True, do_print=False)
    with h5py.File(h5_file, 'r+') as h:
        if key in all_keys and overwrite_if_exists:
            print('key already exists, overwriting value...')
            del h[key]
            h.create_dataset(key, data=values)
        elif key in all_keys and not overwrite_if_exists:
            print("""key already exists, NOT overwriting value..., \nset 'overwrite_if_exists' to True to overwrite""")
        else:
            h.create_dataset(key, data=values)


def bool_pred_to_class_pred_formating(pred):
    pred = pred.flatten()
    zero_pred = np.ones_like(pred) - pred
    x = np.vstack((zero_pred, pred)).T
    return x


def convert_labels_back_to_binary(b, key):
    a = copy.deepcopy(b)
    """
  a is 'bool' array (integers not float predicitons)
  key is the key name of the type of labels being inserted
  can use the key name or the string in the key either will do.
  """
    name_dict = whacc.model_maker.label_naming_shorthand_dict()
    keys = list(name_dict.keys())
    if key == keys[0] or key == name_dict[keys[0]]:
        a[a >= 4] = 0
        a[a >= 2] = 1

    elif key == keys[1] or key == name_dict[keys[1]]:
        a[a >= 3] = 0
        a[a >= 2] = 1

    elif key == keys[2] or key == name_dict[keys[2]]:
        print("""can not convert type """ + key + ' returning None')
        return None
    elif key == keys[3] or key == name_dict[keys[3]]:
        print('already in the correct format')
    elif key == keys[4] or key == name_dict[keys[4]]:
        print("""can not convert type """ + key + ' returning None')
        return None
    elif key == keys[5] or key == name_dict[keys[5]]:
        print("""can not convert type """ + key + ' returning None')
        return None
    elif key == keys[6] or key == name_dict[keys[6]]:
        a[a >= 3] = 0
        one_to_the_left_inds = a == 2
        one_to_the_left_inds = np.append(one_to_the_left_inds[1:], False)
        a[one_to_the_left_inds] = 1
        a[a == 2] = 1
    else:
        raise ValueError("""key does not match. invalid key --> """ + key)
    return a


def update_whacc():
    x = '''python3 "/Users/phil/Dropbox/UPDATE_WHACC_PYPI.py"'''
    out = os.popen(x).read()
    print(out)
    print('ALL DONE')


def make_list(x, suppress_warning=False):
    if not isinstance(x, list):
        if not suppress_warning:
            print("""input is supposed to be a list, converting it but user should do this to suppress this warning""")
        x2 = [x]
        return x2
    else:
        return x


def search_sequence_numpy(arr, seq, return_type='indices'):
    Na, Nseq = arr.size, seq.size
    r_seq = np.arange(Nseq)
    M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

    if return_type == 'indices':
        return np.where(M)[0]
    elif return_type == 'bool':
        return M


def find_trials_with_suspicious_predictions(frame_nums, pred_bool, tmp_weights=[3, 3, 2, 1]):
    all_lens = []
    bins = len(tmp_weights) + 1
    for i, (k1, k2) in enumerate(loop_segments(frame_nums)):
        vals = pred_bool[k1:k2]
        a, b = group_consecutives(vals, step=0)
        y, x = np.histogram([len(k) for k in a], np.linspace(1, bins, bins))
        all_lens.append(y)
    all_lens = np.asarray(all_lens)

    all_lens = all_lens * np.asarray(tmp_weights)
    sorted_worst_estimated_trials = np.flip(np.argsort(np.nanmean(all_lens, axis=1)))
    return sorted_worst_estimated_trials


def open_folder(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def medfilt_confidence_scores(pred_bool_in, kernel_size_in):
    if pred_bool_in.shape[1] == 1:
        pred_bool_out = medfilt(copy.deepcopy(pred_bool_in), kernel_size=kernel_size_in)
    else:
        pred_bool_out = medfilt2d(copy.deepcopy(pred_bool_in), kernel_size=[kernel_size_in, 1])
    return pred_bool_out


def confidence_score_to_class(pred_bool_in, thresh_in=0.5):
    if pred_bool_in.shape[1] == 1:
        pred_bool_out = ((pred_bool_in > thresh_in) * 1).flatten()
    else:
        pred_bool_out = np.argmax(pred_bool_in, axis=1)
    #     NOTE: threshold is not used for the multi class models
    return pred_bool_out


def process_confidence_scores(pred_bool_in, key_name_in, thresh_in=0.5, kernel_size_in=1):
    pred_bool_out = medfilt_confidence_scores(pred_bool_in, kernel_size_in)
    pred_bool_out = confidence_score_to_class(pred_bool_out, thresh_in)
    L_key_ = '_'.join(key_name_in.split('__')[3].split(' '))
    pred_bool_out = convert_labels_back_to_binary(pred_bool_out, L_key_)
    return pred_bool_out


def copy_folder_structure(src, dst):
    src = os.path.abspath(src)
    src_prefix = len(src) + len(os.path.sep)
    Path(dst).mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        for dirname in dirs:
            dirpath = os.path.join(dst, root[src_prefix:], dirname)
            Path(dirpath).mkdir(parents=True, exist_ok=True)


def copy_file_filter(src, dst, keep_strings='', remove_string=None, overwrite=False,
                     just_print_what_will_be_copied=False, disable_tqdm=False, return_list_of_files=False):
    """

    Parameters
    ----------
    return_list_of_files :
    src : source folder
    dst : destination folder
    keep_strings : see utils.lister_it, list of strings to match in order to copy
    remove_string : see utils.lister_it, list of strings to match in order to not copy
    overwrite : will overwrite files if true
    just_print_what_will_be_copied : can just print what will be copied to be sure it is correct
    disable_tqdm : if True it will prevent the TQDM loading bar

    Examples
    ________
    copy_file_filter('/Users/phil/Desktop/FAKE_full_data', '/Users/phil/Desktop/aaaaaaaaaa', keep_strings='/3lag/',
                 remove_string=None, overwrite=True, just_print_what_will_be_copied=False)
    Returns
    -------

    """
    src = src.rstrip(os.sep) + os.sep
    dst = dst.rstrip(os.sep) + os.sep

    all_files_and_dirs = get_files(src, search_term='*')
    to_copy = lister_it(all_files_and_dirs, keep_strings=keep_strings, remove_string=remove_string)

    if just_print_what_will_be_copied:
        _ = [print(str(i) + ' ' + k) for i, k in enumerate(to_copy)]
        if return_list_of_files:
            return to_copy, None
        else:
            return

    to_copy2 = []  # this is so I can tqdm the files and not the folders which would screw with the average copy time.
    for k in to_copy:
        k2 = dst.join(k.split(src))
        if os.path.isdir(k):
            Path(k2).mkdir(parents=True, exist_ok=True)
        else:
            to_copy2.append(k)
    final_copied = []
    for k in tqdm(to_copy2, disable=disable_tqdm):
        k2 = dst.join(k.split(src))
        final_copied.append(k2)
        if overwrite or not os.path.isfile(k2):
            if os.path.isfile(k2):
                os.remove(k2)
            Path(os.path.dirname(k2)).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(k, k2)
        elif not overwrite:
            print('overwrite = False: file exists, skipping--> ' + k2)
    if return_list_of_files:
        return to_copy2, final_copied


def copy_alt_labels_based_on_directory(file_list, alt_label_folder_name='ALT_LABELS'):
    h5_imgs = []
    h5_labels = []
    inds_of_files = []
    for i, k in enumerate(file_list):
        k = k.rstrip(os.sep)
        if os.path.isfile(k):
            fn = os.path.basename(k)
            alt_labels_dir = os.sep.join(k.split(os.sep)[:-2]) + os.sep + alt_label_folder_name + os.sep
            h5_list = get_h5s(alt_labels_dir, 0)
            if 'train' in fn.lower():
                h5_list = lister_it(h5_list, keep_strings='train')
            elif 'val' in fn.lower():
                h5_list = lister_it(h5_list, keep_strings='val')
            if len(h5_list) == 1:
                h5_imgs.append(k)
                h5_labels.append(h5_list[0])
                inds_of_files.append(i)
            else:
                print('File name ' + fn + ' could not find valid match')
                break
    return h5_imgs, h5_labels, inds_of_files


def np_stats(in_arr):
    print('\nmin', np.min(in_arr))
    print('max', np.max(in_arr))
    print('mean', np.mean(in_arr))
    print('shape', in_arr.shape)
    print('len of unique', len(np.unique(in_arr)))
    print('type', type(in_arr))
    try:
        print('Dtype ', in_arr.dtype)
    except:
        pass


def h5_key_exists(h5_in, key_in):
    return key_in in print_h5_keys(h5_in, return_list=True, do_print=False)


def overwrite_h5_key(h5_in, key_in, new_data=None):
    exist_test = h5_key_exists(h5_in, key_in)
    with h5py.File(h5_in, 'r+') as h:
        if exist_test:
            del h[key_in]
        if new_data is not None:
            h[key_in] = new_data


def convert_list_of_strings_for_h5(list_in):
    return [n.encode("ascii", "ignore") for n in list_in]


def intersect_all(arr1, arr2):
    """retun inndex of length len(arr1) instead of numpys length min([len(arr1), len(arr2)])"""
    return [{v: i for i, v in enumerate(arr2)}[v] for v in arr1]


def space_check(path, min_gb=2):
    assert shutil.disk_usage(
        path).free / 10 ** 9 > min_gb, """space_check function: GB limit reached, ending function"""

# "/whacc_data/final_CNN_model_weights/*.hdf5"
#
# git lfs track "whacc_data/*"
