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


if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


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

def lister_it(in_list, keep_strings='', remove_string=None):
    def index_list_of_strings(in_list2, cmp_string):
        return np.asarray([cmp_string in string for string in in_list2])

    if isinstance(keep_strings, str): keep_strings = [keep_strings]
    if isinstance(remove_string, str): remove_string = [remove_string]

    keep_i = np.asarray([False]*len(in_list))
    for k in keep_strings:
        keep_i = np.vstack((keep_i, index_list_of_strings(in_list, k)))
    keep_i = np.sum(keep_i, axis = 0)>0

    remove_i = np.asarray([True]*len(in_list))
    if remove_string is not None:
        for k in remove_string:
            remove_i = np.vstack((remove_i, np.invert(index_list_of_strings(in_list, k))))
        remove_i = np.product(remove_i, axis = 0)>0

    inds = keep_i * remove_i#np.invert(remove_i)
    if inds.size <= 0:
        return []
    else:
        out = np.asarray(in_list)[inds]
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


def get_class_info(c, sort_by_type=True, include_underscore_vars=False, return_name_and_type=False, end_prev_len=40):
    names = []
    type_to_print = []
    for k in dir(c):
        if include_underscore_vars is False and k[0] != '_':
            tmp1 = str(type(eval('c.' + k)))
            type_to_print.append(tmp1.split("""'""")[-2])
            names.append(k)
        elif include_underscore_vars:
            tmp1 = str(type(eval('c.' + k)))
            type_to_print.append(tmp1.split("""'""")[-2])
            names.append(k)
    len_space = ' ' * max(len(k) for k in names)
    len_space_type = ' ' * max(len(k) for k in type_to_print)
    if sort_by_type:
        ind_array = np.argsort(type_to_print)
    else:
        ind_array = np.argsort(names)

    for i in ind_array:
        k1 = names[i]
        k2 = type_to_print[i]
        # k3 = str(c[names[i]])
        k3 = str(eval('c.' + names[i]))
        k1 = (k1 + len_space)[:len(len_space)]
        k2 = (k2 + len_space_type)[:len(len_space_type)]
        if len(k3) > end_prev_len:
            k3 = '...' + k3[-end_prev_len:]
        else:
            k3 = '> ' + k3[-end_prev_len:]

        print(k1 + ' type->   ' + k2 + '  ' + k3)
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

    Parameters
    ----------
    base_dir :
        
    search_term :
        

    Returns
    -------

    """
    file_list = []
    for path in Path(base_dir + '/').rglob(search_term):
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


def loop_segments(frame_num_array):
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
    borders : for touch 0000011100000 it will find teh 111 in it and get all the bordering areas around it. points are
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
    f : h5 file with SINGLE FRAMES this is ment to be a test program. if used long term I will change this part
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
            new_imgs = image_tools.stack_imgs_lag(h['images'][k1:k2], buffer=2, shift_to_the_right_by=0)
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
    k_names = print_h5_keys(f, return_list=True, do_print=False)
    k_names = lister_it(k_names, remove_string='MODEL_')
    k_names = lister_it(k_names, remove_string='images')
    with h5py.File(f, 'r+') as h:
        for kn in k_names:
            try:
                copy_h5_key_to_another_h5(f, f2, kn, kn)
            except:
                del h[kn]
                time.sleep(2)
                copy_h5_key_to_another_h5(f, f2, kn, kn)
    if 'frame_nums' not in k_names and 'trial_nums_and_frame_nums' in k_names:
        tnfn = image_tools.get_h5_key_and_concatenate([f], 'trial_nums_and_frame_nums')
        with h5py.File(f2, 'r+') as h:
            try:
                h.create_dataset('frame_nums', data=(tnfn[1, :]).astype(int))
            except:
                del h['frame_nums']
                time.sleep(2)
                h.create_dataset('frame_nums', data=(tnfn[1, :]).astype(int))


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

