import numpy as np
from pathlib import Path
import os
import glob
from natsort import os_sorted
import scipy.io as spio
import h5py
import matplotlib.pyplot as plt


def lister_it(in_list, keep_strings=None, remove_string=None):
    """

    Parameters
    ----------
    in_list : list
    keep_strings : list
    remove_string : list

    Returns
    -------

    """
    if isinstance(keep_strings, str):
        keep_strings = [keep_strings]
    if isinstance(remove_string, str):
        remove_string = [remove_string]

    if keep_strings is None:
        new_list = in_list
    else:
        new_list = []
        for L in in_list:
            for k in keep_strings:
                if k in L:
                    new_list.append(L)

    if remove_string is None:
        new_list_2 = new_list
    else:
        new_list_2 = []
        for L in new_list:
            for k in remove_string:
                if k not in L:
                    new_list_2.append(L)
    return new_list_2


def plot_pole_tracking_max_vals(h5_file):
    with h5py.File(h5_file, 'r') as hf:
        for i, k in enumerate(hf['max_val_stack'][:]):
            plt.plot()


def get_class_info(c, include_underscores=False):
    """look at the variables and methods of a class, prints an aligned list with their respective 'type'

    Parameters
    ----------
    c : class
        calss variable
    include_underscores : bool
        display the secret methods/variables starting with '_' (Default value = False)

    Returns
    -------
    type
        prints out a nice list of the modethods

    """
    names = []
    type_to_print = []
    for k in dir(c):
        if include_underscores is False and k[0] != '_':
            tmp1 = str(type(eval('c.' + k)))
            type_to_print.append(tmp1.split("""'""")[-2])
            names.append(k)
        elif include_underscores:
            tmp1 = str(type(eval('c.' + k)))
            type_to_print.append(tmp1.split("""'""")[-2])
            names.append(k)
    len_space = ' ' * max(len(k) for k in names)
    for k1, k2 in zip(names, type_to_print):
        k1 = (k1 + len_space)[:len(len_space)]
        print(k1 + ' type->   ' + k2)


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


def get_h5s(base_dir):
    """

    Parameters
    ----------
    base_dir :
        

    Returns
    -------

    """
    H5_file_list = []
    for path in Path(base_dir + '/').rglob('*.h5'):
        H5_file_list.append(str(path.parent) + os.path.sep + path.name)
    H5_file_list.sort()
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


def get_files(base_dir, search_term):
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
