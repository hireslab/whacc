import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import h5py
import copy
import time
import os
from whacc import utils
from pathlib import Path
import warnings
from tqdm.autonotebook import tqdm
import pdb
from IPython.utils import io


# def tqdm_import_helper():
#     with io.capture_output() as captured:  # prevent crazy printing
#         from tqdm.notebook import tqdm
#         try:
#             for k in tqdm(range(1)):
#                 pass
#             return True and isnotebook()
#         except:
#             return False

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

def stack_imgs_lag(imgs, frames_1=None, buffer=2, shift_to_the_right_by=0):
    if frames_1 is None:
        frames_1 = [imgs.shape[0]]
    array_group = []
    for k1, k2 in utils.loop_segments(frames_1):
        x = (np.random.random(imgs[0].shape) * 255).astype(np.uint8)
        tile_axes = [1] * len(x.shape) + [buffer]
        x = np.tile(x[:, :, None], tile_axes)
        tmp1 = x.copy()
        for ii, stack_i in enumerate(range(k1, k2)):
            x = np.concatenate((x, imgs[stack_i][:, :, None]), axis=2)
        x = np.concatenate((x, tmp1), axis=2)
        for k3 in range(k2 - k1):
            array_group.append(x[:, :, k3 + shift_to_the_right_by: k3 + 1 + buffer + shift_to_the_right_by])
    return np.asarray(array_group)


def get_h5_key_and_concatenate(h5_list, key_name='labels'):
    """
    simply extract and concatenate all of one key "key_name" from many H5 files, I use it to get balance the data touch
    and not touch frames when training a model with a list of different H5 files
    Parameters
    ----------
    h5_list : list
        list of full paths to H5 file(s).
    key_name : str
        default 'labels', the key to get the data from the H5 file

    """
    h5_list = utils.make_list(h5_list, suppress_warning=True)
    for i, k in enumerate(h5_list):
        with h5py.File(k, 'r') as h:
            try:
                x = h[key_name][:]
            except:
                x = h[key_name]

            if i == 0:
                out = np.asarray(x)
            else:
                out = np.concatenate((out, x))
    return out


def get_h5_key_and_dont_concatenate(h5_list, key_name='labels'):
    """
    simply extract and concatenate all of one key "key_name" from many H5 files, I use it to get balance the data touch
    and not touch frames when training a model with a list of different H5 files
    Parameters
    ----------
    h5_list : list
        list of full paths to H5 file(s).
    key_name : str
        default 'labels', the key to get the data from the H5 file

    """
    out = []
    for i, k in enumerate(h5_list):
        with h5py.File(k, 'r') as h:
            out.append(list(h[key_name][:]))
    return out


def clone_h5_basic_info(H5_list, fold_name=None, file_end='_QUICK_SAVE.h5'):
    """
    copies all the info form H5 into another H5 file NOT INCLUDING the labels or images. so it have all the file info,
    like names and pole locations and polate match max value stack. anything with 'images' , 'MODEL__' or 'labels' is
    not copied over to the new file.
    Parameters
    ----------
    H5_list : list
        list of H5 files to clone
    fold_name : str
        default None, where to place the cloned H5 files. if left blank it will place in the same folder as the original file
    file_end : str
        default '_QUICK_SAVE.h5', how to change the name of the H5 file to be cloned to differentiate it from the original
    Returns
    -------
    all_new_h5s: list
        list of new H5 full file names
    """
    if fold_name is not None:
        try:
            os.mkdir(fold_name)
        except:
            pass
        all_new_h5s = []

    for h5 in H5_list:
        if fold_name is not None:
            new_fn = fold_name + os.path.sep + os.path.basename(h5)[:-3] + file_end
        else:  #
            new_fn = os.path.dirname(h5) + os.path.sep + os.path.basename(h5)[:-3] + file_end
        all_new_h5s.append(new_fn)
        try:
            os.remove(new_fn)
        except:
            pass
        with h5py.File(new_fn, 'w') as f1:
            with h5py.File(h5, 'r') as f2:
                for i, k in enumerate(f2.keys()):
                    if 'images' != k and 'MODEL__' not in k and 'labels' not in k:
                        f1.create_dataset(k, data=f2[k][:])
                f2.close()
            f1.close()
        return all_new_h5s


def del_h5_with_term(h5_list, str_2_cmp):
    """
    Parameters
    ----------
    h5_list : list
        list of H5 strings (full path)
    str_2_cmp : str
        will delete keys with this in their title ... e.g. '__RETRAIN'
    """
    for k2 in h5_list:
        with h5py.File(k2, 'a') as h5_source:
            for k in h5_source.keys():
                if str_2_cmp in k:
                    print('del--> ' + k)
                    del h5_source[k]
            print('_______')


def split_h5_loop_segments(h5_to_split_list, split_percentages, temp_base_name, chunk_size=10000,
                           add_numbers_to_name=True,
                           disable_TQDM=False, set_seed=None, color_channel=True, force_random_each_frame=False):
    """Randomly splits images from a list of H5 file(s) into len(split_percentages) different H5 files.

    Parameters
    ----------
    h5_to_split_list : list
        list of strings with full file names to the H5 file(s) to be split
    split_percentages : list
        list of numbers, can be ints [20, 1, 1] and or floats [.8, .2], it simply takes the sum and creates a percentage
    temp_base_name : str or list
        full path to new h5 file e.g "'/Users/phil/tempH5_" and the program will add the number and the ".h5"
        in this case tempH5_0.h5, tempH5_1.h5, tempH5_2.h5 etc. or if it is a list it must be equal in length to
        'split_percentages' and each file will be named based on that list
    chunk_size = int
        default 10000, max amount of frames to hold in memory at a time before storing in H5 file. Should almost never
        be an issue but just in case you can set to a lower value if you experience memory issues.
    add_numbers_to_name = bool
        default true, just in case you don't want the numbers on the end of your h5 file.
    Returns
    Examples
    --------
    from whacc import image_tools, utils
    h5_to_split_list = "/Users/phil/Downloads/untitled folder 2/AH0000x000000_small_tester.h5"
    h5_to_split_list = [h5_to_split_list]
    utils.print_h5_keys(h5_to_split_list[0])
    bd = '/Users/phil/Downloads/untitled folder 2/'
    image_tools.split_h5_loop_segments(h5_to_split_list, [1, 3], [bd+'TRASH', bd+'TRASH2'], chunk_size=10000, add_numbers_to_name=False,
                 disable_TQDM=False, set_seed = None)
    -------
    """
    if isinstance(temp_base_name, str):
        temp_base_name = [temp_base_name] * len(split_percentages)
    else:
        assert len(temp_base_name) == len(
            split_percentages), """if 'temp_base_name' is a list of strings, it must be equal in length to 'split_percentages'"""

    for i, k in enumerate(temp_base_name):
        if k[-3:] == '.h5':
            temp_base_name[i] = temp_base_name[i][:-3]

    frame_num_array_list = get_h5_key_and_dont_concatenate(h5_to_split_list, 'frame_nums')
    if force_random_each_frame:
        for i, k in enumerate(frame_num_array_list):
            frame_num_array_list[i] = list(np.ones(np.sum(k)).astype(int))

    total_frames = len(get_h5_key_and_concatenate(h5_to_split_list, key_name='labels'))
    cnt1 = 0
    h5_creators = dict()
    split_percentages = split_percentages / np.sum(split_percentages)
    # assert(sum(split_percentages)==1)
    final_names = []
    for iii, h5_to_split in enumerate(h5_to_split_list):
        with h5py.File(h5_to_split, 'r') as h:
            tmp_frame_list = frame_num_array_list[iii]
            L = len(tmp_frame_list)

            if set_seed is not None:
                np.random.seed(set_seed)
            mixed_inds = np.random.choice(L, L, replace=False)

            random_segment_inds = np.split(mixed_inds, np.ceil(L * np.cumsum(split_percentages[:-1])).astype('int'))
            random_segment_inds = [sorted(tmpk) for tmpk in random_segment_inds]
            random_frame_inds = [[None]] * len(random_segment_inds)
            list_of_new_frame_nums = [[None]] * len(random_segment_inds)
            loop_seg_list = list(utils.loop_segments(tmp_frame_list))
            for pi, p in enumerate(random_segment_inds):
                tmp1 = []
                tmp2 = []
                for pp in p:
                    x = list(loop_seg_list[pp])
                    tmp1 += list(range(x[0], x[1]))
                    tmp2.append(tmp_frame_list[pp])
                random_frame_inds[pi] = tmp1
                list_of_new_frame_nums[pi] = tmp2

            for i, k in enumerate(split_percentages):  # for each new h5 created
                if iii == 0:  # create the H5 creators
                    if add_numbers_to_name:
                        final_names.append(temp_base_name[i] + '_' + str(i) + '.h5')
                    else:
                        final_names.append(temp_base_name[i] + '.h5')
                    h5_creators[i] = h5_iterative_creator(final_names[-1],
                                                          overwrite_if_file_exists=True,
                                                          close_and_open_on_each_iteration=True,
                                                          color_channel=color_channel)
                ims = []
                labels = []
                for ii in tqdm(sorted(random_frame_inds[i]), disable=disable_TQDM, total=total_frames, initial=cnt1):
                    cnt1 += 1
                    ims.append(h['images'][ii])
                    labels.append(h['labels'][ii])
                    if ii > 0 and ii % chunk_size == 0:
                        h5_creators[i].add_to_h5(np.asarray(ims), np.asarray(labels))
                        ims = []
                        labels = []
                h5_creators[i].add_to_h5(np.asarray(ims), np.asarray(labels))
                with h5py.File(h5_creators[i].h5_full_file_name,
                               'r+') as h2:  # wanted to do this to allow NONE as input and still have frame nums, but I need to have an append after creating and its a pain
                    frame_nums = np.asarray(list_of_new_frame_nums[i])
                    if 'frame_nums' not in h2.keys():
                        h2.create_dataset('frame_nums', shape=np.shape(frame_nums), maxshape=(None,), chunks=True,
                                          data=frame_nums)
                    else:
                        h2['frame_nums'].resize(h2['frame_nums'].shape[0] + frame_nums.shape[0], axis=0)
                        h2['frame_nums'][-frame_nums.shape[0]:] = frame_nums
    # # add the frame info to each
    # for i, frame_nums in enumerate(list_of_new_frame_nums):
    #     with h5py.File(h5_creators[i].h5_full_file_name, 'r+') as h:
    #         h.create_dataset('frame_nums', shape=np.shape(frame_nums), data=frame_nums)
    return final_names


def make_sure_frame_nums_exist(h5file):
    with h5py.File(h5file, 'r+') as h:
        key_list = list(h.keys())
        if 'frame_nums' in key_list:
            print("""'frame_nums' already in the key list""")
            return None
        if 'trial_nums_and_frame_nums' not in key_list:
            print(
                """key 'trial_nums_and_frame_nums' must be in the provided h5 this is the only reason program exists""")
            return None
        frame_nums = h['trial_nums_and_frame_nums'][1, :]
        h.create_dataset('frame_nums', shape=np.shape(frame_nums), data=frame_nums)


def split_h5(h5_to_split_list, split_percentages, temp_base_name, chunk_size=10000, add_numbers_to_name=True,
             disable_TQDM=False, skip_if_label_is_neg_1=False, set_seed=None, color_channel=True):
    """Randomly splits images from a list of H5 file(s) into len(split_percentages) different H5 files.

    Parameters
    ----------
    h5_to_split_list : list
        list of strings with full file names to the H5 file(s) to be split
    split_percentages : list
        list of numbers, can be ints [20, 1, 1] and or floats [.8, .2], it simply takes the sum and creates a percentage
    temp_base_name : str or list
        full path to new h5 file e.g "'/Users/phil/tempH5_" and the program will add the number and the ".h5"
        in this case tempH5_0.h5, tempH5_1.h5, tempH5_2.h5 etc. or if it is a list it must be equal in length to
        'split_percentages' and each file will be named based on that list
    chunk_size = int
        default 10000, max amount of frames to hold in memory at a time before storing in H5 file. Should almost never
        be an issue but just in case you can set to a lower value if you experience memory issues.
    add_numbers_to_name = bool
        default true, just in case you don't want the numbers on the end of your h5 file.
    Returns
    -------
    """
    if isinstance(temp_base_name, str):
        temp_base_name = [temp_base_name] * len(split_percentages)
    else:
        assert len(temp_base_name) == len(
            split_percentages), """if 'temp_base_name' is a list of strings, it must be equal in length to 'split_percentages'"""
    total_frames = len(get_h5_key_and_concatenate(h5_to_split_list, key_name='labels'))
    cnt1 = 0
    h5_creators = dict()
    split_percentages = split_percentages / np.sum(split_percentages)
    # assert(sum(split_percentages)==1)
    final_names = []
    for iii, h5_to_split in enumerate(h5_to_split_list):
        with h5py.File(h5_to_split, 'r') as h:
            L = len(h['labels'][:])
            if set_seed is not None:
                np.random.seed(set_seed)
            mixed_inds = np.random.choice(L, L, replace=False)
            if skip_if_label_is_neg_1:  # remove -1s
                mixed_inds = mixed_inds[mixed_inds != -1]
            random_frame_inds = np.split(mixed_inds, np.ceil(L * np.cumsum(split_percentages[:-1])).astype('int'))
            for i, k in enumerate(split_percentages):
                if iii == 0:  # create the H5 creators
                    if add_numbers_to_name:
                        final_names.append(temp_base_name[i] + '_' + str(i) + '.h5')
                    else:
                        final_names.append(temp_base_name[i] + '.h5')
                    h5_creators[i] = h5_iterative_creator(final_names[-1],
                                                          overwrite_if_file_exists=True,
                                                          close_and_open_on_each_iteration=True,
                                                          color_channel=color_channel)
                ims = []
                labels = []
                # print('starting ' + str(iii*i + 1) + ' of ' + str(len(split_percentages)*len(h5_to_split_list)))
                for ii in tqdm(sorted(random_frame_inds[i]), disable=disable_TQDM, total=total_frames, initial=cnt1):
                    cnt1 += 1
                    ims.append(h['images'][ii])
                    labels.append(h['labels'][ii])
                    if ii > 0 and ii % chunk_size == 0:
                        h5_creators[i].add_to_h5(np.asarray(ims), np.asarray(labels))
                        ims = []
                        labels = []
                h5_creators[i].add_to_h5(np.asarray(ims), np.asarray(labels))
    return final_names


class h5_iterative_creator():
    """Create an H5 file using a for loop easily. used to create the augmented H5 file for training
    
    Attributes:

    Parameters
    ----------
    h5_new_full_file_name : string
        full path name to your H5 file to be created
    overwrite_if_file_exists : bool
        overwrites the h5 file if it already exists
    max_img_height : int
        default 61, only the max size, can be larger in case you are going to have larger images
    max_img_width : int
        default 61, only the max size, can be larger in case you are going to have larger images
    close_and_open_on_each_iteration : bool
        default True, this prevents the user from forgetting to close H5 which
        can lead to corruption.

    Example
    _______
    h5creator = h5_iterative_creator(new_H5_file)
    h5creator.add_to_h5(img_stack1, labels_stack1)
    h5creator.add_to_h5(img_stack2, labels_stack2)
    h5creator.add_to_h5(img_stack3, labels_stack3)

    """

    def __init__(self, h5_new_full_file_name,
                 overwrite_if_file_exists=False,
                 max_img_height=61,
                 max_img_width=61,
                 close_and_open_on_each_iteration=True,
                 color_channel=True,
                 add_to_existing_H5=False,
                 ignore_image_range_warning=False,
                 dtype_img=h5py.h5t.STD_U8BE,
                 dtype_labels=h5py.h5t.STD_I32LE,
                 image_key_name = 'images',
                 label_key_name = 'labels'):
        self.img_key = image_key_name
        self.label_key_name = label_key_name
        self.dtype_img = dtype_img
        self.dtype_labels = dtype_labels
        self.ignore_image_range_warning = False
        self.max_shape = None
        if not close_and_open_on_each_iteration:
            print('**remember to CLOSE the H5 file when you are done!!!**')
        if overwrite_if_file_exists and os.path.isfile(h5_new_full_file_name):
            os.remove(h5_new_full_file_name)
        self.h5_full_file_name = h5_new_full_file_name
        if add_to_existing_H5:
            self.hf_file = h5py.File(h5_new_full_file_name, "r+")
        else:
            self.hf_file = h5py.File(h5_new_full_file_name, "w")
        self.color_channel = color_channel
        self.max_img_height = max_img_height
        self.max_img_width = max_img_width
        self._went_through_create_h5 = False
        self.close_it = close_and_open_on_each_iteration
        if self.close_it:
            self.hf_file.close()

    def add_to_h5(self, images, labels):
        """
        Parameters
        ----------
        images : numpy tensor
            chunk of images
        labels : numpy array
            array oof labels
        """
        if self.close_it:
            self.open_or_close_h5('r+')
        if self._went_through_create_h5:  # already initialized with the correct size
            self._add_next_chunk_to_h5(images, labels)
        else:
            self._create_h5(images, labels)
        if self.close_it:
            self.open_or_close_h5('close')

    def check_images_uint8(self, images):
        if not self.ignore_image_range_warning:
            min_img = np.min(images)
            max_img = np.max(images)
            if min_img < 0 or max_img > 255:
                warnings.warn(
                    'image data must be uint8 compatible, 0 to 255, but given range is ' + str(min_img) + ' to ' + str(
                        max_img))
            if -1 <= min_img <= 1 and -1 <= max_img <= 1:
                warnings.warn(
                    'image data must be uint8 compatible, 0 to 255, but given range is ' + str(min_img) + ' to ' + str(
                        max_img))
                warnings.warn('it seems your values may be formatted between -1 and 1')
            if 0 <= min_img <= 1 and 0 <= max_img <= 1:
                warnings.warn(
                    'image data must be uint8 compatible, 0 to 255, but given range is ' + str(min_img) + ' to ' + str(
                        max_img))
                warnings.warn('it seems your values may be formatted between 0 and 1')

    def _create_h5(self, images, labels):
        """
        Parameters
        ----------
        images :

        labels :

        """
        self.check_images_uint8(images)
        if self.max_shape is None:
            # max_shape = (None, self.max_img_height, self.max_img_width, 3)
            max_shape = list(images.shape)
            max_shape[0] = None

        # if set_multiplier:
        self.hf_file.create_dataset("multiplier", [1], h5py.h5t.STD_I32LE, data=images.shape[0])
        if self.color_channel:
            self.hf_file.create_dataset(self.img_key,
                                        np.shape(images),
                                        self.dtype_img,
                                        # jk need this to not explode the size of the data... commented this out because I wanted to use not 0-255 numbers
                                        maxshape=max_shape,
                                        chunks=True,
                                        data=images)
        else:
            self.hf_file.create_dataset(self.img_key,
                                        np.shape(images),
                                        self.dtype_img,
                                        # jk need this to not explode the size of the data... commented this out because I wanted to use not 0-255 numbers
                                        maxshape=max_shape,
                                        chunks=True,
                                        data=images)
        self.hf_file.create_dataset(self.label_key_name,
                                    np.shape(labels),
                                    self.dtype_labels,  # ....... commented this out because we may want floats....
                                    maxshape=(None,),
                                    chunks=True,
                                    data=labels)
        self._went_through_create_h5 = True

    def _add_next_chunk_to_h5(self, images, labels):
        """

        Parameters
        ----------
        images :

        labels :
            

        Returns
        -------

        
        """
        self.check_images_uint8(images)
        self.hf_file[self.img_key].resize(self.hf_file[self.img_key].shape[0] + images.shape[0], axis=0)
        self.hf_file[self.label_key_name].resize(self.hf_file[self.label_key_name].shape[0] + labels.shape[0], axis=0)

        self.hf_file[self.img_key][-images.shape[0]:] = images
        self.hf_file[self.label_key_name][-labels.shape[0]:] = labels

    def read_h5(self):
        """ """
        self.open_or_close_h5('r')
        print('''**remember to CLOSE the H5 file when you are done!!!** with ".close_h5()" method''')

    def close_h5(self):
        """ """
        self.open_or_close_h5('close')
        print('H5 file was closed')

    def open_or_close_h5(self, mode_='r'):
        """

        Parameters
        ----------
        mode_ : str
            mode can be H5py modes 'r', 'r+' 'w' (w overwrites file!) etc OR 'close' to
            # ensure it is closed. separate function to prevent a bunch of try statements (Default value = 'r')

        Returns
        -------

        
        """
        try:
            self.hf_file.close()
        finally:
            if mode_.lower() != 'close':
                self.hf_file = h5py.File(self.h5_full_file_name, mode_)


#
def augment_helper(keras_datagen, num_aug_ims, num_reg_ims, in_img, in_label):
    """

    Parameters
    ----------
    keras_datagen : keras_datagen: keras_datagen: keras.preprocessing.image.ImageDataGenerator
        from keras.preprocessing.image import ImageDataGenerator-- keras_datagen = ImageDataGenerator(...)
    num_aug_ims : int
        number of augmented images to generate from single input image
    num_reg_ims : int
        number of copies of in_img (original) to produce for output. will be stacked at the beginning of all_augment variable.
        Use dot see augmentation when testing and can be useful if splitting into many H5s if you want an original in each.
    in_img : numpy array
        numpy array either 3D with color channel for the last dim ot 2D
    in_label : int
        the label associate with in_img. simply repeats it creating 'out_labels' the be size of 'all_augment'

    Returns
    -------

    
    """
    if len(in_img.shape) == 2:  # or not np.any(np.asarray(in_img.shape)==3)
        in_img = np.repeat(in_img[..., np.newaxis], 3, -1)  # for 2D arrays without color channels
    set_zoom = keras_datagen.zoom_range
    in_img = np.expand_dims(in_img, 0)

    it = keras_datagen.flow(in_img, batch_size=1)
    all_augment = np.tile(in_img, [num_reg_ims, 1, 1, 1])
    for i in range(num_aug_ims):  ##
        if set_zoom != [0, 0]:  # if zoom is being used...
            # keras 'zoom' is annoying. it zooms x and y differently randomly
            # in order to get an equal zoom I use the following workaround.
            z_val = np.random.uniform(low=set_zoom[0], high=set_zoom[1])
            keras_datagen.zoom_range = [z_val, z_val]
            it = keras_datagen.flow(in_img, batch_size=1)
        batch = it.next()
        image = batch[0].astype('uint8')
        all_augment = np.append(all_augment, np.expand_dims(image, 0), 0)
    out_labels = np.repeat(in_label, sum([num_aug_ims, num_reg_ims]))
    keras_datagen.zoom_range = set_zoom
    return all_augment, out_labels


def img_unstacker(img_array, num_frames_wide=8, color_channel=True):
    """unstacks image stack and combines them into one large image for easy display. reads left to right and then top to bottom.

    Parameters
    ----------
    img_array : numpy array
        stacked image array
    num_frames_wide : int
        width of destacked image. if = 8 with input 20 images it will be 8 wide 3 long and 4 blank images (Default value = 8)

    Returns
    -------

    
    """
    im_stack = None
    for i, k in enumerate(img_array):
        if i % num_frames_wide == 0:
            if i != 0:  # stack it
                if im_stack is None:
                    im_stack = im_stack_tmp
                else:
                    im_stack = np.vstack((im_stack, im_stack_tmp))
            im_stack_tmp = k  # must be at the end
        else:
            im_stack_tmp = np.hstack((im_stack_tmp, k))
    x = num_frames_wide - len(img_array) % num_frames_wide
    if x != 0:
        if x != num_frames_wide:
            for i in range(x):
                im_stack_tmp = np.hstack((im_stack_tmp, np.ones_like(k)))
    if im_stack is None:
        return im_stack_tmp
    else:
        im_stack = np.vstack((im_stack, im_stack_tmp))
        return im_stack


def original_image(x):
    """This is used to transform batch generated images [-1 1] to the original image [0,255] for plotting

    Parameters
    ----------
    x :
        

    Returns
    -------

    
    """
    image = tf.cast((x + 1) * 127.5, tf.uint8)
    return image


def predict_multiple_H5_files(H5_file_list, model_2_load, append_model_and_labels_to_name_string=False,
                              batch_size=1000, model_2_load_is_model=False, save_on=False,
                              label_save_name=None, disable_TQDM=False,
                              save_labels_to_this_h5_file_instead=None) -> object:
    """

    Parameters
    ----------
    H5_file_list : list: list
        list of string(s) of H5 file full paths
    model_2_load : param append_model_and_labels_to_name_string: if True label_save_name =  'MODEL__' + label_save_name + '__labels',
        
    it is a simple way to keep track of labels form many models in a single H5 file. also make sit easier to find :
        
    those labels for later processing. :
        either full path to model folder ending with ".ckpt" OR the loaded model itself. if the later,
        the user MUST set "model_2_load_is_model" is True and "label_save_name" must be explicitly defined (when using model
        path we use the model name to name the labels).
    append_model_and_labels_to_name_string : bool
        if True label_save_name =  'MODEL__' + label_save_name + '__labels',it is a simple way to keep track of labels
        form many models in a single H5 file. also make sit easier to find those labels for later processing. (Default value = False)
    batch_size : int
        number of images to process per batch,  -- slower prediction speeds << ideal predictionsspeed <<
        memory issues and crashes -- 1000 is normally pretty good on Google CoLab (Default value = 1000)
    model_2_load_is_model : bool
        lets the program know if you are directly inserting a model (instead of a path to model folder) (Default value = False)
    save_on : bool
        saves to H5 file. either the original H5 (image source) or new H5 if a path to "save_labels_to_this_h5_file_instead"
        is given (Default value = False)
    label_save_name : string
        h5 file key used to save the labels to, default is 'MODEL__' + **model_name** + '__labels'
    disable_TQDM : bool
        if True, turns off loading progress bar. (Default value = False)
    save_labels_to_this_h5_file_instead : string
        full path to H5 file to insert labels into instead of the H5 used as the image source (Default value = None)

    Returns
    -------

    
    """
    for i, H5_file in enumerate(H5_file_list):
        # save_what_is_left_of_your_h5_file(H5_file, do_del_and_rename = 1) # only matters if file is corrupt otherwise doesnt touch it

        gen = ImageBatchGenerator(batch_size, [H5_file])

        if model_2_load_is_model:
            if label_save_name is None and save_on == True:
                assert 1 == 0, 'label_save_name must be assigned if you are loading a model in directly and saveon == True.'
            model = model_2_load
        else:
            if label_save_name is None:
                label_save_name = model_2_load.split(os.path.sep)[-1].split('.')[0]
                label_save_name = 'MODEL__' + label_save_name + '__labels'
                append_model_and_labels_to_name_string = False  # turn off because defaults to this naming scheme if user doesnt put in name
            model = tf.keras.models.load_model(model_2_load)

        if append_model_and_labels_to_name_string:
            label_save_name = 'MODEL__' + label_save_name + '__labels'

        start = time.time()
        labels_2_save = np.asarray([])

        for k in tqdm(range(gen.__len__()), disable=disable_TQDM):
            TMP_X, tmp_y = gen.getXandY(k)
            outY = model.predict(TMP_X)
            labels_2_save = np.append(labels_2_save, outY)
        total_seconds = time.time() - start
        time_per_mil = np.round(1000000 * total_seconds / len(labels_2_save))
        print(str(time_per_mil) + ' seconds per 1 million images predicted')

        if save_on:
            if save_labels_to_this_h5_file_instead is not None:  # add to differnt H5 file
                H5_file = save_labels_to_this_h5_file_instead  # otherwise it will add to the current H5 file
                # based on the loop through "H5_file_list" above
            try:
                hf.close()
            except:
                pass
            with h5py.File(H5_file, 'r+') as hf:
                try:
                    del hf[label_save_name]
                    time.sleep(10)  # give time to process the deleted file... maybe???
                    hf.create_dataset(label_save_name, data=np.float64(labels_2_save))
                except:
                    hf.create_dataset(label_save_name, data=np.float64(labels_2_save))
                hf.close()
    return labels_2_save


def get_total_frame_count(h5_file_list):
    """

    Parameters
    ----------
    h5_file_list :
        

    Returns
    -------

    
    """
    total_frame_count = []
    for H5_file in h5_file_list:
        with h5py.File(H5_file, 'r') as H5:
            total_frame_count.append(H5['images'].shape[0])

    return total_frame_count


def batch_size_file_ind_selector(num_in_each, batch_size):
    """batch_size_file_ind_selector - needed for ImageBatchGenerator to know which H5 file index
    to use depending on the iteration number used in __getitem__ in the generator.
    this all depends on the variable batch size.
    
    Example: the output of the following...
    batch_size_file_ind_selector([4000, 4001, 3999], [2000])
    would be [0, 0, 1, 1, 1, 2, 2] which means that there are 2 chunks in the first
    H5 file, 3 in the second and 2 in the third based on chunk size of 2000

    Parameters
    ----------
    num_in_each :
        param batch_size:
    batch_size :
        

    Returns
    -------

    
    """
    break_into = np.ceil(np.array(num_in_each) / batch_size)
    extract_inds = np.array([])
    for k, elem in enumerate(break_into):
        tmp1 = np.array(np.ones(np.int(elem)) * k)
        extract_inds = np.concatenate((extract_inds, tmp1), axis=0)
    return extract_inds


# file_inds_for_H5_extraction is the same as extract_inds output from the above function
def reset_to_first_frame_for_each_file_ind(file_inds_for_H5_extraction):
    """reset_to_first_frame_for_each_file_ind - uses the output of batch_size_file_ind_selector
    to determine when to reset the index for each individual H5 file. using the above example
    the out put would be [0, 0, 2, 2, 2, 5, 5], each would be subtracted from the indexing to
    set the position of the index to 0 for each new H5 file.

    Parameters
    ----------
    file_inds_for_H5_extraction :
        

    Returns
    -------

    
    """
    subtract_for_index = []
    for k, elem in enumerate(file_inds_for_H5_extraction):
        tmp1 = np.diff(file_inds_for_H5_extraction)
        tmp1 = np.where(tmp1 != 0)
        tmp1 = np.append(-1, tmp1[0]) + 1
        subtract_for_index.append(tmp1[np.int(file_inds_for_H5_extraction[k])])
    return subtract_for_index


class ImageBatchGenerator(keras.utils.Sequence):
    """ """

    def __init__(self, batch_size, h5_file_list, label_key='labels'):
        h5_file_list = utils.make_list(h5_file_list, suppress_warning=True)
        num_frames_in_all_H5_files = get_total_frame_count(h5_file_list)
        file_inds_for_H5_extraction = batch_size_file_ind_selector(
            num_frames_in_all_H5_files, batch_size)
        subtract_for_index = reset_to_first_frame_for_each_file_ind(
            file_inds_for_H5_extraction)
        # self.to_fit = to_fit #set to True to return XY and False to return X
        self.label_key = label_key
        self.batch_size = batch_size
        self.H5_file_list = h5_file_list
        self.num_frames_in_all_H5_files = num_frames_in_all_H5_files
        self.file_inds_for_H5_extraction = file_inds_for_H5_extraction
        self.subtract_for_index = subtract_for_index
        self.IMG_SIZE = 96

    def __len__(self):
        return len(self.file_inds_for_H5_extraction)

    def __getitem__(self, num_2_extract):
        b = self.batch_size
        h = self.H5_file_list
        i = self.file_inds_for_H5_extraction
        H5_file = h[np.int(i[num_2_extract])]
        with h5py.File(H5_file, 'r') as H5:
            # H5 = h5py.File(H5_file, 'r')

            images = H5['images']
            num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
            raw_X = images[b * num_2_extract_mod:b * (num_2_extract_mod + 1)]
            rgb_tensor = self.image_transform(raw_X)

            labels_tmp = H5[self.label_key]
            raw_Y = labels_tmp[b * num_2_extract_mod:b * (num_2_extract_mod + 1)]
            H5.close()
        return rgb_tensor, raw_Y

    # def __getitem__(self, num_2_extract):
    #     b = self.batch_size
    #     h = self.H5_file_list
    #     i = self.file_inds_for_H5_extraction
    #     H5_file = h[np.int(i[num_2_extract])]
    #     H5 = h5py.File(H5_file, 'r')
    #     #  list(H5.keys())
    #
    #     images = H5['images']
    #     num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
    #     raw_X = images[b * num_2_extract_mod:b * (num_2_extract_mod + 1)]
    #     rgb_tensor = self.image_transform(raw_X)
    #
    #     # if self.to_fit:
    #     #   labels_tmp = H5[self.label_key]
    #     #   raw_Y = labels_tmp[b*num_2_extract_mod:b*(num_2_extract_mod+1)]
    #     #   return rgb_tensor, raw_Y
    #     # else:
    #     return rgb_tensor

    def getXandY(self, num_2_extract):
        """

        Parameters
        ----------
        num_2_extract :
            

        Returns
        -------

        
        """
        b = self.batch_size
        h = self.H5_file_list
        i = self.file_inds_for_H5_extraction
        H5_file = h[np.int(i[num_2_extract])]
        H5 = h5py.File(H5_file, 'r')
        #  list(H5.keys())

        images = H5['images']
        num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
        raw_X = images[b * num_2_extract_mod:b * (num_2_extract_mod + 1)]
        rgb_tensor = self.image_transform(raw_X)
        labels_tmp = H5[self.label_key]
        raw_Y = labels_tmp[b * num_2_extract_mod:b * (num_2_extract_mod + 1)]
        return rgb_tensor, raw_Y

    def image_transform(self, raw_X):
        """input num_of_images x H x W, image input must be grayscale
        MobileNetV2 requires certain image dimensions
        We use N x 61 x 61 formated images
        self.IMG_SIZE is a single number to change the images into, images must be square

        Parameters
        ----------
        raw_X :
            

        Returns
        -------

        
        """
        # rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
        # rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
        # rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2
        # rgb_tensor = tf.image.resize(rgb_tensor, (self.IMG_SIZE, self.IMG_SIZE))  # resizing
        # self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)
        # return rgb_tensor
        if len(raw_X.shape) == 4 and raw_X.shape[3] == 3:
            rgb_batch = copy.deepcopy(raw_X)
        else:
            rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
        rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
        rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2
        rgb_tensor = tf.image.resize(rgb_tensor, (self.IMG_SIZE, self.IMG_SIZE))  # resizing
        self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)
        return rgb_tensor

    def plot_batch_distribution(self):
        """ """
        # randomly select a batch and generate images and labels
        batch_num = np.random.choice(np.arange(0, self.__len__()))
        samp_x, samp_y = self.getXandY(batch_num)

        # look at the distribution of classes
        plt.pie([1 - np.mean(samp_y), np.mean(samp_y)],
                labels=['non-touch frames', 'touch frames'], autopct='%1.1f%%', )
        plt.title('class distribution from batch ' + str(batch_num))
        plt.show()

        # generate indices for positive and negative classes
        images_to_sample = 20
        neg_class = [i for i, val in enumerate(samp_y) if val == 0]
        pos_class = [i for i, val in enumerate(samp_y) if val == 1]
        neg_index = np.random.choice(neg_class, images_to_sample)
        pos_index = np.random.choice(pos_class, images_to_sample)

        # plot sample positive and negative class images
        plt.figure(figsize=(10, 10))
        samp_x = (samp_x + 1) / 2
        for i in range(images_to_sample):
            plt.subplot(5, 10, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            _ = plt.imshow(samp_x[neg_index[i]])
            plt.xlabel('0')

            plt.subplot(5, 10, images_to_sample + i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(samp_x[pos_index[i]])
            plt.xlabel('1')
        plt.suptitle('sample images from batch  ' + str(batch_num))
        plt.show()


def image_transform_(IMG_SIZE, raw_X):
    """
    input num_of_images x H x W, image input must be grayscale
    MobileNetV2 requires certain image dimensions
    We use N x 61 x 61 formated images
    self.IMG_SIZE is a single number to change the images into, images must be square

    Parameters
    ----------
    raw_X :


    Returns
    -------


    """

    if len(raw_X.shape) == 4 and raw_X.shape[3] == 3:
        rgb_batch = copy.deepcopy(raw_X)
    else:
        rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
    rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
    rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2
    rgb_tensor = tf.image.resize(rgb_tensor, (IMG_SIZE, IMG_SIZE))  # resizing
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    return rgb_tensor


class ImageBatchGenerator(keras.utils.Sequence):
    """ """

    def __init__(self, batch_size, h5_file_list, label_key='labels'):
        h5_file_list = utils.make_list(h5_file_list, suppress_warning=True)
        num_frames_in_all_H5_files = get_total_frame_count(h5_file_list)
        file_inds_for_H5_extraction = batch_size_file_ind_selector(
            num_frames_in_all_H5_files, batch_size)
        subtract_for_index = reset_to_first_frame_for_each_file_ind(
            file_inds_for_H5_extraction)
        # self.to_fit = to_fit #set to True to return XY and False to return X
        self.label_key = label_key
        self.batch_size = batch_size
        self.H5_file_list = h5_file_list
        self.num_frames_in_all_H5_files = num_frames_in_all_H5_files
        self.file_inds_for_H5_extraction = file_inds_for_H5_extraction
        self.subtract_for_index = subtract_for_index
        self.IMG_SIZE = 96

    def __len__(self):
        return len(self.file_inds_for_H5_extraction)

    def __getitem__(self, num_2_extract):
        b = self.batch_size
        h = self.H5_file_list
        i = self.file_inds_for_H5_extraction
        H5_file = h[np.int(i[num_2_extract])]
        with h5py.File(H5_file, 'r') as H5:
            # H5 = h5py.File(H5_file, 'r')

            images = H5['images']
            num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
            raw_X = images[b * num_2_extract_mod:b * (num_2_extract_mod + 1)]
            rgb_tensor = self.image_transform(raw_X)

            labels_tmp = H5[self.label_key]
            raw_Y = labels_tmp[b * num_2_extract_mod:b * (num_2_extract_mod + 1)]
            H5.close()
        return rgb_tensor, raw_Y

    # def __getitem__(self, num_2_extract):
    #     b = self.batch_size
    #     h = self.H5_file_list
    #     i = self.file_inds_for_H5_extraction
    #     H5_file = h[np.int(i[num_2_extract])]
    #     H5 = h5py.File(H5_file, 'r')
    #     #  list(H5.keys())
    #
    #     images = H5['images']
    #     num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
    #     raw_X = images[b * num_2_extract_mod:b * (num_2_extract_mod + 1)]
    #     rgb_tensor = self.image_transform(raw_X)
    #
    #     # if self.to_fit:
    #     #   labels_tmp = H5[self.label_key]
    #     #   raw_Y = labels_tmp[b*num_2_extract_mod:b*(num_2_extract_mod+1)]
    #     #   return rgb_tensor, raw_Y
    #     # else:
    #     return rgb_tensor

    def getXandY(self, num_2_extract):
        """

        Parameters
        ----------
        num_2_extract :


        Returns
        -------


        """
        b = self.batch_size
        h = self.H5_file_list
        i = self.file_inds_for_H5_extraction
        H5_file = h[np.int(i[num_2_extract])]
        H5 = h5py.File(H5_file, 'r')
        #  list(H5.keys())

        images = H5['images']
        num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
        raw_X = images[b * num_2_extract_mod:b * (num_2_extract_mod + 1)]
        rgb_tensor = self.image_transform(raw_X)
        labels_tmp = H5[self.label_key]
        raw_Y = labels_tmp[b * num_2_extract_mod:b * (num_2_extract_mod + 1)]
        return rgb_tensor, raw_Y

    def image_transform(self, raw_X):
        """input num_of_images x H x W, image input must be grayscale
        MobileNetV2 requires certain image dimensions
        We use N x 61 x 61 formated images
        self.IMG_SIZE is a single number to change the images into, images must be square

        Parameters
        ----------
        raw_X :


        Returns
        -------


        """
        # rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
        # rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
        # rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2
        # rgb_tensor = tf.image.resize(rgb_tensor, (self.IMG_SIZE, self.IMG_SIZE))  # resizing
        # self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)
        # return rgb_tensor
        if len(raw_X.shape) == 4 and raw_X.shape[3] == 3:
            rgb_batch = copy.deepcopy(raw_X)
        else:
            rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
        rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
        rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2
        rgb_tensor = tf.image.resize(rgb_tensor, (self.IMG_SIZE, self.IMG_SIZE))  # resizing
        self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)
        return rgb_tensor

    def plot_batch_distribution(self):
        """ """
        # randomly select a batch and generate images and labels
        batch_num = np.random.choice(np.arange(0, self.__len__()))
        samp_x, samp_y = self.getXandY(batch_num)

        # look at the distribution of classes
        plt.pie([1 - np.mean(samp_y), np.mean(samp_y)],
                labels=['non-touch frames', 'touch frames'], autopct='%1.1f%%', )
        plt.title('class distribution from batch ' + str(batch_num))
        plt.show()

        # generate indices for positive and negative classes
        images_to_sample = 20
        neg_class = [i for i, val in enumerate(samp_y) if val == 0]
        pos_class = [i for i, val in enumerate(samp_y) if val == 1]
        neg_index = np.random.choice(neg_class, images_to_sample)
        pos_index = np.random.choice(pos_class, images_to_sample)

        # plot sample positive and negative class images
        plt.figure(figsize=(10, 10))
        samp_x = (samp_x + 1) / 2
        for i in range(images_to_sample):
            plt.subplot(5, 10, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            _ = plt.imshow(samp_x[neg_index[i]])
            plt.xlabel('0')

            plt.subplot(5, 10, images_to_sample + i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(samp_x[pos_index[i]])
            plt.xlabel('1')
        plt.suptitle('sample images from batch  ' + str(batch_num))
        plt.show()


def image_transform_(IMG_SIZE, raw_X):
    """
    input num_of_images x H x W, image input must be grayscale
    MobileNetV2 requires certain image dimensions
    We use N x 61 x 61 formated images
    self.IMG_SIZE is a single number to change the images into, images must be square

    Parameters
    ----------
    raw_X :


    Returns
    -------


    """

    if len(raw_X.shape) == 4 and raw_X.shape[3] == 3:
        rgb_batch = copy.deepcopy(raw_X)
    else:
        rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
    rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
    rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2
    rgb_tensor = tf.image.resize(rgb_tensor, (IMG_SIZE, IMG_SIZE))  # resizing
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    return rgb_tensor


# finished extractor........


# lstm_len = 5
#
# batch_size = 10
# num_2_extract = 24


class ImageBatchGenerator_simple(keras.utils.Sequence):

    def __init__(self, batch_size, h5_file_list, label_key='labels', IMG_SIZE=None):

        h5_file_list = utils.make_list(h5_file_list, suppress_warning=True)
        num_frames_in_all_H5_files = get_total_frame_count(h5_file_list)
        file_inds_for_H5_extraction = batch_size_file_ind_selector(num_frames_in_all_H5_files, batch_size)
        subtract_for_index = reset_to_first_frame_for_each_file_ind(file_inds_for_H5_extraction)
        self.label_key = label_key
        self.batch_size = batch_size
        self.H5_file_list = h5_file_list
        self.num_frames_in_all_H5_files = num_frames_in_all_H5_files
        self.file_inds_for_H5_extraction = file_inds_for_H5_extraction
        self.subtract_for_index = subtract_for_index
        self.IMG_SIZE = IMG_SIZE

    def image_transform(self, raw_X):
        if len(raw_X.shape) >= 4 and raw_X.shape[-1] == 3:
            rgb_batch = copy.deepcopy(raw_X)
        else:
            rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
        rgb_tensor = rgb_batch
        rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
        rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2
        if self.IMG_SIZE is not None:
            if len(raw_X.shape) <= 4:
                rgb_tensor = tf.image.resize(rgb_tensor, (self.IMG_SIZE, self.IMG_SIZE))  # resizing
            elif len(raw_X.shape) == 5:
                s = list(raw_X.shape)
                s[-3:-1] = [self.IMG_SIZE, self.IMG_SIZE]
                new_img = np.zeros(s).astype('float32')
                for lstm_i in range(raw_X.shape[1]):
                    new_img[:, lstm_i, :, :, :] = tf.image.resize(rgb_tensor[:, lstm_i, ...],
                                                                  (self.IMG_SIZE, self.IMG_SIZE)).numpy()
                rgb_tensor = new_img
            else:
                assert 1 == 0, "shape is screwed up..."
        # rgb_tensor = tf.cast(rgb_tensor, np.uint8)
        # self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)
        return rgb_tensor

    def __getitem__(self, num_2_extract):
        h = self.H5_file_list
        i = self.file_inds_for_H5_extraction
        H5_file = h[np.int(i[num_2_extract])]
        num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
        i1 = num_2_extract_mod * self.batch_size
        i2 = num_2_extract_mod * self.batch_size + self.batch_size
        with h5py.File(H5_file, 'r') as h:
            out = h['images'][i1:i2]
            out = self.image_transform(out)
            raw_Y = h[self.label_key][i1:i2]
            return out, raw_Y

    def __len__(self):
        return len(self.file_inds_for_H5_extraction)

    def getXandY(self, num_2_extract):
        rgb_tensor, raw_Y = self.__getitem__(num_2_extract)
        return rgb_tensor, raw_Y


class ImageBatchGenerator_LSTM(keras.utils.Sequence):
    """ """

    def __init__(self, lstm_len, batch_size, h5_file_list, label_key='labels', IMG_SIZE=96):
        assert lstm_len % 2 == 1, "number of images must be odd"
        h5_file_list = utils.make_list(h5_file_list, suppress_warning=True)
        num_frames_in_all_H5_files = get_total_frame_count(h5_file_list)
        file_inds_for_H5_extraction = batch_size_file_ind_selector(
            num_frames_in_all_H5_files, batch_size)
        subtract_for_index = reset_to_first_frame_for_each_file_ind(
            file_inds_for_H5_extraction)
        # self.to_fit = to_fit #set to True to return XY and False to return X
        self.label_key = label_key
        self.batch_size = batch_size
        self.H5_file_list = h5_file_list
        self.num_frames_in_all_H5_files = num_frames_in_all_H5_files
        self.file_inds_for_H5_extraction = file_inds_for_H5_extraction
        self.subtract_for_index = subtract_for_index
        self.IMG_SIZE = IMG_SIZE
        self.lstm_len = lstm_len
        self.get_frame_edges()

    def get_frame_edges(self):
        self.all_edges_list = []
        b = self.lstm_len // 2
        s = [b * 2, self.lstm_len, self.IMG_SIZE, self.IMG_SIZE, 3]
        for H5_file in self.H5_file_list:
            with h5py.File(H5_file,
                           'r') as h:  # 0(0, 1) 1(0) 3998(-1) 3999(-2, -1) ...  4000(0, 1) 4001(0) 7998(-1) 7999(-2, -1) #0,0     0,1     1,0     3998,-1     3999,-2     3999,-1
                full_edges_mask = np.ones(s)
                edge_ind = np.flip(np.arange(1, b + 1))
                for i in np.arange(1, b + 1):
                    full_edges_mask[i - 1, :edge_ind[i - 1], ...] = np.zeros_like(
                        full_edges_mask[i - 1, :edge_ind[i - 1], ...])
                    full_edges_mask[-i, -edge_ind[i - 1]:, ...] = np.zeros_like(
                        full_edges_mask[-i, -edge_ind[i - 1]:, ...])
                all_edges = []
                for i1, i2 in utils.loop_segments(h['frame_nums']):  # 0, 1, 3998, 3999 ; 4000, 4001, 7998, 7999; ...
                    edges = (np.asarray([[i1], [i2 - b]]) + np.arange(0, b).T).flatten()
                    all_edges.append(edges)
                all_edges = np.asarray(all_edges)
            self.all_edges_list.append(all_edges)
            full_edges_mask = full_edges_mask.astype(int)
            self.full_edges_mask = full_edges_mask == 0

    def __getitem__(self, num_2_extract):
        h = self.H5_file_list
        i = self.file_inds_for_H5_extraction
        all_edges = self.all_edges_list[np.int(i[num_2_extract])]
        H5_file = h[np.int(i[num_2_extract])]
        num_2_extract_mod = num_2_extract - self.subtract_forImageBatchGenerator_feature_array_index[num_2_extract]
        with h5py.File(H5_file, 'r') as h:
            b = self.lstm_len // 2
            tot_len = h['images'].shape[0]
            assert tot_len - b > self.batch_size, "reduce batch size to be less than total length of images minus floor(lstm_len) - 1, MAX->" + str(
                tot_len - b - 1)
            i1 = num_2_extract_mod * self.batch_size - b
            i2 = num_2_extract_mod * self.batch_size + self.batch_size + b
            edge_left_trigger = abs(min(i1, 0))
            edge_right_trigger = min(abs(min(tot_len - i2, 0)), b)
            x = h['images'][max(i1, 0):min(i2, tot_len)]
            if edge_left_trigger + edge_right_trigger > 0:  # in case of edge cases
                pad_shape = list(x.shape)
                pad_shape[0] = edge_left_trigger + edge_right_trigger
                pad = np.zeros(pad_shape).astype('uint8')
                if edge_left_trigger > edge_right_trigger:
                    x = np.concatenate((pad, x), axis=0)
                else:
                    x = np.concatenate((x, pad), axis=0)
            x = self.image_transform(x)
            s = list(x.shape)
            s.insert(1, self.lstm_len)
            out = np.zeros(s).astype('float32')  # before was uint8
            # out = tf.cast(out, tf.float32)

            for i in range(self.lstm_len):
                i1 = max(0, b - i)
                i2 = min(s[0], s[0] + b - i)
                i3 = max(0, i - b)
                i4 = min(s[0], s[0] + i - b)
                # print('take ', i3,' to ',  i4, ' and place in ', i1,' to ', i2)
                out[i1:i2, i, ...] = x[i3:i4, ...]
            out = out[b:s[0] - b, ...]
            i1 = num_2_extract_mod * self.batch_size
            i2 = num_2_extract_mod * self.batch_size + self.batch_size
            raw_Y = h[self.label_key][i1:i2]
            # black out edges from frame to frame
            adjust_these_edge_frames = np.intersect1d(all_edges.flatten(), np.arange(i1, i2))
            for atef in adjust_these_edge_frames:
                # mask_ind = np.where(atef == all_edges)[1][0]
                # out[atef] = out[atef] * (self.full_edges_mask[mask_ind]

                mask_ind = np.where(atef == all_edges)[1][0]
                mask_ = self.full_edges_mask[mask_ind]
                out[atef - i1][mask_] = -1
            return out, raw_Y

    # gray mask(set array to -1 not 0 ),DONE
    # doesnt fill the edges as expected, DONE
    # outputs format 0-255 not -1 to 1 DONE this was just custom code not from image_tools
    # need to test this with uneven frames
    def __len__(self):
        return len(self.file_inds_for_H5_extraction)

    def getXandY(self, num_2_extract):
        """

        Parameters
        ----------
        num_2_extract :


        Returns
        -------

        """
        rgb_tensor, raw_Y = self.__getitem__(num_2_extract)
        return rgb_tensor, raw_Y

    def image_transform(self, raw_X):
        """input num_of_images x H x W, image input must be grayscale
        MobileNetV2 requires certain image dimensions
        We use N x 61 x 61 formated images
        self.IMG_SIZE is a single number to change the images into, images must be square

        Parameters
        ----------
        raw_X :


        Returns
        -------


        """
        if len(raw_X.shape) == 4 and raw_X.shape[3] == 3:
            rgb_batch = copy.deepcopy(raw_X)
        else:
            rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
        rgb_tensor = rgb_batch
        # rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
        rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2 #commented before
        rgb_tensor = tf.image.resize(rgb_tensor, (self.IMG_SIZE, self.IMG_SIZE))  # resizing
        # rgb_tensor = tf.cast(rgb_tensor, np.uint8)# un commented before
        self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)
        return rgb_tensor


class ImageBatchGenerator_LSTM_2(keras.utils.Sequence):
    """ """

    def __init__(self, lstm_len, batch_size, h5_file_list, label_key='labels', IMG_SIZE=96):
        assert lstm_len % 2 == 1, "number of images must be odd"
        h5_file_list = utils.make_list(h5_file_list, suppress_warning=True)
        num_frames_in_all_H5_files = get_total_frame_count(h5_file_list)
        file_inds_for_H5_extraction = batch_size_file_ind_selector(
            num_frames_in_all_H5_files, batch_size)
        subtract_for_index = reset_to_first_frame_for_each_file_ind(
            file_inds_for_H5_extraction)
        # self.to_fit = to_fit #set to True to return XY and False to return X
        self.label_key = label_key
        self.batch_size = batch_size
        self.H5_file_list = h5_file_list
        self.num_frames_in_all_H5_files = num_frames_in_all_H5_files
        self.file_inds_for_H5_extraction = file_inds_for_H5_extraction
        self.subtract_for_index = subtract_for_index
        self.IMG_SIZE = IMG_SIZE
        self.lstm_len = lstm_len
        self.get_frame_edges()

    def get_frame_edges(self):
        self.all_edges_list = []
        b = self.lstm_len // 2
        s = [b * 2, self.lstm_len, self.IMG_SIZE, self.IMG_SIZE, 3]
        for H5_file in self.H5_file_list:
            with h5py.File(H5_file,
                           'r') as h:  # 0(0, 1) 1(0) 3998(-1) 3999(-2, -1) ...  4000(0, 1) 4001(0) 7998(-1) 7999(-2, -1) #0,0     0,1     1,0     3998,-1     3999,-2     3999,-1
                full_edges_mask = np.ones(s)
                edge_ind = np.flip(np.arange(1, b + 1))
                for i in np.arange(1, b + 1):
                    full_edges_mask[i - 1, :edge_ind[i - 1], ...] = np.zeros_like(
                        full_edges_mask[i - 1, :edge_ind[i - 1], ...])
                    full_edges_mask[-i, -edge_ind[i - 1]:, ...] = np.zeros_like(
                        full_edges_mask[-i, -edge_ind[i - 1]:, ...])
                all_edges = []
                for i1, i2 in utils.loop_segments(h['frame_nums']):  # 0, 1, 3998, 3999 ; 4000, 4001, 7998, 7999; ...
                    edges = (np.asarray([[i1], [i2 - b]]) + np.arange(0, b).T).flatten()
                    all_edges.append(edges)
                all_edges = np.asarray(all_edges)
            self.all_edges_list.append(all_edges)
            full_edges_mask = full_edges_mask.astype(int)
            self.full_edges_mask = full_edges_mask == 0

    def __getitem__(self, num_2_extract):
        h = self.H5_file_list
        i = self.file_inds_for_H5_extraction
        all_edges = self.all_edges_list[np.int(i[num_2_extract])]
        H5_file = h[np.int(i[num_2_extract])]
        num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
        with h5py.File(H5_file, 'r') as h:
            b = self.lstm_len // 2
            tot_len = h['images'].shape[0]
            assert tot_len - b > self.batch_size, "reduce batch size to be less than total length of images minus floor(lstm_len) - 1, MAX->" + str(
                tot_len - b - 1)
            i1 = num_2_extract_mod * self.batch_size - b
            i2 = num_2_extract_mod * self.batch_size + self.batch_size + b
            edge_left_trigger = abs(min(i1, 0))
            edge_right_trigger = min(abs(min(tot_len - i2, 0)), b)
            x = h['images'][max(i1, 0):min(i2, tot_len)]
            if edge_left_trigger + edge_right_trigger > 0:  # in case of edge cases
                pad_shape = list(x.shape)
                pad_shape[0] = edge_left_trigger + edge_right_trigger
                pad = np.zeros(pad_shape).astype('uint8')
                if edge_left_trigger > edge_right_trigger:
                    x = np.concatenate((pad, x), axis=0)
                else:
                    x = np.concatenate((x, pad), axis=0)
            x = self.image_transform(x)
            s = list(x.shape)
            s.insert(1, self.lstm_len)
            out = np.zeros(s).astype('float32')  # before was uint8
            # out = tf.cast(out, tf.float32)

            for i in range(self.lstm_len):
                i1 = max(0, b - i)
                i2 = min(s[0], s[0] + b - i)
                i3 = max(0, i - b)
                i4 = min(s[0], s[0] + i - b)
                # print('take ', i3,' to ',  i4, ' and place in ', i1,' to ', i2)
                out[i1:i2, i, ...] = x[i3:i4, ...]
            out = out[b:s[0] - b, ...]
            i1 = num_2_extract_mod * self.batch_size
            i2 = num_2_extract_mod * self.batch_size + self.batch_size
            raw_Y = h[self.label_key][i1:i2]
            # black out edges from frame to frame
            adjust_these_edge_frames = np.intersect1d(all_edges.flatten(), np.arange(i1, i2))
            for atef in adjust_these_edge_frames:
                # mask_ind = np.where(atef == all_edges)[1][0]
                # out[atef] = out[atef] * (self.full_edges_mask[mask_ind]

                mask_ind = np.where(atef == all_edges)[1][0]
                mask_ = self.full_edges_mask[mask_ind]
                out[atef - i1][mask_] = -1
            return out, raw_Y

    # gray mask(set array to -1 not 0 ),DONE
    # doesnt fill the edges as expected, DONE
    # outputs format 0-255 not -1 to 1 DONE this was just custom code not from image_tools
    # need to test this with uneven frames
    def __len__(self):
        return len(self.file_inds_for_H5_extraction)

    def getXandY(self, num_2_extract):
        """

        Parameters
        ----------
        num_2_extract :


        Returns
        -------

        """
        rgb_tensor, raw_Y = self.__getitem__(num_2_extract)
        return rgb_tensor, raw_Y

    def image_transform(self, raw_X):
        """input num_of_images x H x W, image input must be grayscale
        MobileNetV2 requires certain image dimensions
        We use N x 61 x 61 formated images
        self.IMG_SIZE is a single number to change the images into, images must be square

        Parameters
        ----------
        raw_X :


        Returns
        -------


        """
        if len(raw_X.shape) == 4 and raw_X.shape[3] == 3:
            rgb_batch = copy.deepcopy(raw_X)
        else:
            rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
        rgb_tensor = rgb_batch
        # rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
        rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2 #commented before
        rgb_tensor = tf.image.resize(rgb_tensor, (self.IMG_SIZE, self.IMG_SIZE))  # resizing
        # rgb_tensor = tf.cast(rgb_tensor, np.uint8)# un commented before
        self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)
        return rgb_tensor


def convert_h5_to_LSTM_h5(h5_file_list, lstm_h5_name, lstm_len=7, batch_size=100, label_key='labels', IMG_SIZE=96,
                          disable_tqdm=False):
    """
    convert any H5 files with the 3 oclor channels into the LSTM format, then use iamge_tools.ImageBatchGenerator_simple
    to directly draw from the H5 files to save time with converting them in the generator each time
    Parameters
    ----------
    h5_file_list :
    lstm_h5_name :
    lstm_len :
    batch_size :
    label_key :
    IMG_SIZE :
    disable_tqdm :

    Returns
    -------

    """
    utils.make_list(h5_file_list, suppress_warning=True)
    G = ImageBatchGenerator_LSTM(lstm_len, batch_size, h5_file_list, label_key=label_key, IMG_SIZE=IMG_SIZE)
    Path(os.path.dirname(lstm_h5_name)).mkdir(parents=True, exist_ok=True)
    h5creator = h5_iterative_creator(lstm_h5_name)
    for k in tqdm(range(G.__len__()), disable=disable_tqdm):
        x, y = G.__getitem__(k)
        h5creator.add_to_h5(x, y)
    frame_nums = get_h5_key_and_concatenate(h5_file_list, 'frame_nums')
    utils.force_write_to_h5(h5creator.h5_full_file_name, frame_nums, 'frame_nums')


#
# class ImageBatchGenerator_feature_array(keras.utils.Sequence):
#     """ """
#
#     def __init__(self, lstm_len, batch_size, h5_file_list, label_key='labels', feature_len=2048,
#                  label_index_to_lstm_len=None):
#         assert lstm_len % 2 == 1, "number of images must be odd"
#         if label_index_to_lstm_len is None:
#             label_index_to_lstm_len = lstm_len // 2  # in the middle
#         h5_file_list = utils.make_list(h5_file_list, suppress_warning=True)
#         num_frames_in_all_H5_files = get_total_frame_count(h5_file_list)
#         file_inds_for_H5_extraction = batch_size_file_ind_selector(
#             num_frames_in_all_H5_files, batch_size)
#         subtract_for_index = reset_to_first_frame_for_each_file_ind(
#             file_inds_for_H5_extraction)
#         # self.to_fit = to_fit #set to True to return XY and False to return X
#         self.label_key = label_key
#         self.batch_size = batch_size
#         self.H5_file_list = h5_file_list
#         self.num_frames_in_all_H5_files = num_frames_in_all_H5_files
#         self.file_inds_for_H5_extraction = file_inds_for_H5_extraction
#         self.subtract_for_index = subtract_for_index
#         self.label_index_to_lstm_len = label_index_to_lstm_len
#         self.lstm_len = lstm_len
#         self.feature_len = feature_len
#         self.get_frame_edges()
#
#     def get_frame_edges(self):
#         self.all_edges_list = []
#         b = self.lstm_len // 2
#         self.lstm_len
#         self.feature_len
#         s = [b * 2, self.lstm_len, self.feature_len]
#         for H5_file in self.H5_file_list:
#             with h5py.File(H5_file,
#                            'r') as h:  # 0(0, 1) 1(0) 3998(-1) 3999(-2, -1) ...  4000(0, 1) 4001(0) 7998(-1) 7999(-2, -1) #0,0     0,1     1,0     3998,-1     3999,-2     3999,-1
#                 full_edges_mask = np.ones(s)
#                 edge_ind = np.flip(np.arange(1, b + 1))
#                 for i in np.arange(1, b + 1):
#                     full_edges_mask[i - 1, :edge_ind[i - 1], ...] = np.zeros_like(
#                         full_edges_mask[i - 1, :edge_ind[i - 1], ...])
#                     full_edges_mask[-i, -edge_ind[i - 1]:, ...] = np.zeros_like(
#                         full_edges_mask[-i, -edge_ind[i - 1]:, ...])
#                 all_edges = []
#                 for i1, i2 in utils.loop_segments(h['frame_nums']):  # 0, 1, 3998, 3999 ; 4000, 4001, 7998, 7999; ...
#                     edges = (np.asarray([[i1], [i2 - b]]) + np.arange(0, b).T).flatten()
#                     all_edges.append(edges)
#                 all_edges = np.asarray(all_edges)
#             self.all_edges_list.append(all_edges)
#             full_edges_mask = full_edges_mask.astype(int)
#             self.full_edges_mask = full_edges_mask == 0
#
#     def __getitem__(self, num_2_extract):
#         h = self.H5_file_list
#         i = self.file_inds_for_H5_extraction
#         all_edges = self.all_edges_list[np.int(i[num_2_extract])]
#         H5_file = h[np.int(i[num_2_extract])]
#         num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
#         with h5py.File(H5_file, 'r') as h:
#             b = self.lstm_len // 2
#             tot_len = h['images'].shape[0]
#             assert tot_len - b > self.batch_size, "reduce batch size to be less than total length of images minus floor(lstm_len) - 1, MAX->" + str(
#                 tot_len - b - 1)
#             i1 = num_2_extract_mod * self.batch_size - b
#             i2 = num_2_extract_mod * self.batch_size + self.batch_size + b
#             edge_left_trigger = abs(min(i1, 0))
#             edge_right_trigger = min(abs(min(tot_len - i2, 0)), b)
#             x = h['images'][max(i1, 0):min(i2, tot_len)]
#             if edge_left_trigger + edge_right_trigger > 0:  # in case of edge cases
#                 pad_shape = list(x.shape)
#                 pad_shape[0] = edge_left_trigger + edge_right_trigger
#                 pad = np.zeros(pad_shape).astype('uint8')
#                 if edge_left_trigger > edge_right_trigger:
#                     x = np.concatenate((pad, x), axis=0)
#                 else:
#                     x = np.concatenate((x, pad), axis=0)
#             x = self.image_transform(x)
#             s = list(x.shape)
#             s.insert(1, self.lstm_len)
#             out = np.zeros(s).astype('float32')  # before was uint8
#             # out = tf.cast(out, tf.float32)
#
#             for i in range(self.lstm_len):
#                 i1 = max(0, b - i)
#                 i2 = min(s[0], s[0] + b - i)
#                 i3 = max(0, i - b)
#                 i4 = min(s[0], s[0] + i - b)
#                 # print('take ', i3,' to ',  i4, ' and place in ', i1,' to ', i2)
#                 out[i1:i2, i, ...] = x[i3:i4, ...]
#             out = out[b:s[0] - b, ...]
#             i1 = num_2_extract_mod * self.batch_size
#             i2 = num_2_extract_mod * self.batch_size + self.batch_size
#             raw_Y = h[self.label_key][i1:i2]
#             # black out edges from frame to frame
#             adjust_these_edge_frames = np.intersect1d(all_edges.flatten(), np.arange(i1, i2))
#             for atef in adjust_these_edge_frames:
#                 # mask_ind = np.where(atef == all_edges)[1][0]
#                 # out[atef] = out[atef] * (self.full_edges_mask[mask_ind]
#
#                 mask_ind = np.where(atef == all_edges)[1][0]
#                 mask_ = self.full_edges_mask[mask_ind]
#                 out[atef - i1][mask_] = -1
#             return out, raw_Y
#
#     # gray mask(set array to -1 not 0 ),DONE
#     # doesnt fill the edges as expected, DONE
#     # outputs format 0-255 not -1 to 1 DONE this was just custom code not from image_tools
#     # need to test this with uneven frames
#     def __len__(self):
#         return len(self.file_inds_for_H5_extraction)
#
#     def getXandY(self, num_2_extract):
#         """
#
#         Parameters
#         ----------
#         num_2_extract :
#
#
#         Returns
#         -------
#
#         """
#         rgb_tensor, raw_Y = self.__getitem__(num_2_extract)
#         return rgb_tensor, raw_Y
#
#     def image_transform(self, raw_X):
#         """input num_of_images x H x W, image input must be grayscale
#         MobileNetV2 requires certain image dimensions
#         We use N x 61 x 61 formated images
#         self.IMG_SIZE is a single number to change the images into, images must be square
#
#         Parameters
#         ----------
#         raw_X :
#
#
#         Returns
#         -------
#
#
#         """
#         rgb_batch = copy.deepcopy(raw_X)
#         # if len(raw_X.shape) == 4 and raw_X.shape[3] == 3:
#         #     rgb_batch = copy.deepcopy(raw_X)
#         # else:
#         #     rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
#         rgb_tensor = rgb_batch
#         # rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
#         rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2 #commented before
#         # rgb_tensor = tf.image.resize(rgb_tensor, (self.feature_len))  # resizing
#         # rgb_tensor = tf.cast(rgb_tensor, np.uint8)# un commented before
#         self.IMG_SHAPE = (self.feature_len)
#         return rgb_tensor

class ImageBatchGenerator_feature_array(keras.utils.Sequence):

    def __init__(self, time_length, batch_size, h5_file_list, label_key='labels', feature_len=None,
                 label_index_to_lstm_len=None, edge_value=-1, remove_any_time_points_with_edges=True):
        """

        Parameters
        ----------
        time_length : total time points
        batch_size : batch output for generator, if set to None then will default to all frames which may use all your RAM
        h5_file_list : list of h5 strings or single h5 string
        label_key : where y output comes from
        feature_len : length of the features per time point
        label_index_to_lstm_len : determines look back and look forward index refers to where the 'current' time point is
        within the range of look_back_len; e.g. look_back_len = 7 label_index_to_lstm_len = 3 (middle index of 7) then
        time point 0 will be at 3 and index 0, 1, 2 will be the past values and index 4, 5, 6 will be the future values.
        look_back_len = 7 label_index_to_lstm_len = 0 (first index) then current time point will be at index 0 and all
        other time point (1, 2, 3, 4, 5, 6) will be future values. Default is middle time point
        edge_value : what to replace the edge values with, when time shifting you will have edges with no value, this
        will replace those values with this number.
        remove_any_time_points_with_edges : if true then batch size will not be the actual batch size it will be batch
        size - the number of time points with edges in them, x and y will still match and this method is preferred for
        training due to it not including unknown values.
        """
        assert time_length % 2 == 1, "number of images must be odd"
        if label_index_to_lstm_len is None:
            label_index_to_lstm_len = time_length // 2  # in the middle
        h5_file_list = utils.make_list(h5_file_list, suppress_warning=True)
        num_frames_in_all_H5_files = get_total_frame_count(h5_file_list)
        if batch_size is None:
            batch_size = int(np.sum(num_frames_in_all_H5_files))

        file_inds_for_H5_extraction = batch_size_file_ind_selector(
            num_frames_in_all_H5_files, batch_size)
        subtract_for_index = reset_to_first_frame_for_each_file_ind(
            file_inds_for_H5_extraction)
        self.remove_any_time_points_with_edges = remove_any_time_points_with_edges
        self.label_key = label_key

        self.batch_size = batch_size
        self.H5_file_list = h5_file_list
        self.num_frames_in_all_H5_files = num_frames_in_all_H5_files
        self.file_inds_for_H5_extraction = file_inds_for_H5_extraction
        self.subtract_for_index = subtract_for_index
        self.label_index_to_lstm_len = label_index_to_lstm_len
        self.lstm_len = time_length
        self.feature_len = feature_len
        self.edge_value = edge_value
        if remove_any_time_points_with_edges:
            self.edge_value = np.nan
            print(
                'remove_any_time_points_with_edges == True : forcing edge_value to np.nan to aid in removing these time points')

        self.get_frame_edges()
        # self.full_edges_mask = self.full_edges_mask - (self.lstm_len // 2 - self.label_index_to_lstm_len)

    def __getitem__(self, num_2_extract):
        h = self.H5_file_list
        i = self.file_inds_for_H5_extraction
        all_edges = self.all_edges_list[np.int(i[num_2_extract])]
        H5_file = h[np.int(i[num_2_extract])]
        num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]

        with h5py.File(H5_file, 'r') as h:
            b = self.lstm_len // 2
            tot_len = h['images'].shape[0]

            # assert tot_len - b > self.batch_size, "reduce batch size to be less than total length of images minus floor(lstm_len) - 1, MAX->" + str(
            #     tot_len - b - 1)

            i1 = num_2_extract_mod * self.batch_size - b
            i2 = num_2_extract_mod * self.batch_size + self.batch_size + b
            edge_left_trigger = abs(min(i1, 0))
            edge_right_trigger = min(abs(min(tot_len - i2, 0)), b)
            x = h['images'][max(i1, 0):min(i2, tot_len)]
            if edge_left_trigger + edge_right_trigger > 0:  # in case of edge cases
                pad_shape = list(x.shape)
                pad_shape[0] = edge_left_trigger + edge_right_trigger
                pad = np.zeros(pad_shape).astype('float32')
                if edge_left_trigger > edge_right_trigger:
                    x = np.concatenate((pad, x), axis=0)
                else:
                    x = np.concatenate((x, pad), axis=0)

            s = list(x.shape)
            s.insert(1, self.lstm_len)
            out = np.zeros(s).astype('float32')  # before was uint8
            Z = self.label_index_to_lstm_len - self.lstm_len // 2
            for i in range(self.lstm_len):
                i_temp = i
                i = i - Z
                i1 = max(0, b - i)
                i2 = min(s[0], s[0] + b - i)
                i3 = max(0, i - b)
                i4 = min(s[0], s[0] + i - b)
                # print('take ', i3, ' to ', i4, ' and place in ', i1, ' to ', i2)
                out[i1:i2, i_temp, ...] = x[i3:i4, ...]

            out = out[b:s[0] - b, ...]
            i1 = num_2_extract_mod * self.batch_size
            i2 = num_2_extract_mod * self.batch_size + self.batch_size
            raw_Y = h[self.label_key][i1:i2]

            adjust_these_edge_frames = np.intersect1d(all_edges.flatten(), np.arange(i1, i2))
            b2 = b - self.label_index_to_lstm_len  # used to adjust mask postion based on where the center value is
            for atef in adjust_these_edge_frames:
                mask_ind = np.where(atef == all_edges)[1][0]
                mask_ind = mask_ind - b2
                mask_ind = mask_ind % self.full_edges_mask.shape[0]  # wrap around index

                mask_ = self.full_edges_mask[mask_ind]
                mask_ = mask_ == 1
                out_ind = atef + i1 - b2
                out_ind = out_ind % out.shape[0]  # wrap around index
                out[out_ind][mask_] = self.edge_value

            s = out.shape
            out = np.reshape(out, (s[0], s[1] * s[2]))
            if self.remove_any_time_points_with_edges:
                keep_inds = ~np.isnan(np.mean(out, axis=1))
                out = out[keep_inds]
                raw_Y = raw_Y[keep_inds]

            return out, raw_Y

    def __len__(self):
        return len(self.file_inds_for_H5_extraction)

    def getXandY(self, num_2_extract):
        """

        Parameters
        ----------
        num_2_extract :


        Returns
        -------

        """
        rgb_tensor, raw_Y = self.__getitem__(num_2_extract)
        return rgb_tensor, raw_Y

    def image_transform(self, raw_X):
        """input num_of_images x H x W, image input must be grayscale
        MobileNetV2 requires certain image dimensions
        We use N x 61 x 61 formated images
        self.IMG_SIZE is a single number to change the images into, images must be square

        Parameters
        ----------
        raw_X :


        Returns
        -------


        """
        # kept this cause this is the format of the image generators I know this is redundant
        rgb_batch = copy.deepcopy(raw_X)
        rgb_tensor = rgb_batch
        self.IMG_SHAPE = (self.feature_len)
        return rgb_tensor

    def get_frame_edges(self):
        self.all_edges_list = []
        b = self.lstm_len // 2

        s = [b * 2, self.lstm_len, self.feature_len]
        for H5_file in self.H5_file_list:
            with h5py.File(H5_file, 'r') as h:
                full_edges_mask = np.ones(s)
                tmp1 = np.arange(1, self.lstm_len)
                front_edge = tmp1[:self.label_index_to_lstm_len]
                back_edge = tmp1[:self.lstm_len - self.label_index_to_lstm_len - 1]

                edge_ind = np.flip(front_edge)
                for i in front_edge:
                    # print(i - 1, ':', edge_ind[i - 1])
                    # print(full_edges_mask[i - 1, :edge_ind[i - 1], ...].shape)
                    # print('\n')
                    full_edges_mask[i - 1, :edge_ind[i - 1], ...] = np.zeros_like(
                        full_edges_mask[i - 1, :edge_ind[i - 1], ...])

                edge_ind = np.flip(back_edge)
                for i in back_edge:
                    # print(-i, -edge_ind[i - 1], ':')
                    # print(full_edges_mask[-i, -edge_ind[i - 1]:, ...].shape)
                    # print('\n')
                    full_edges_mask[-i, -edge_ind[i - 1]:, ...] = np.zeros_like(
                        full_edges_mask[-i, -edge_ind[i - 1]:, ...])

                all_edges = []
                for i1, i2 in utils.loop_segments(h['frame_nums']):  # 0, 1, 3998, 3999 ; 4000, 4001, 7998, 7999; ...
                    edges = (np.asarray([[i1], [i2 - b]]) + np.arange(0, b).T).flatten()
                    all_edges.append(edges)

                all_edges = np.asarray(all_edges)
            self.all_edges_list.append(all_edges)
            full_edges_mask = full_edges_mask.astype(int)
            self.full_edges_mask = full_edges_mask == 0


def rename_h5_key_from_images_to_feature(h5_in):
    if isinstance(h5_in, list):
        for k in h5_in:
            rename_h5_key_from_images_to_feature(k)
        return

    if utils.h5_key_exists(h5_in, 'images'):
        print("""'images' exists, changing it to 'FD__original'""")
        with h5py.File(h5_in, 'r+') as h:
            h['FD__original'] = h['images'][:]
            del h['images']
    else:
        print("""'images' key does not exist, skipping""")


# apply the below on the files and combine the things we need to combine
def combine_and_index_feature_data(h5s_to_combine, new_h5, label_key='labels', feature_key='FD__original'):
    h5c = h5_iterative_creator(new_h5,
                               overwrite_if_file_exists=True,
                               max_img_height=1,
                               max_img_width=2048,
                               close_and_open_on_each_iteration=True,
                               color_channel=False,
                               add_to_existing_H5=False,
                               ignore_image_range_warning=False,
                               dtype_img=h5py.h5t.IEEE_F32LE,
                               dtype_labels=h5py.h5t.IEEE_F32LE)

    h5_lengths = []
    h5_inds_each = []
    inds_to_h5_file_names = []
    frame_nums_all = []
    for i, k in enumerate(tqdm(h5s_to_combine)):
        features = get_h5_key_and_concatenate(k, feature_key)
        y = get_h5_key_and_concatenate(k, label_key)

        h5c.add_to_h5(features, y)

        h5_lengths.append(len(y))
        h5_inds_each.append(np.arange(len(y)))
        inds_to_h5_file_names.append(np.zeros(len(y)).astype(int) + i)

        frame_nums_all.append(get_h5_key_and_concatenate(k, 'frame_nums').astype(int))

    h5s_to_combine = [n.encode("ascii", "ignore") for n in h5s_to_combine]
    rename_h5_key_from_images_to_feature(new_h5)

    with h5py.File(new_h5, 'r+') as h:
        h['all_combined_h5_names'] = h5s_to_combine
        h['h5_lengths'] = np.asarray(h5_lengths).astype(int)
        h['h5_inds_all'] = np.arange(np.sum(h5_lengths)).astype(int)
        h['h5_inds_each'] = np.hstack(h5_inds_each).astype(int)
        h['inds_to_h5_file_names'] = np.hstack(inds_to_h5_file_names).astype(int)
        h['frame_nums'] = np.hstack(frame_nums_all).astype(int)
        h['has_data_been_randomized'] = False


def randomize_original_feature_data_and_inds(h5_in, rand_seed=1):
    assert rand_seed is not None, """random seed can't be None, otherwise inds and data will not match!!!"""

    with h5py.File(h5_in, 'r+') as h:
        np.random.seed(rand_seed)
        h['h5_inds_all'][:] = np.random.permutation(h['h5_inds_all'][:])
        np.random.seed(rand_seed)
        h['h5_inds_each'][:] = np.random.permutation(h['h5_inds_each'][:])
        np.random.seed(rand_seed)
        h['inds_to_h5_file_names'][:] = np.random.permutation(h['inds_to_h5_file_names'][:])
        np.random.seed(rand_seed)
        h['FD__original'][:, :] = np.random.permutation(h['FD__original'][:, :])

        h['has_data_been_randomized'] = True


def check_if_permuted_h5_matches_og_h5_indexes(h5_1, ind):
    with h5py.File(h5_1, 'r') as h:
        i1 = np.where(h['h5_inds_all'][:] == ind)[0]
        x1 = h['inds_to_h5_file_names'][i1]
        print(h['all_combined_h5_names'][x1])
        print(h['h5_inds_all'][i1])
        print(h['h5_inds_each'][i1])
        print(x1)
        print(h['FD__original'][i1, :])


def get_rand_bool_inds(h5_in, split_segs=None, rand_seed=1, label_key='labels', write_to_h5=True):
    assert rand_seed is not None, """random seed can't be None, otherwise inds and data will not match!!!"""
    if split_segs is None:
        split_segs = get_h5_key_and_concatenate(h5_in, 'split_segs')
    L = len(get_h5_key_and_concatenate(h5_in, label_key))
    np.random.seed(rand_seed)
    permuted_inds = np.random.permutation(np.arange(L))
    all_bool_inds = np.zeros([split_segs.shape[1], L], dtype=bool)
    for i, k in enumerate(split_segs.T):
        i1, i2 = k[0], k[1]
        perm_ind = np.sort(permuted_inds[i1:i2])
        permuted_inds[i1:i2] = perm_ind
        for kk in perm_ind:
            all_bool_inds[i, kk] = True
    if write_to_h5:
        utils.overwrite_h5_key(h5_in, 'permuted_inds', permuted_inds)
        utils.overwrite_h5_key(h5_in, 'all_bool_inds', all_bool_inds)
    return permuted_inds, np.asarray(all_bool_inds)


def divide_data_indexing(h5_in, split_names=['train', 'val', 'test'], split_percent_single=[0.7, 0.15, 0.15],
                         max_len_each_set=10000, label_key='labels', write_to_h5=True):
    L = len(get_h5_key_and_concatenate(h5_in, label_key))
    num_divs = int(np.ceil(L / max_len_each_set))
    split_percent_single = list(split_percent_single/np.sum(split_percent_single))

    split_percentages = split_percent_single * num_divs
    array_inds = np.arange(L)
    split_percentages = split_percentages / np.sum(split_percentages)
    segment_inds = list(np.split(array_inds, np.ceil(L * np.cumsum(split_percentages[:-1])).astype('int')))
    tmp1 = [0]
    _ = [tmp1.append(k[-1]) for k in segment_inds]
    split_segs = np.asarray(utils.loop_segments(np.diff(tmp1), returnaslist=True))

    tmp2 = split_names * num_divs
    tmp3 = np.repeat(list(range(num_divs)), 3)
    split_segs_names = [i1 + '_num_' + str(i2) + '_percent_' + str(int(100*i3)) for i1, i2, i3 in
                        zip(tmp2, tmp3, split_percent_single * num_divs)]
    if write_to_h5:
        utils.overwrite_h5_key(h5_in, 'split_segs', split_segs)
        utils.overwrite_h5_key(h5_in, 'split_segs_names', utils.convert_list_of_strings_for_h5(split_segs_names))

    return split_segs, split_segs_names

def convert_to_3lag(f, f2):
    """
    Parameters
    ----------
    f : base file
    f2 : new 3lag file (doesn't exist yet
    """
    assert f != f2, """can't be the same file"""
    if os.path.isfile(f2):
        utils.open_folder(os.path.dirname(f2))
        assert False, """\nthe 3lag file you want to create already exists please delete it you you want to overwrite\n""" + f2 + "\nopening folder..."
    utils.stack_lag_h5_maker(f, f2, buffer=2, shift_to_the_right_by=0)
    utils.copy_over_all_non_image_keys(f, f2)
