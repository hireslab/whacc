import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import h5py
import copy
from tqdm import tqdm
import time
import os


class h5_iterative_creator():
    """
    Create an H5 file using a for loop easily. used to create the augmented H5 file for training

    Attributes:

    :param h5_new_full_file_name: full path name to your H5 file to be created
    :type h5_new_full_file_name: string
    :param overwrite_if_file_exists: overwrites the h5 file if it already exists
    :type overwrite_if_file_exists: bool
    :param max_img_height: default 61, only the max size, can be larger in case you are going to have larger images
    :type max_img_height: int
    :param max_img_width: default 61, only the max size, can be larger in case you are going to have larger images
    :type max_img_width: int
    :param close_and_open_on_each_iteration: default True, this prevents the user form forgetting to close H5 which
        can lead to corruption.
    :type close_and_open_on_each_iteration: bool
    """

    def __init__(self, h5_new_full_file_name,
                 overwrite_if_file_exists=False,
                 max_img_height=61,
                 max_img_width=61,
                 close_and_open_on_each_iteration=True):

        if not close_and_open_on_each_iteration:
            print('**remember to CLOSE the H5 file when you are done!!!**')
        if overwrite_if_file_exists and os.path.isfile(h5_new_full_file_name):
            os.remove(h5_new_full_file_name)
        self.h5_full_file_name = h5_new_full_file_name
        self.hf_file = h5py.File(h5_new_full_file_name, "w")

        self.max_img_height = max_img_height
        self.max_img_width = max_img_width
        self._went_thorugh_create_h5 = False
        self.close_it = close_and_open_on_each_iteration
        if self.close_it:
            self.hf_file.close()

    def add_to_h5(self, images, labels):
        if self.close_it:
            self.open_or_close_h5('r+')
        if self._went_thorugh_create_h5:  # already initalized with the correct size
            self._add_next_chunk_to_h5(images, labels)
        else:
            self._create_h5(images, labels)
        if self.close_it:
            self.open_or_close_h5('close')

    def _create_h5(self, images, labels):
        self.hf_file.create_dataset("multiplier", [1], h5py.h5t.STD_I32LE, data=images.shape[0])
        self.hf_file.create_dataset('images',
                                    np.shape(images),
                                    h5py.h5t.STD_U8BE,
                                    maxshape=(None, self.max_img_height, self.max_img_width, 3),
                                    chunks=True)
        self.hf_file.create_dataset('labels',
                                    np.shape(labels),
                                    h5py.h5t.STD_I32LE,
                                    maxshape=(None,),
                                    chunks=True)
        self._went_thorugh_create_h5 = True

    def _add_next_chunk_to_h5(self, images, labels):
        self.hf_file['images'].resize(self.hf_file['images'].shape[0] + images.shape[0], axis=0)
        self.hf_file['labels'].resize(self.hf_file['labels'].shape[0] + labels.shape[0], axis=0)

        self.hf_file['images'][-images.shape[0]:] = images
        self.hf_file['labels'][-labels.shape[0]:] = labels

    def read_h5(self):
        self.open_or_close_h5('r')
        print('''**remember to CLOSE the H5 file when you are done!!!** with ".close_h5()" method''')

    def close_h5(self):
        self.open_or_close_h5('close')

    def open_or_close_h5(self, mode_='r'):
        """

        :param mode_: mode can be H5py modes 'r', 'r+' 'w' (w overwrites file!) etc OR 'close' to
        # ensure it is closed. separate function to prevent a bunch of try statements
        :type mode_: str
        """
        try:
            self.hf_file.close()
        finally:
            if mode_.lower() != 'close':
                self.hf_file = h5py.File(self.h5_full_file_name, mode_)


#
def augment_helper(keras_datagen, num_aug_ims, num_reg_ims, in_img, in_label):
    """
    :param keras_datagen: --from keras.preprocessing.image import ImageDataGenerator-- keras_datagen = ImageDataGenerator(...)
    :type keras_datagen: keras.preprocessing.image.ImageDataGenerator
    :param num_aug_ims: number of augmented images to generate from single input image
    :type num_aug_ims: int
    :param num_reg_ims: number of copies of in_img to produce. will be stacked at the beginning of all_augment variable.
        Use dot see augmentation when testing and can be useful if splitting into many H5s if you want an original in each.
    :type num_reg_ims: int
    :param in_img: numpy array either 3D with color channel for the last dim ot 2D
    :type in_img: numpy array
    :param in_label: the label associate with in_img. simply repeats it creating 'out_labels' the be size of 'all_augment'
    :type in_label: int
    :return:
        - all_augment - augmented images stacked
        - out_labels - repeated array of length len(all_augment) created from in_label
    :rtype:
        - all_augment - numpy array
        - out_labels - numpy array
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


def img_unstacker(img_array, num_frames_wide=8):
    """
    unstacks image stack and combines them into one large image for easy display. reads left to right and then top to bottom.
    :param img_array: stacked image array
    :type img_array: numpy array
    :param num_frames_wide: width of destacked image. if = 8 with input 20 images it will be 8 wide 3 long and 4 blank images
    :type num_frames_wide: int
    :return: - im_stack - one large image
    :rtype: numpy array
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
    """
    This is used to transform batch generated images [-1 1] to the original image [0,255] for plotting

    """
    image = tf.cast((x + 1) * 127.5, tf.uint8)
    return image


def predict_multiple_H5_files(H5_file_list, model_2_load, append_model_and_labels_to_name_string=False,
                              batch_size=1000, model_2_load_is_model=False, save_on=False,
                              label_save_name=None, disable_TQDM=False,
                              save_labels_to_this_h5_file_instead=None) -> object:
    """
    :param H5_file_list: list of string(s) of H5 file full paths
    :type H5_file_list: list
    :param model_2_load: either full path to model folder ending with ".ckpt" OR the loaded model itself. if the later,
        the user MUST set "model_2_load_is_model" is True and "label_save_name" must be explicitly defined (when using model
        path we use the model name to name the labels).
    :type model_2_load:
    :param append_model_and_labels_to_name_string: if True label_save_name =  'MODEL__' + label_save_name + '__labels',
        it is a simple way to keep track of labels form many models in a single H5 file. also make sit easier to find
        those labels for later processing.
    :type append_model_and_labels_to_name_string: bool
    :param batch_size: number of images to process per batch,  -- slower prediction speeds << ideal predictionsspeed <<
        memory issues and crashes -- 1000 is normally pretty good on Google CoLab
    :type batch_size: int
    :param model_2_load_is_model:lets the program know if you are directly inserting a model (instead of a path to model folder) 
    :type model_2_load_is_model: bool 
    :param save_on: saves to H5 file. either the original H5 (image source) or new H5 if a path to "save_labels_to_this_h5_file_instead"
        is given
    :type save_on: bool
    :param label_save_name: h5 file key used to save the labels to, default is 'MODEL__' + **model_name** + '__labels'
    :type label_save_name: string
    :param disable_TQDM: if True, turns off loading progress bar.
    :type disable_TQDM: bool
    :param save_labels_to_this_h5_file_instead: full path to H5 file to insert labels into instead of the H5 used as teh image source
    :type save_labels_to_this_h5_file_instead: string
    :return: labels_2_save - predictions ranging from 0 to 1 for not-touch and touch respectively
    :rtype: numpy array
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
    total_frame_count = []
    for H5_file in h5_file_list:
        H5 = h5py.File(H5_file, 'r')
        images = H5['images']
        total_frame_count.append(images.shape[0])

    return total_frame_count


def batch_size_file_ind_selector(num_in_each, batch_size):
    """
    batch_size_file_ind_selector - needed for ImageBatchGenerator to know which H5 file index
    to use depending on the iteration number used in __getitem__ in the generator.
    this all depends on the variable batch size.

    Example: the output of the following...
    batch_size_file_ind_selector([4000, 4001, 3999], [2000])
    would be [0, 0, 1, 1, 1, 2, 2] which means that there are 2 chunks in the first
    H5 file, 3 in the second and 2 in the third based on chunk size of 2000
    """
    break_into = np.ceil(np.array(num_in_each) / batch_size)
    extract_inds = np.array([])
    for k, elem in enumerate(break_into):
        tmp1 = np.array(np.ones(np.int(elem)) * k)
        extract_inds = np.concatenate((extract_inds, tmp1), axis=0)
    return extract_inds


# file_inds_for_H5_extraction is the same as extract_inds output from the above function
def reset_to_first_frame_for_each_file_ind(file_inds_for_H5_extraction):
    """
    reset_to_first_frame_for_each_file_ind - uses the output of batch_size_file_ind_selector
    to determine when to reset the index for each individual H5 file. using the above example
    the out put would be [0, 0, 2, 2, 2, 5, 5], each would be subtracted from the indexing to
    set the position of the index to 0 for each new H5 file.
    """
    subtract_for_index = []
    for k, elem in enumerate(file_inds_for_H5_extraction):
        tmp1 = np.diff(file_inds_for_H5_extraction)
        tmp1 = np.where(tmp1 != 0)
        tmp1 = np.append(-1, tmp1[0]) + 1
        subtract_for_index.append(tmp1[np.int(file_inds_for_H5_extraction[k])])
    return subtract_for_index


class ImageBatchGenerator(keras.utils.Sequence):

    def __init__(self, batch_size, h5_file_list):
        num_frames_in_all_H5_files = get_total_frame_count(h5_file_list)
        file_inds_for_H5_extraction = batch_size_file_ind_selector(
            num_frames_in_all_H5_files, batch_size)
        subtract_for_index = reset_to_first_frame_for_each_file_ind(
            file_inds_for_H5_extraction)
        # self.to_fit = to_fit #set to True to return XY and False to return X
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
        H5 = h5py.File(H5_file, 'r')
        #  list(H5.keys())

        images = H5['images']
        num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
        raw_X = images[b * num_2_extract_mod:b * (num_2_extract_mod + 1)]
        rgb_tensor = self.image_transform(raw_X)

        # if self.to_fit:
        #   labels_tmp = H5['labels']
        #   raw_Y = labels_tmp[b*num_2_extract_mod:b*(num_2_extract_mod+1)]
        #   return rgb_tensor, raw_Y
        # else:
        return rgb_tensor

    def getXandY(self, num_2_extract):
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
        labels_tmp = H5['labels']
        raw_Y = labels_tmp[b * num_2_extract_mod:b * (num_2_extract_mod + 1)]
        return rgb_tensor, raw_Y

    def image_transform(self, raw_X):
        '''
        input num_of_images x H x W, image input must be grayscale
        MobileNetV2 requires certain image dimensions
        We use N x 61 x 61 formated images
        self.IMG_SIZE is a single number to change the images into, images must be square
        '''
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
