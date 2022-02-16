from keras.preprocessing.image import ImageDataGenerator
from imgaug import augmenters as iaa  # optional program to further augment data
from tqdm import tqdm
from whacc.model_maker import *
from whacc import image_tools
from whacc import utils
import os
from tqdm.contrib import tzip
import h5py


def make_all_h5s_from_subset(base_dir_all, subset_h5_file, num_to_augment=1):
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
                if i == 0:
                    out = np.asarray(h[key_name][:])
                else:
                    out = np.concatenate((out, h[key_name][:]))
        return out

    def repeat_labels(h5_file, label_key):  # with a shape test
        with h5py.File(h5_file, 'r+') as h:
            num_repeats = h['images'][:].shape[0] / h[label_key][:].shape[0]
            if num_repeats > 1 and round(num_repeats) == num_repeats:
                labels = h[label_key][:]
                rep_labels = np.repeat(labels, num_repeats, axis=0)
                del h[label_key]
                h[label_key] = rep_labels

    def foo_get_frame_nums_from_all_inds(h5_subset_file):
        tmp1 = image_tools.get_h5_key_and_concatenate([h5_subset_file], 'all_inds')
        tmp1, tmp2 = utils.group_consecutives(tmp1)
        frame_nums = []
        for k in tmp2:
            frame_nums.append(len(k))
        return frame_nums

    # **********$%$%$%$%$%%$
    # these indacate areas where user may want to customize like make a test set for example
    # **********$%$%$%$%$%%$

    tmp_h5s = base_dir_all + '/tmp_h5s/'
    subset_h5s = [subset_h5_file]
    for h5_subset_file in tqdm(subset_h5s):
        a = h5_subset_file.split('/')[-1].split('RETRAIN_')[-1].split('.h5')[0]
        base_h5 = base_dir_all + '/DATA/data_' + a
        single_frame_h5s = base_h5 + '/single_frame/'
        Path(tmp_h5s).mkdir(parents=True, exist_ok=True)
        Path(single_frame_h5s).mkdir(parents=True, exist_ok=True)
        # vvvvv add the frame nums to the H5 file
        frame_nums = foo_get_frame_nums_from_all_inds(h5_subset_file)

        utils.add_to_h5(h5_subset_file, 'frame_nums', frame_nums, overwrite_if_exists=True)
        # ^^^^^ add the frame nums to the H5 file
        # vvvvvv single frame and then split
        utils.reduce_to_single_frame_from_color(h5_subset_file, tmp_h5s + 'temp.h5')

        # split_h5s = image_tools.split_h5([tmp_h5s + 'temp.h5'],
        #                                    split_percentages=[.8, .2],  # **********$%$%$%$%$%%$
        #                                    temp_base_name=[single_frame_h5s + 'train',
        #                                                    single_frame_h5s + 'val'],
        #                                    chunk_size=10000,
        #                                    add_numbers_to_name=False,
        #                                    disable_TQDM=True,
        #                                    set_seed=0,
        #                                    color_channel=False)

        split_h5s = image_tools.split_h5_loop_segments([tmp_h5s + 'temp.h5'],
                                                       split_percentages=[.8, .2],  # **********$%$%$%$%$%%$
                                                       temp_base_name=[single_frame_h5s + 'train',
                                                                       single_frame_h5s + 'val'],
                                                       # **********$%$%$%$%$%%$
                                                       chunk_size=10000,
                                                       add_numbers_to_name=False,
                                                       disable_TQDM=True,
                                                       set_seed=0,
                                                       color_channel=False,
                                                       force_random_each_frame=False)
        # ^^^^^^ single frame and then split
        # vvvvvvvv convert to all types
        utils.make_all_H5_types(
            single_frame_h5s)  # auto generate all the h5 types using a single set of flat (no color) image H5
        utils.make_alt_labels_h5s(single_frame_h5s)  # auto generate the different types of labels
        # ^^^^^^^^ convert to all types
        # vvvvv AUGMENT #**********$%$%$%$%$%%$
        datagen = ImageDataGenerator(rotation_range=360,  #
                                     width_shift_range=.3,  #
                                     height_shift_range=.3,  #
                                     shear_range=.00,  #
                                     zoom_range=.25,
                                     brightness_range=[0.4, 1.4])  #
        gaussian_noise = iaa.AdditiveGaussianNoise(loc=0, scale=3)
        num_aug = 1  # DO NOT CHANGE THIS CHANGE num_to_augment, this preserves the order of the images to allow image subtraction
        xxx = utils.get_h5s(base_h5)
        xxx = utils.lister_it(xxx, remove_string=['ALT_LABELS', 'single_frame'])

        for each_h5 in xxx:
            if 'images' in utils.print_h5_keys(each_h5, True, False):
                combine_list = []
                combine_list.append(each_h5)

                shutil.rmtree(tmp_h5s, ignore_errors=True)
                Path(tmp_h5s).mkdir(parents=True, exist_ok=True)
                for ii in range(num_to_augment):
                    new_H5_file = each_h5.split('.')[0] + '_AUG_' + str(
                        ii) + '.h5'  # create new file name based on teh original H5 name
                    new_H5_file = tmp_h5s + os.path.basename(new_H5_file)
                    combine_list.append(
                        new_H5_file)  # all h5s to combine into one after all augmented files have been created
                    h5creator = image_tools.h5_iterative_creator(new_H5_file, overwrite_if_file_exists=True,
                                                                 close_and_open_on_each_iteration=True,
                                                                 color_channel=True)

                    with h5py.File(each_h5, 'r') as hf:
                        for image, label in tzip(hf['images'][:], hf['labels'][:]):
                            aug_img_stack, labels_stack = image_tools.augment_helper(datagen, num_aug, 0, image, label)
                            aug_img_stack = gaussian_noise.augment_images(aug_img_stack)  # optional
                            h5creator.add_to_h5(aug_img_stack[:, :, :, :], labels_stack)
                    utils.copy_h5_key_to_another_h5(each_h5, new_H5_file, 'frame_nums',
                                                    'frame_nums')  # copy the frame nums to the aug files
                # combine all the
                final_name = each_h5.split('.h5')[0] + '_AUG.h5'
                image_tools.split_h5_loop_segments(combine_list,  # combine all the
                                                   [1],
                                                   final_name,
                                                   add_numbers_to_name=False,
                                                   set_seed=0,
                                                   color_channel=True)
                with h5py.File(final_name, 'r+') as h:
                    h['multiplier'][:] = num_to_augment + 1
    # shutil.rmtree(tmp_h5s)
    shutil.rmtree(tmp_h5s, ignore_errors=True)
    """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
    """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
    """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
    """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
    """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

    # def test_equal(h5_file, label_key):  # with a shape test
    #     with h5py.File(h5_file, 'r') as h:
    #         num_repeats = h['images'][:].shape[0] / h[label_key][:].shape[0]
    #     return num_repeats == 1

    x = base_dir_all + '/DATA/'
    for k in next(os.walk(x))[1]:
        a = utils.get_h5s(x + os.sep + k)
        a = utils.lister_it(a, keep_strings='train')
        src_h5 = utils.lister_it(a, '/ALT_LABELS/', remove_string=['single_frame'])[0]
        dest_h5s = utils.lister_it(a, remove_string=['single_frame', '/ALT_LABELS/'])
        for src_label_key in utils.print_h5_keys(src_h5, 1):
            for dest_H5 in dest_h5s:
                if src_label_key in utils.print_h5_keys(dest_H5, 1, 0):
                    with h5py.File(dest_H5, 'r+') as h:
                        del h[src_label_key]
                utils.copy_h5_key_to_another_h5(src_h5, dest_H5, src_label_key, src_label_key)
                repeat_labels(dest_H5, src_label_key)

    for k in next(os.walk(x))[1]:
        a = utils.get_h5s(x + os.sep + k)
        a = utils.lister_it(a, keep_strings='val')
        src_h5 = utils.lister_it(a, '/ALT_LABELS/', remove_string=['single_frame'])[0]
        dest_h5s = utils.lister_it(a, remove_string=['single_frame', '/ALT_LABELS/'])
        for src_label_key in utils.print_h5_keys(src_h5, 1):
            for dest_H5 in dest_h5s:
                if src_label_key in utils.print_h5_keys(dest_H5, 1, 0):
                    with h5py.File(dest_H5, 'r+') as h:
                        del h[src_label_key]
                utils.copy_h5_key_to_another_h5(src_h5, dest_H5, src_label_key, src_label_key)
                repeat_labels(dest_H5, src_label_key)


base_dir_all = '/Volumes/GoogleDrive/My Drive/Colab data/test_AH1030_subset/'  # where to put the files (will create this folder automatically)
base_dir_all = '/Volumes/GoogleDrive/My Drive/Colab data/test_AH1030_half_data/'  # where to put the files (will create this folder automatically)
subset_h5_file = '/Volumes/GoogleDrive-114825029448473821206/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/data_ANM234232_140120_AH1030_AAAA_a/regular/ANM234232_140120_AH1030_AAAA_a_regular.h5'
num_to_augment = 1
# subset_h5_file = '/Volumes/GoogleDrive/My Drive/Colab data/test_AH1030_subset/ANM234232_140120_AH1030_AAAA_a_subset.h5'
make_all_h5s_from_subset(base_dir_all, subset_h5_file, num_to_augment=num_to_augment)

tmp1 = '/Volumes/GoogleDrive-114825029448473821206/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/data_ANM234232_140120_AH1030_AAAA_a/3lag/ANM234232_140120_AH1030_AAAA_a_3lag.h5'
bd = '/Volumes/GoogleDrive-114825029448473821206/My Drive/Colab data/test_AH1030_half/'
split_h5s = image_tools.split_h5_loop_segments([tmp1],
                                               split_percentages=[.5, .5],  # **********$%$%$%$%$%%$
                                               temp_base_name=[bd + 'train',
                                                               bd + 'val'],
                                               # **********$%$%$%$%$%%$
                                               chunk_size=10000,
                                               add_numbers_to_name=False,
                                               disable_TQDM=False,
                                               set_seed=0,
                                               color_channel=True,
                                               force_random_each_frame=False)

# h5_file = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED/ANM234232_140120_AH1030_AAAA_a_ALT_LABELS.h5'
# real_bool = image_tools.get_h5_key_and_concatenate(h5_file, '[0, 1]- (no touch, touch)')

