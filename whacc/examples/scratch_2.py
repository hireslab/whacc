# # # # # import h5py
# # # # # import matplotlib.pyplot as plt
# # # # # from whacc import image_tools
# # # # # import os
# # # # # from whacc import utils
# # # # #
# # # # # for h5_file in utils.get_h5s('/Users/phil/Dropbox/Autocurator/data/samsons_subsets/OLD_2'):
# # # # #     with h5py.File(h5_file, 'r') as h:
# # # # #         plt.figure(figsize=[20, 10])
# # # # #         plt.imshow(image_tools.img_unstacker(h['images'][:],40))
# # # # #         plt.title(h5_file.split('/')[-1])
# # # # #         plt.savefig(h5_file[:-3] + "_fig_plot.png")
# # # # #         plt.close()
# # # # #
# # # # #
# # # # # for k in utils.get_h5s('/Users/phil/Dropbox/Autocurator/data/samsons_subsets/OLD_2'):
# # # # #     with h5py.File(k, 'r') as h:
# # # # #         k2 = k[:-3]+'_TMP.h5'
# # # # #         with h5py.File(k2, 'w') as h2:
# # # # #             h2.create_dataset('all_inds', data=h['all_inds'][50:])
# # # # #             h2.create_dataset('images', data=h['images'][50:])
# # # # #             h2.create_dataset('in_range', data=h['in_range'][50:])
# # # # #             h2.create_dataset('labels', data=h['labels'][50:])
# # # # #             h2.create_dataset('retrain_H5_info', data=h['retrain_H5_info'][:])
# # # # #     os.remove(k)
# # # # #     os.rename(k2, k)
# # # # #
# # # # #
# # # # # import h5py
# # # # # import matplotlib.pyplot as plt
# # # # # from whacc import image_tools
# # # # #
# # # # # h5_file= '/Users/phil/Downloads/AH1159X18012021xS404_subset (1).h5'
# # # # # with h5py.File(h5_file, 'r') as h:
# # # # #     plt.figure(figsize=[20, 10])
# # # # #     plt.imshow(image_tools.img_unstacker(h['images'][:],10))
# # # # #     plt.title(h5_file.split('/')[-1])
# # # # #     print(len(h['labels'][:]))
# # # # #
# # # # #
# # # # #
# # # # #
# # # #
# # # # import numpy as np
# # # # import whacc
# # # # from whacc import analysis
# # # # import matplotlib.pyplot as plt
# # # #
# # # #
# # # # a = analysis.pole_plot(
# # # #     '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/master_train_test1.h5',
# # # #     pred_val = [0,0,0,0,0,0,0,.2,.4,.5,.6,.7,.8,.8,.6,.4,.2,.1,0,0,0,0],
# # # #     true_val = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
# # # #     len_plot = 10)
# # # #
# # # # a.plot_it()
# # # #
# # # #
# # # f1 = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/delete_after_oct_2021/AH0000x000000.h5"
# # # import h5py
# # #
# # # with h5py.File(f1, 'r') as h:
# # #     k = 'file_name_nums'
# # #     print(k, '   ', h[k].shape)
# # #     k = 'full_file_names'
# # #     print(k, '   ', (h[k].shape))
# # #     k = 'images'
# # #     print(k, '   ', h[k].shape)
# # #     k = 'in_range'
# # #     print(k, '   ', h[k].shape)
# # #     k = 'labels'
# # #     print(k, '   ', h[k].shape)
# # #     k = 'locations_x_y'
# # #     print(k, '   ', h[k].shape)
# # #     k = 'max_val_stack'
# # #     print(k, '   ', h[k].shape)
# # #     k = 'multiplier'
# # #     print(k, '   ', h[k].shape)
# # #     k = 'trial_nums_and_frame_nums'
# # #     print(k, '   ', h[k].shape)
# # #
# # # with h5py.File(save_directory + file_name, 'r+') as hf:  # with -> auto close in case of failure
# # #     hf.create_dataset("asdfasdf", data=loc_stack_all2)
# # #     for i, k in enumerate(loc_stack_all):
# # #         print(i)
# # #         hf.create_dataset("aaa"+str(i), data=k)
# # #
# # # loc_stack_all2 = loc_stack_all.copy()
# # #
# # #
# # #
# # #
# # # for i, k in enumerate(loc_stack_all):
# # #     if not k.shape==(4000, 2):
# # #         print(i)
# # #     # for kk in k.flatten():
# # #     #     print(kk.shape)
# # #     #         # if not kk.dtype=='int64':
# # #     #         #     print('booooo')
# # #     #
# # #     #
# #
# #
# # # import h5py
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # cnt = 0
# # # f = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000.h5"
# # # with h5py.File(f, 'r') as h:
# # #     print(len(h['labels']))
# # #     for k in range(0, 1163998, 20000):
# # #         plt.figure()
# # #         plt.imshow(h['images'][k])
# # #         cnt+=1
# # #         # if cnt>=20 :
# # #         #     asdfasdf
# #
# # from whacc import utils
# # from whacc import image_tools
# # import copy
# # import h5py
# # f = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000.h5'
# # f2 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_80border_single_frame.h5'
# #
# # h5c = image_tools.h5_iterative_creator(f2,
# #                  overwrite_if_file_exists=False,
# #                  max_img_height=61,
# #                  max_img_width=61,
# #                  close_and_open_on_each_iteration=True,
# #                  color_channel=False)
# # with h5py.File(f, 'r') as h:
# #     print(len(h['labels'][:]))
# #     h_cont = copy.deepcopy(h['labels'][:])
# # utils.create_master_dataset(h5c, [f], [h_cont], borders=80, max_pack_val=100)
# #
# #
# # ####
# #
# #
# # f = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_80border_single_frame.h5"
# # with h5py.File(f, 'r') as h:
# #     utils.print_list_with_inds(h.keys())
# #     tmp1 = copy.deepcopy(h['frame_nums'][:])
# #
# #
# # """
# # add teh frame and trail number to add teh frame num info to each with the same name!!!!
# # """
# #
# # h5file = "/Users/phil/Downloads/untitled folder 2/AH0000x000000_small_tester.h5"
# # def make_sure_frame_nums_exist(h5file):
# #     with h5py.File(h5file, 'r+') as h:
# #         key_list = list(h.keys())
# #         if 'frame_nums' in key_list:
# #             print("""'frame_nums' already in the key list""")
# #             return None
# #         assert 'trial_nums_and_frame_nums' in key_list, """key 'trial_nums_and_frame_nums' must be in the provided h5 this is the only reason program exists"""
# #         frame_nums = h['trial_nums_and_frame_nums'][1, :]
# #         h.create_dataset('frame_nums', shape=np.shape(frame_nums), data=frame_nums)
# # make_sure_frame_nums_exist(h5file)
# #
# # frame_num_array_list = get_h5_key_and_dont_concatenate(h5_to_split_list, key_name='trial_nums_and_frame_nums')
# # frame_num_array_list = list(frame_num_array_list[0][-1])
# # frame_num_array_list = [frame_num_array_list]
# # bd = '/Users/phil/Downloads/untitled folder 2/'
# # split_h5_loop_segments(h5_to_split_list, [1, 3], [bd+'TRASH', bd+'TRASH2'], frame_num_array_list, chunk_size=10000, add_numbers_to_name=False,
# #              disable_TQDM=False, set_seed = None)
# #
# #
# # h5file = '/Users/phil/Downloads/untitled folder 2/TRASH2.h5'
# # utils.print_h5_keys(h5file)
# # with h5py.File(h5file, 'r') as h:
# #     print(len(h['labels'][:]))
# #     print(h['frame_nums'][:])
# #
#
#
from whacc import image_tools, utils
h5_to_split_list = "/Users/phil/Downloads/untitled folder 2/AH0000x000000_small_tester.h5"
h5_to_split_list = [h5_to_split_list]
utils.print_h5_keys(h5_to_split_list[0])
bd = '/Users/phil/Downloads/untitled folder 2/'
image_tools.split_h5_loop_segments(h5_to_split_list, [1, 3], [bd+'TRASH', bd+'TRASH2'], chunk_size=10000, add_numbers_to_name=False,
             disable_TQDM=False, set_seed = None)


from whacc import utils
from whacc import image_tools
from keras.preprocessing.image import ImageDataGenerator
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from imgaug import augmenters as iaa  # optional program to further augment data
h5_subset_file = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/3lag/AH0000x000000_5border_single_frame_onset_offset_for_aug_3lag_remove_2_edges.h5'
for h5_subset_file in utils.get_files('/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/', '*remove_2_edges.h5'):
    datagen = ImageDataGenerator(rotation_range=360,  #
                                 width_shift_range=.1,  #
                                 height_shift_range=.1,  #
                                 shear_range=.00,  #
                                 zoom_range=.25,
                                 brightness_range=[0.2, 1.2])  #

    with h5py.File(h5_subset_file, 'r') as hf:
        test_img = hf['images'][2]  # grab a single image
    num_aug = 20

    aug_ims, _ = image_tools.augment_helper(datagen, num_aug, 0 ,test_img, -1)  # make 99 of the augmented images and output 1 of th original
    # '-1' is the label associate ith the image, we dont care becasue we just want the iamge so it doenst matter what it is here

    # plt.figure(figsize=[5, 5])
    # plt.imshow(image_tools.img_unstacker(aug_ims, 10), )
    # plt.show()

    gaussian_noise = iaa.AdditiveGaussianNoise(loc = 0, scale=3)
    aug_ims_2 = gaussian_noise.augment_images(aug_ims)
    # plt.figure(figsize=[10, 10])
    # plt.imshow(image_tools.img_unstacker(aug_ims_2, 10))
    # plt.show()

    num_aug = 1
    for ii in range(10):
        # once we are happy with our augmentation process we can make an augmented H5 file using class
        new_H5_file = h5_subset_file.split('.')[0] + '_AUG_' + str(ii) + '.h5' # create new file name based on teh original H5 name
        h5creator = image_tools.h5_iterative_creator(new_H5_file, overwrite_if_file_exists=True,
                                                     close_and_open_on_each_iteration=True, color_channel=True)
        utils.get_class_info(h5creator)
        with h5py.File(h5_subset_file, 'r') as hf:
            for image, label in zip(tqdm(hf['images'][:]), hf['labels'][:]):
                aug_img_stack, labels_stack = image_tools.augment_helper(datagen, num_aug, 0, image, label)
                aug_img_stack = gaussian_noise.augment_images(aug_img_stack) # optional
                h5creator.add_to_h5(aug_img_stack[:, :, :, :], labels_stack)




h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_0.h5'
a = analysis.pole_plot(h5)
a.next()
h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_1.h5'
b = analysis.pole_plot(h5)
b.next()



from whacc import utils
all_h5s = utils.get_h5s('/Users/phil/Dropbox/HIRES_LAB/curation_for_auto_curator/finished_contacts/')
all_h5s_imgs = utils.get_h5s('/Users/phil/Dropbox/HIRES_LAB/curation_for_auto_curator/H5_data/')
h_cont = utils._get_human_contacts_(all_h5s)
h5c = image_tools.h5_iterative_creator('/Users/phil/Dropbox/HIRES_LAB/curation_for_auto_curator/holy_set_80_border_single_frame.h5',
                                       overwrite_if_file_exists = True,
                                       color_channel = False)
utils.create_master_dataset(h5c, all_h5s_imgs, h_cont, borders = 80, max_pack_val = 100)












#
# from whacc import image_tools
#
# # bd = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/'
# # h5_to_split_list = ['/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_80border_single_frame.h5']
# # image_tools.split_h5_loop_segments(h5_to_split_list, [7, 3], [bd+'train', bd+'val'], chunk_size=1000, add_numbers_to_name=False,
# #              disable_TQDM=False, set_seed = 0, color_channel=False)
# #
#
#
# bd = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border_aug_0_to_9_test/'
# h5_to_split_list = ['/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoC2uratorDiverseDataset/AH0000x000000/AH0000x000000_80border_single_frame.h5',
#                     '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_9.h5']
#                     # '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_8.h5',
#                     # '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_7.h5',
#                     # '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_6.h5',
#                     # '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_5.h5',
#                     # '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_4.h5',
#                     # '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_3.h5',
#                     # '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_2.h5',
#                     # '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_1.h5',
#                     # '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_3border_single_frame_onset_offset_for_aug_AUG_0.h5',]
# image_tools.split_h5_loop_segments(h5_to_split_list, [1, 1, 1, 99], [bd+'train1',bd+'train2',bd+'train3', bd+'val4'], chunk_size=1000, add_numbers_to_name=False,
#              disable_TQDM=False, set_seed = 0, color_channel=False)
#
#
# import h5py
# with h5py.File(bd+'train1.h5', 'r') as h:
#     print(h['images'][:].shape)
#     print(h['labels'][:].shape)
#     print(sum(h['frame_nums'][:]))
#
#
#
#


from whacc import utils
x = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/ALT_LABELS/train_regular_ALT_LABELS.h5'
keys = utils.print_h5_keys(x, return_list=True)
tmp1 = []
for k in keys:
    tmp1.append(image_tools.get_h5_key_and_concatenate([x], k))

for k in tmp1:
    print(list(set(k)))


x = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/regular/val_regular.h5'
x = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/3lag_diff/AH0000x000000_5border_single_frame_onset_offset_for_aug_3lag_diff_remove_2_edges_AUG_5.h5'
x = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/for_aug/3lag_diff/train.h5'
x = l1[0]
x = l2[0]
x = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/regular/val_regular.h5'
a = image_tools.get_h5_key_and_concatenate([x])
print(list(set(a)))






def label_naming_shorthand_dict(name_key=None):
    label_naming_shorthand_dict = {
        '[0, 1, 2, 3, 4, 5]- (no touch, touch, onset, one after onset, offset, one after offset)': 'on-off_set_and_one_after',
        '[0, 1, 2, 3]- (no touch, touch, onset, offset': 'on-off_set',
        '[0, 1, 2]- (no event, onset, offset)': 'only_on-off_set',
        '[0, 1]- (no touch, touch)': 'regular',
        '[0, 1]- (not offset, offset)': 'only_offset',
        '[0, 1]- (not onset, onset)': 'only_onset',
        '[0, 1, 2, 3]- (no touch, touch, one after onset, offset)': 'overlap_whisker_on-off'}
    if name_key is None:
        return label_naming_shorthand_dict
    else:
        return label_naming_shorthand_dict[name_key]

def copy_over_new_labels(label_key_name, image_h5_list, label_h5_list):
    label_key_shorthand = label_naming_shorthand_dict(label_key_name)
    for img_src, lab_src in zip(image_h5_list, label_h5_list):
        utils.copy_h5_key_to_another_h5(lab_src, img_src, label_key_name, 'labels')

from whacc import image_tools
from whacc import utils
from whacc import analysis
import matplotlib.pyplot as plt

x = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/small_h5s/data/3lag/small_train_3lag.h5'

alt_labels = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/small_h5s/data/ALT_LABELS/small_train_ALT_LABELS.h5'

copy_over_new_labels('[0, 1, 2, 3]- (no touch, touch, one after onset, offset)', [x], [alt_labels])

a = analysis.pole_plot(x)
a.len_plot = 11
a.current_frame = 36
a.plot_it()
plt.ylim([-.5, 6])
