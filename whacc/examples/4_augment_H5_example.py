from whacc import utils
from whacc import image_tools
from keras.preprocessing.image import ImageDataGenerator
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

from imgaug import augmenters as iaa  # optional program to further augment data
'*remove_2_edges.h5'
h5_subset_file = '/Users/phil/Dropbox/Autocurator/data/samsons_subsets/use/train_and_validate/validation_set.h5'
# datagen = ImageDataGenerator(rotation_range=360,  #
#                              width_shift_range=.05,  #
#                              height_shift_range=.05,  #
#                              shear_range=.05,  #
#                              zoom_range=.1,
#                              brightness_range=[0.2, 1.1])  #

datagen = ImageDataGenerator(rotation_range=360,  #
                             width_shift_range=.05,  #
                             height_shift_range=.05,  #
                             shear_range=.00,  #
                             zoom_range=.02,
                             brightness_range=[0.2, 1.1])  #

with h5py.File(h5_subset_file, 'r') as hf:
    test_img = hf['images'][0]  # grab a single image
num_aug = 4

aug_ims, _ = image_tools.augment_helper(datagen, num_aug, 0 ,test_img, -1)  # make 99 of the augmented images and output 1 of th original
# '-1' is the label associate ith the image, we dont care becasue we just want the iamge so it doenst matter what it is here

plt.figure(figsize=[5, 5])
plt.imshow(image_tools.img_unstacker(aug_ims, 10), )
plt.show()

gaussian_noise = iaa.AdditiveGaussianNoise(loc = 0, scale=3)
aug_ims_2 = gaussian_noise.augment_images(aug_ims)
plt.figure(figsize=[10, 10])
plt.imshow(image_tools.img_unstacker(aug_ims_2, 10))
plt.show()



# once we are happy with our augmentation process we can make an augmented H5 file using class
new_H5_file = h5_subset_file.split('.')[0] + '_AUG_min.h5' # create new file name based on teh original H5 name
h5creator = image_tools.h5_iterative_creator(new_H5_file, overwrite_if_file_exists=True,
                                             close_and_open_on_each_iteration=True)


utils.get_class_info(h5creator)
with h5py.File(h5_subset_file, 'r') as hf:
    for image, label in zip(tqdm(hf['images'][:]), hf['labels'][:]):
        aug_img_stack, labels_stack = image_tools.augment_helper(datagen, num_aug, 0, image, label)
        aug_img_stack = gaussian_noise.augment_images(aug_img_stack) # optional
        h5creator.add_to_h5(aug_img_stack, labels_stack)



#  test augment pattern on a blank pattern
# import numpy as np
# gaussian_noise = iaa.AdditiveGaussianNoise(loc = 0, scale=3)
# aug_ims_2 = gaussian_noise.augment_image(100*np.ones_like(aug_ims[0])).astype('int16')
# aug_ims_2[0, 0, :] = 255
# aug_ims_2[0, 1, :] = 0
# plt.figure(figsize=[5, 5])
# plt.imshow(aug_ims_2)
# plt.show()
# print(np.unique(aug_ims_2))


# #-=-=-=-=--=-=-=-=-=-=-
# from whacc import utils
# from whacc import image_tools
# from keras.preprocessing.image import ImageDataGenerator
# import h5py
# import matplotlib.pyplot as plt
# from tqdm import tqdm
#
# from imgaug import augmenters as iaa  # optional program to further augment data
#
# h5_subset_file = '/Users/phil/Dropbox/HIRES_LAB/curation_for_auto_curator/ALL_HUM_CONT_161.h5'
# num_aug = 4
# xxx = '_AUG_more_zoom_and_shift.h5'
#
# datagen = ImageDataGenerator(rotation_range=360,  #
#                              width_shift_range=.1,  #
#                              height_shift_range=.1,  #
#                              shear_range=.00,  #
#                              zoom_range=.1,
#                              brightness_range=[0.2, 1.1])  #
#
# with h5py.File(h5_subset_file, 'r') as hf:
#     test_img = hf['images'][85]  # grab a single image
#
# aug_ims, _ = image_tools.augment_helper(datagen, num_aug, 0 ,test_img, -1)  # make 99 of the augmented images and output 1 of th original
# # '-1' is the label associate ith the image, we dont care becasue we just want the iamge so it doenst matter what it is here
#
# plt.figure(figsize=[5, 5])
# plt.imshow(image_tools.img_unstacker(aug_ims, 10), )
# plt.show()
#
# gaussian_noise = iaa.AdditiveGaussianNoise(loc = 0, scale=3)
# aug_ims_2 = gaussian_noise.augment_images(aug_ims)
# plt.figure(figsize=[10, 10])
# plt.imshow(image_tools.img_unstacker(aug_ims_2, 10))
# plt.show()
#
#
#
# # once we are happy with out augmentation process we can make an augmented H5 file using class
# new_H5_file = h5_subset_file.split('.')[0] + xxx # create new file name based on teh original H5 name
# h5creator = image_tools.h5_iterative_creator(new_H5_file, overwrite_if_file_exists=True,
#                                              close_and_open_on_each_iteration=True,
#                                              color_channel=False)
#
#
# utils.get_class_info(h5creator)
# with h5py.File(h5_subset_file, 'r') as hf:
#     for image, label in zip(tqdm(hf['images'][:]), hf['labels'][:]):
#         aug_img_stack, labels_stack = image_tools.augment_helper(datagen, num_aug, 0, image, label)
#         aug_img_stack = gaussian_noise.augment_images(aug_img_stack) # optional
#         h5creator.add_to_h5(aug_img_stack[:, :, :, 0], labels_stack)




