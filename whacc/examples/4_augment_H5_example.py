from whacc import utils
from whacc import image_tools
from keras.preprocessing.image import ImageDataGenerator
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

from imgaug import augmenters as iaa  # optional program to further augment data

h5_subset_file = '/Users/phil/Dropbox/Autocurator/data/samsons_subsets/use/train_and_validate/training_set.h5'

datagen = ImageDataGenerator(rotation_range=360,  #
                             width_shift_range=.05,  #
                             height_shift_range=.05,  #
                             shear_range=.05,  #
                             zoom_range=.1,
                             brightness_range=[0.2, 1.1])  #

with h5py.File(h5_subset_file, 'r') as hf:
    test_img = hf['images'][0]  # grab a single image

aug_ims, _ = image_tools.augment_helper(datagen, 50, 0 ,test_img, -1)  # make 99 of the augmented images and output 1 of th original
# '-1' is the label associate ith the image, we dont care becasue we just want the iamge so it doenst matter what it is here

plt.figure(figsize=[5, 5])
plt.imshow(image_tools.img_unstacker(aug_ims, 10), )
plt.show()

gaussian_noise = iaa.AdditiveGaussianNoise(loc = 0, scale=3)
aug_ims_2 = gaussian_noise.augment_images(aug_ims)
plt.figure(figsize=[10, 10])
plt.imshow(image_tools.img_unstacker(aug_ims_2, 10))
plt.show()





# once we are happy with out augmentation process we can make an augmented H5 file using class
new_H5_file = h5_subset_file.split('.')[0] + '_AUG.h5' # create new file name based on teh original H5 name
h5creator = image_tools.h5_iterative_creator(new_H5_file, overwrite_if_file_exists=True,
                                             close_and_open_on_each_iteration=True)

utils.get_class_info(h5creator)
with h5py.File(h5_subset_file, 'r') as hf:
    for image, label in zip(tqdm(hf['images'][:]), hf['labels'][:]):
        aug_img_stack, labels_stack = image_tools.augment_helper(datagen, 30, 0, image, label)
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
