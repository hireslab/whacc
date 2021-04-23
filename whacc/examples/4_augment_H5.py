from whacc import utils
from whacc import image_tools
from keras.preprocessing.image import ImageDataGenerator
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

from imgaug import augmenters as iaa  # optional program to further augment data

h5_subset_file = '/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/AH0667x170317_subset.h5'

datagen = ImageDataGenerator(rotation_range=360,  #
                             width_shift_range=.05,  #
                             height_shift_range=.05,  #
                             shear_range=.1,  #
                             zoom_range=.2,
                             brightness_range=[0.2, 1.1])  #

with h5py.File(h5_subset_file, 'r') as hf:
    test_img = hf['images'][0]  # grab a single image

aug_ims, _ = image_tools.augment_helper(datagen, 99, 1, test_img,
                                        -1)  # make 99 of the augmented images and output 1 of th original
# '-1' is the label associate ith the image, we dont care becasue we just want the iamge so it doenst matter what it is here

plt.figure(figsize=[5, 5])
plt.imshow(image_tools.img_unstacker(aug_ims, 10), )
plt.show()

gaussian_noise = iaa.AdditiveGaussianNoise(5, 10)  # range of 5 to 10
aug_ims_2 = gaussian_noise.augment_images(aug_ims)

plt.figure(figsize=[5, 5])
plt.imshow(image_tools.img_unstacker(aug_ims_2, 10))
plt.show()

# once we are happy with out augmentation process we can make an augmented H5 file using class
new_H5_file = h5_subset_file.split('.')[0] + '_AUG.h5' # create new file name based on teh original H5 name

h5creator = image_tools.h5_iterative_creator(new_H5_file, overwrite_if_file_exists=True,
                                             close_and_open_on_each_iteration=True)

utils.get_class_info(h5creator)
with h5py.File(h5_subset_file, 'r') as hf:
    for image, label in tqdm(zip(hf['images'][:], hf['labels'][:])):
        aug_img_stack, labels_stack = image_tools.augment_helper(datagen, 30, 0, image, label)
        aug_img_stack = gaussian_noise.augment_images(aug_img_stack) # optional
        h5creator.add_to_h5(aug_img_stack, labels_stack)



