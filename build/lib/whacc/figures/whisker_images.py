import matplotlib.pyplot as plt
from whacc import utils
import numpy as np
from whacc import image_tools
from keras.preprocessing.image import ImageDataGenerator
import h5py
from imgaug import augmenters as iaa  # optional program to further augment data

import os
from PIL import Image, ImageEnhance

rc = {"axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,
      "xtick.bottom" : False,
      "xtick.labelbottom" : False,
      "ytick.labelleft" : False,
      "ytick.left" : False}
plt.rcParams.update(rc)

a = 662
b = 11
base_dir = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/'
all_imgs = []
in_list = utils.lister_it(utils.get_h5s(base_dir, 0), '/val_', 'ALT_LABELS')
in_list = [in_list[k] for k in [2, 0, 1]]
for h5_file in in_list:
    with h5py.File(h5_file, 'r') as hf:
        img = image_tools.img_unstacker(hf['images'][a:a+b], num_frames_wide=b)
        all_imgs.append(img)
img = np.vstack(all_imgs)

img1 = Image.fromarray(img)
converter = ImageEnhance.Color(img1)
fig, ax = plt.subplots(nrows=6, ncols=1,sharex=True, sharey=False)
for k in range(6):
    img2 = converter.enhance(k+1)
    ax[k].imshow(img2)

save_name = '/Users/phil/Dropbox/HIRES_LAB/WhACC PAPER/figures/image_mod_types/image_example'
fig.set_size_inches(20*2, 15*2)
plt.savefig(save_name+'.pdf', dpi=600, transparent = True)
plt.close()




h5_file = in_list[1]
h5_file = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/3lag/val_3lag.h5'


datagen = ImageDataGenerator(rotation_range=360,  #
                            width_shift_range=.1,  #
                            height_shift_range=.1,  #
                            shear_range=.00,  #
                            zoom_range=.25,
                            brightness_range=[0.2, 1.2])  #
gaussian_noise = iaa.AdditiveGaussianNoise(loc = 0, scale=3)

num_aug = 10
img_ind = 5
for seed in [2]: # 'best' random seeds to show all modifications [1, 2, 7, 13] 2 seems the best
    np.random.seed(seed)
    all_imgs = []
    with h5py.File(h5_file, 'r') as hf:
        image = hf['images'][a+img_ind-1]
        aug_img_stack, labels_stack = image_tools.augment_helper(datagen, num_aug, 0, image, -1)
        aug_img_stack = gaussian_noise.augment_images(aug_img_stack)
        all_imgs.append(aug_img_stack)
    example_augmented_image = image_tools.img_unstacker(np.squeeze(np.asarray(all_imgs)), num_aug)

    img1 = Image.fromarray(example_augmented_image)
    converter = ImageEnhance.Color(img1)
    fig, ax = plt.subplots(nrows=6, ncols=1,sharex=True, sharey=False)
    for k in range(6):
        img2 = converter.enhance(k+1)
        ax[k].imshow(img2)


save_name = '/Users/phil/Dropbox/HIRES_LAB/WhACC PAPER/figures/image_mod_types/image_example_augmented_at_touch_frame_5_seed_'+str(seed)
fig.set_size_inches(20*2, 15*2)
plt.savefig(save_name+'.pdf', dpi=600, transparent = True)
plt.close()









h5_file = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/3lag/val_3lag.h5'
all_imgs = []
with h5py.File(h5_file, 'r') as hf:
        print(hf['labels'].shape)
        img = image_tools.img_unstacker(hf['images'][a:a+b], num_frames_wide=b)
        all_imgs.append(img)
img = np.vstack(all_imgs)

plt.imshow(img)


