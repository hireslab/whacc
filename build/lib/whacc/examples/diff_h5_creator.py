from whacc import utils
from whacc import image_tools
import numpy as np
import h5py
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
import copy

#  make a lagged H5 file where frame 1 is stacked 3 deep 1 is (rand, rand, 1) 2 is (rand, 1, 2) 3 is (1, 2, 3)

f = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_80border_single_frame.h5"
f2 = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_80border_single_frame_lag_3_deep_real_at_end.h5"

h5c = image_tools.h5_iterative_creator(f2, overwrite_if_file_exists=True)

with h5py.File(f, 'r+') as h:
    x = h['frame_nums'][:]
    for ii, (k1, k2) in enumerate(tqdm(utils.loop_segments(x), total=len(x))):
        new_imgs = image_tools.stack_imgs_lag(h['images'][k1:k2])
        h5c.add_to_h5(new_imgs, np.ones(new_imgs.shape[0]) * -1)

# copy over the other info from the OG h5 file
utils.copy_h5_key_to_another_h5(f, f2, 'labels')
utils.copy_h5_key_to_another_h5(f, f2, 'frame_nums')

# make a copy of the file and rename it
f3 = f2[:-3] + '_DIFF.h5'
shutil.copy(f2, f3)
# change color channel 0 and 1 to diff images from color channel 3 so color channels 0, 1, and 2 are 0-2, 1-2, and 2
with h5py.File(f3, 'r+') as h:
    for i in tqdm(range(h['images'].shape[0])):
        k = copy.deepcopy(h['images'][i])
        for img_i in range(2):
            k = k.astype(float)
            a = k[:, :, img_i] - k[:, :, -1]
            a = ((a + 255) / 2).astype(np.uint8)
            h['images'][i, :, :, img_i] = a

# plot and example of the new H5 files so we know what we are dealing with
dis_file = f3
with h5py.File(dis_file, 'r') as h:
    ind1 = 80
    print(h['images'][ind1].shape)
    plt.imshow(h['images'][ind1])
    plt.figure()
    plt.plot(h['labels'][:4000])

    plt.figure()
    plt.imshow(h['images'][ind1][:, :, 0])
    plt.figure()
    plt.imshow(h['images'][ind1][:, :, 1])
    plt.figure()
    plt.imshow(h['images'][ind1][:, :, 2])

    utils.print_list_with_inds(h.keys())

"""

"""
from whacc import analysis
a = analysis.pole_plot()

def four_class_labels_from_binary(x):
    a = np.asarray(x)
    b = np.asarray([0]+list(np.diff(a)))
    c = a+b
    c[c==-1] = 3
    return c
x = [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0]
four_class_labels_from_binary(x)


h5tmp = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_80border_single_frame_lag_3_deep_real_at_end.h5"
a = analysis.pole_plot(h5tmp, true_val=image_tools.get_h5_key_and_concatenate([tmph5], 'labels'))
a.current_frame = 75
a.plot_it()
plt.ylim([-.5, 3.5])
plt.show()
# 000002111130000



