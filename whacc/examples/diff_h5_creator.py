from whacc import utils
from whacc import image_tools
import numpy as np
import h5py
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
import copy
import os


#  make a lagged H5 file where frame 1 is stacked 3 deep 1 is (rand, rand, 1) 2 is (rand, 1, 2) 3 is (1, 2, 3)
f = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/small_h5s/single_frame/small_test.h5'
f2 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/small_h5s/single_frame/small_testasdfasdfasdf.h5'
def expand_single_frame_to_3_color_h5(f, f2):
    h5c = image_tools.h5_iterative_creator(f2, overwrite_if_file_exists=True)
    with h5py.File(f, 'r+') as h:
        x = h['frame_nums'][:]
        for ii, (k1, k2) in enumerate(tqdm(utils.loop_segments(x), total=len(x))):
            new_imgs = h['images'][k1:k2]
            new_imgs = np.tile(new_imgs[:, :, :, None], 3)
            h5c.add_to_h5(new_imgs, np.ones(new_imgs.shape[0]) * -1)

    # copy over the other info from the OG h5 file
    utils.copy_h5_key_to_another_h5(f, f2, 'labels', 'labels')
    utils.copy_h5_key_to_another_h5(f, f2, 'frame_nums', 'frame_nums')

new_imgs = expand_single_frame_to_3_color_h5(f, f2)

def stack_lag_h5_maker(f, f2, buffer=2, shift_to_the_right_by=0):
    """

    Parameters
    ----------
    f : h5 file with SINGLE FRAMES this is ment to be a test program. if used long term I will change this part
    f2 :
    buffer :
    shift_to_the_right_by :

    Returns
    -------

    """
    h5c = image_tools.h5_iterative_creator(f2, overwrite_if_file_exists=True)
    with h5py.File(f, 'r+') as h:
        x = h['frame_nums'][:]
        for ii, (k1, k2) in enumerate(tqdm(utils.loop_segments(x), total=len(x))):
            new_imgs = image_tools.stack_imgs_lag(h['images'][k1:k2], buffer=buffer, shift_to_the_right_by=shift_to_the_right_by)
            h5c.add_to_h5(new_imgs, np.ones(new_imgs.shape[0]) * -1)

    # copy over the other info from the OG h5 file
    utils.copy_h5_key_to_another_h5(f, f2, 'labels')
    utils.copy_h5_key_to_another_h5(f, f2, 'frame_nums')


def diff_lag_h5_maker(f3):
    """
    need to use the stack_lag_h5_maker first and then send a copy of that into this one again these program are only a temp
    solution, if we use these methods for the main model then I will make using them more fluid and not depend on one another
    Parameters
    ----------
    f3 : the file from stack_lag_h5_maker output

    Returns
    -------

    """
    # change color channel 0 and 1 to diff images from color channel 3 so color channels 0, 1, and 2 are 0-2, 1-2, and 2
    with h5py.File(f3, 'r+') as h:
        for i in tqdm(range(h['images'].shape[0])):
            k = copy.deepcopy(h['images'][i])
            for img_i in range(2):
                k = k.astype(float)
                a = k[:, :, img_i] - k[:, :, -1]
                a = ((a + 255) / 2).astype(np.uint8)
                h['images'][i, :, :, img_i] = a


def make_all_H5_types(base_dir_all_h5s):
    for f in utils.get_h5s(base_dir_all_h5s):
        # f = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/small_h5s/single_frame/small_test.h5"
        basename = os.path.basename(f)[:-3] + '_3lag.h5'
        basedir = os.sep.join(f.split(os.sep)[:-2]) + os.sep + '3lag' + os.sep
        if not os.path.isdir(basedir):
            os.mkdir(basedir)
        f2 = basedir + basename

        stack_lag_h5_maker(f, f2, buffer=2, shift_to_the_right_by=0)

        basename = os.path.basename(f2)[:-3] + '_diff.h5'
        basedir = os.sep.join(f.split(os.sep)[:-2]) + os.sep + 'diff' + os.sep
        if not os.path.isdir(basedir):
            os.mkdir(basedir)
        f3 = basedir + basename
        shutil.copy(f2, f3)
        diff_lag_h5_maker(f3)


base_dir_all_h5s = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/small_h5s/single_frame/'
make_all_H5_types(base_dir_all_h5s)

# plot and example of the new H5 files so we know what we are dealing with
dis_file = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/small_h5s/3lag/small_val_3lag.h5'
with h5py.File(dis_file, 'r') as h:
    ind1 = 19
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

x = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
four_class_labels_from_binary(x)

h5tmp = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_80border_single_frame_lag_3_deep_real_at_end.h5"
a = analysis.pole_plot(h5tmp, true_val=image_tools.get_h5_key_and_concatenate([tmph5], 'labels'))
a.current_frame = 75
a.plot_it()
plt.ylim([-.5, 3.5])
plt.show()
# 000002111130000


f = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_80border_single_frame.h5"

x1 = 2281
f4 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/small_test.h5'
h5c = image_tools.h5_iterative_creator(f4, overwrite_if_file_exists=True, color_channel=False)
with h5py.File(f, 'r') as h:
    h5c.add_to_h5(h['images'][x1:x1 + 250], h['labels'][x1:x1 + 250])

    # print(h['frame_nums'][:10])
    # plt.figure()
    # plt.plot(h['labels'][x1:x1+250])
    # plt.figure()
    # x1 = 824
    # plt.plot(h['labels'][x1:x1+250])
    # plt.figure()
    # x1 = 2281
    # plt.plot(h['labels'][x1:x1+250])


# for f in utils.get_h5s('/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/small_h5s/single_frame/'):
#     with h5py.File(f, 'r+') as h:
#         frame_nums = [len(h['labels'][:])]
#         h.create_dataset('frame_nums', shape=np.shape(frame_nums), data=frame_nums)

def four_class_labels_from_binary(x):
    a = np.asarray(x)
    b = np.asarray([0] + list(np.diff(a)))
    c = a + b
    c[c == -1] = 3
    return c


"""
Label structure (
[0, 1]- (no touch, touch) , x1
[0, 1, 2, 3] (no touch, touch, onset, offset), x2
[0, 1]- (not onset, onset), 
[0, 1]- (not offset, offset), 
[0, 1, 2]-(no event, onset, offset) 

"""

base_dir_all_h5s = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/small_h5s/single_frame/'
def make_alt_labels_h5s(base_dir_all_h5s):
    for f in utils.get_h5s(base_dir_all_h5s):
        basename = '_ALT_LABELS.'.join(os.path.basename(f).split('.'))
        basedir = os.sep.join(f.split(os.sep)[:-2]) + os.sep + 'ALT_LABELS' + os.sep
        if not os.path.isdir(basedir):
            os.mkdir(basedir)
        new_h5_name = basedir+basename

        with h5py.File(f, 'r') as h:

            x1 = copy.deepcopy(h['labels'][:])# [0, 1]- (no touch, touch)

            x2 = four_class_labels_from_binary(x1)# [0, 1, 2, 3]- (no touch, touch, onset, offset)

            x3 = copy.deepcopy(x2)
            x3[x3 != 2] = 0
            x3[x3 == 2] = 1#[0, 1]- (not onset, onset)

            x4 = copy.deepcopy(x2)#[0, 1]- (not offset, offset)
            x4[x4 != 3] = 0
            x4[x4 == 3] = 1

            x5 = copy.deepcopy(x2)# [0, 1, 2]- (no event, onset, offset)
            x5[x5 == 1] = 0
            x5[x5 == 2] = 1
            x5[x5 == 3] = 2
        with h5py.File(new_h5_name, 'w') as h:
            h.create_dataset('[0, 1]- (no touch, touch)', shape=np.shape(x1), data=x1)
            h.create_dataset('[0, 1, 2, 3]- (no touch, touch, onset, offset', shape=np.shape(x2), data=x2)
            h.create_dataset('[0, 1]- (not onset, onset)', shape=np.shape(x3), data=x3)
            h.create_dataset('[0, 1]- (not offset, offset)', shape=np.shape(x4), data=x4)
            h.create_dataset('[0, 1, 2]- (no event, onset, offset)', shape=np.shape(x5), data=x5)




        plt.figure()
        plt.plot(x1+0)
        plt.plot(x2+4)
        plt.plot(x3+8)
        plt.plot(x4+12)
        plt.plot(x5+16)

