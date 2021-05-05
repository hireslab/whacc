import h5py
import matplotlib.pyplot as plt
from whacc import image_tools
import os
from whacc import utils

for h5_file in utils.get_h5s('/Users/phil/Dropbox/Autocurator/data/samsons_subsets/OLD_2'):
    with h5py.File(h5_file, 'r') as h:
        plt.figure(figsize=[20, 10])
        plt.imshow(image_tools.img_unstacker(h['images'][:],40))
        plt.title(h5_file.split('/')[-1])
        plt.savefig(h5_file[:-3] + "_fig_plot.png")
        plt.close()


for k in utils.get_h5s('/Users/phil/Dropbox/Autocurator/data/samsons_subsets/OLD_2'):
    with h5py.File(k, 'r') as h:
        k2 = k[:-3]+'_TMP.h5'
        with h5py.File(k2, 'w') as h2:
            h2.create_dataset('all_inds', data=h['all_inds'][50:])
            h2.create_dataset('images', data=h['images'][50:])
            h2.create_dataset('in_range', data=h['in_range'][50:])
            h2.create_dataset('labels', data=h['labels'][50:])
            h2.create_dataset('retrain_H5_info', data=h['retrain_H5_info'][:])
    os.remove(k)
    os.rename(k2, k)


import h5py
import matplotlib.pyplot as plt
from whacc import image_tools

h5_file= '/Users/phil/Downloads/AH1159X18012021xS404_subset (1).h5'
with h5py.File(h5_file, 'r') as h:
    plt.figure(figsize=[20, 10])
    plt.imshow(image_tools.img_unstacker(h['images'][:],10))
    plt.title(h5_file.split('/')[-1])
    print(len(h['labels'][:]))
