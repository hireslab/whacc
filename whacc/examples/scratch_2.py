import h5py
import matplotlib.pyplot as plt
from whacc import image_tools


h5_file = "/Users/phil/Dropbox/Autocurator/data/samsons_subsets/test/AH1159X27012021xS414_subset_2.h5"
with h5py.File(h5_file, 'r') as h:
    plt.imshow(image_tools.img_unstacker(h['images'][:], 21))


