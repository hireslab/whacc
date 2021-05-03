from whacc import utils
from whacc import image_tools

H5_list_to_train = utils.get_h5s('/Users/phil/Dropbox/Autocurator/data/samsons_subsets/')
H5_list_to_train = utils.lister_it(H5_list_to_train,
                                   keep_strings=['subset'], remove_string=['OLD_pole_down_included'])  # get only the H5 files with the word 'subset'
print(H5_list_to_train)
bd = '/Users/phil/Dropbox/Autocurator/data/samsons_subsets/test/'  # base directory to put files
split_h5_files = image_tools.split_h5(H5_list_to_train, [8, 3], temp_base_name=[bd + 'training_set', 'validation_set'],
                                      add_numbers_to_name=False)



import h5py
import numpy as np
H5_list_to_train = utils.get_h5s('/Users/phil/Dropbox/Autocurator/data/samsons_subsets/test/')
for k in H5_list_to_train:
    with h5py.File(k, 'r') as h:
        print(k)
        print(len(np.unique(h['labels'][:])))
        # print(h.keys())

#
# H5_list_to_train = utils.get_h5s('/Users/phil/Dropbox/Autocurator/data/samsons_subsets/train_and_validate/')
# H5_list_to_train = utils.lister_it(H5_list_to_train,
#                                    keep_strings=['subset'])  # get only the H5 files with the word 'subset'
# print(H5_list_to_train)
# bd = '/Users/phil/Dropbox/Autocurator/data/samsons_subsets/train_and_validate/'  # base directory to put files
# split_h5_files = image_tools.split_h5(H5_list_to_train, [8, 3], temp_base_name=[bd + 'training', bd + 'validation'],
#                                       add_numbers_to_name=False)