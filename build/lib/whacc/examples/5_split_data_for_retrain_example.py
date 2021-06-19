from whacc import utils
from whacc import image_tools

bd = '/Users/phil/Dropbox/Autocurator/data/samsons_subsets/use/train_and_validate/'
H5_list_to_train = utils.get_h5s(bd)
H5_list_to_train = utils.lister_it(H5_list_to_train,
                                   keep_strings=['subset'])  # get only the H5 files with the word 'subset'
print(H5_list_to_train)
split_h5_files = image_tools.split_h5(H5_list_to_train, [8, 3], temp_base_name=[bd + 'training_set', bd + 'validation_set'],
                                      add_numbers_to_name=False)
#_________
bd = '/Users/phil/Dropbox/Autocurator/data/samsons_subsets/use/test/'
H5_list_to_train = utils.get_h5s(bd)
H5_list_to_train = utils.lister_it(H5_list_to_train,
                                   keep_strings=['subset'])  # get only the H5 files with the word 'subset'
print(H5_list_to_train)
split_h5_files = image_tools.split_h5(H5_list_to_train, [1], temp_base_name=[bd + 'test_set'],
                                      add_numbers_to_name=False)

# import h5py
# import numpy as np
# H5_list_to_train = utils.get_h5s('/Users/phil/Dropbox/Autocurator/data/samsons_subsets/use/test/')
# for k in H5_list_to_train:
#     with h5py.File(k, 'r') as h:
#         print(k)
#         print(len(np.unique(h['labels'][:])))
#         # print(h.keys())
#
# #
# # H5_list_to_train = utils.get_h5s('/Users/phil/Dropbox/Autocurator/data/samsons_subsets/train_and_validate/')
# # H5_list_to_train = utils.lister_it(H5_list_to_train,
# #                                    keep_strings=['subset'])  # get only the H5 files with the word 'subset'
# # print(H5_list_to_train)
# # bd = '/Users/phil/Dropbox/Autocurator/data/samsons_subsets/train_and_validate/'  # base directory to put files
# # split_h5_files = image_tools.split_h5(H5_list_to_train, [8, 3], temp_base_name=[bd + 'training', bd + 'validation'],
# #                                       add_numbers_to_name=False)
