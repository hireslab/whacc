from natsort import os_sorted
from whacc import utils
from whacc.feature_maker import get_feature_data_names
import numpy as np

h5_in = '/Users/phil/Dropbox/AH0407x160609_3lag_feature_data.h5'

feature_list = os_sorted(utils.lister_it(utils.print_h5_keys(h5_in, 1, 0), keep_strings='FD__'))

final_feature_names, feature_names, feature_nums, feature_list_short = get_feature_data_names(feature_list)

features_used_of_10 = utils.get_selected_features(greater_than_or_equal_to=4)

d = dict()
d['full_feature_names_and_neuron_nums'] = final_feature_names
d['full_feature_names'] = feature_names
d['full_neuron_nums'] = feature_nums
d['feature_list_short_type'] = feature_list_short
d['features_used_of_10'] =features_used_of_10
d['feature_list_unaltered'] = feature_list

utils.save_obj(d, '/Users/phil/Dropbox/HIRES_LAB/GitHub/whacc/whacc/whacc_data/feature_data/feature_data_dict')





