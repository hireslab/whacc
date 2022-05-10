from natsort import natsorted
from whacc import utils
from whacc.feature_maker import get_feature_data_names
import numpy as np
import os
#
# h5_in = '/Users/phil/Dropbox/AH0407x160609_3lag_feature_data.h5'
#
# feature_list = natsorted(utils.lister_it(utils.print_h5_keys(h5_in, 1, 0), keep_strings='FD__'))
#
# final_feature_names, feature_names, feature_nums, feature_list_short = get_feature_data_names(feature_list)
#
# features_used_of_10 = utils.get_selected_features(greater_than_or_equal_to=4)
#
# d = dict()
# d['full_feature_names_and_neuron_nums'] = final_feature_names
# d['full_feature_names'] = feature_names
# d['full_neuron_nums'] = feature_nums
# d['feature_list_short_type'] = feature_list_short
# d['features_used_of_10'] =features_used_of_10
# d['feature_list_unaltered'] = feature_list
#
# utils.save_obj(d, '/Users/phil/Dropbox/HIRES_LAB/GitHub/whacc/whacc/whacc_data/feature_data/feature_data_dict')
#
#
#
#
#
#
#
#
# bd = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/final_feature_variable/'
# for k in utils.get_files(bd, '*'):
#     var_name = os.path.basename(k).strip('.pkl')
#     print(var_name)
#     exec_str = var_name + ' = utils.load_obj(k)'
#     exec(exec_str)
#
#
# utils.get_dict_info(d)
#
# d = dict()
# d['full_feature_names_and_neuron_nums'] = final_feature_names
# d['full_feature_names'] = feature_names
# d['full_neuron_nums'] = feature_nums
# d['feature_list_short_type'] = feature_list_short
# d['features_used_of_10'] = feature_did_contribute_count_of_10
# d['feature_list_unaltered'] = feature_list_WINDOWS
#
# utils.save_obj(d, '/Users/phil/Dropbox/HIRES_LAB/GitHub/whacc/whacc/whacc_data/feature_data/feature_data_dict')
#
#
# # from natsort import natsorted
# # natsorted(final_feature_names)[:20]
# #
# # sorted(final_feature_names)[:20]
#
# """
# can first sort by somthing ....
# then sort neuron number
# """



bd = '/Volumes/GoogleDrive-114825029448473821206/My Drive/final_features_window_version_corrected_v2/'
for k in utils.get_files(bd, '*'):
    var_name = os.path.basename(k).strip('.pkl')
    print(var_name)
    exec_str = var_name + ' = utils.load_obj(k)'
    exec(exec_str)


"""feature_list
feature_list_short
feature_names
feature_nums
final_feature_names"""

d = dict()
d['full_feature_names_and_neuron_nums'] = final_feature_names
d['full_feature_names'] = feature_names
d['full_neuron_nums'] = feature_nums
d['feature_list_short_type'] = feature_list_short
d['feature_list_unaltered'] = feature_list

# d['features_used_of_10'] = feature_did_contribute_count_of_10


# d = utils.load_obj('/Users/phil/Dropbox/HIRES_LAB/GitHub/whacc/whacc/whacc_data/feature_data/feature_data_dict.pkl')

fi_path ='/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/feature_index_for_feature_selection/only_gain_more_than_2_features_bool_2105_features.npy'
feature_index = np.load(fi_path)

d['final_selected_features_bool'] = feature_index

utils.save_obj(d, '/Users/phil/Dropbox/HIRES_LAB/GitHub/whacc/whacc/whacc_data/feature_data/feature_data_dict')
