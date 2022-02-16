# from whacc import subset_H5_generator
from whacc import utils, image_tools
from whacc.subset_h5_generator import subset_h5_generator
import numpy as np
import matplotlib.pyplot as plt
import cv2
H5_list = utils.get_h5s('/Users/phil/Dropbox/Autocurator/testing_data/MP4s/')

H5_list = ['/Volumes/GoogleDrive/My Drive/Colab data/curation_for_auto_curator/H5_data/ANM234232_140120_AH1030_AAAA_a.h5']
# tmp1 = '/Users/phil/Downloads/AH1120_200322__.h5'

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

to_pred_h5s = '/Volumes/GoogleDrive/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED'
H5_list_subset = utils.get_h5s(to_pred_h5s)
h5_file = H5_list_subset[-1]
tmp1 = utils.lister_it(utils.print_h5_keys(h5_file, 1, 0), keep_strings='MODEL_3_regular', remove_string='viterbi')
key_name = tmp1[45]
label_key_or_array = image_tools.get_h5_key_and_concatenate(h5_file, key_name).flatten()
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# label_key_or_array = 'labels'
sg = subset_h5_generator(H5_list[0], label_key_or_array)

sg.get_example_segments(seg_len_before_touch=8,
                        seg_len_after_touch=8,
                        min_y=.2,
                        max_y=.5,
                        num_to_sample=10,
                        min_seg_size=4,
                        start_and_stop_pole_times=[1300, 3000])

# you want a lot of brown which means the model isnt sure and these are teh best frames to correct
sg.plot_all_onset_or_offset(sg.onset_list, fig_size = [10, 10])
sg.plot_all_onset_or_offset(sg.offset_list, fig_size = [10, 10])

import matplotlib.pyplot as plt
import numpy as np
plt.figure()
plt.hist(sg.labels, 100) # histogram pf labels. should be mostly 0's and 1's if not then the predictions might be really bad

plt.figure()
for i, k in enumerate(sg.onset_list):
    plt.plot(np.asarray(range(len(k)))+len(k)*i, k, 'ok')
for i, k in enumerate(sg.offset_list):
    plt.plot(np.asarray(range(len(k)))+len(k)*i, k, 'or')

len_c = []
for k in sg.chunks:
    if len(k)<=100:
        len_c.append(len(k))
plt.figure()
plt.hist(len_c, 100)

save_on = False
if save_on:
    sg.save_subset_h5_file()


# change the labels into the correct human curated labels
H5_list = ['/Volumes/GoogleDrive/My Drive/Colab data/curation_for_auto_curator/H5_data/ANM234232_140120_AH1030_AAAA_a.h5']
to_pred_h5s = '/Volumes/GoogleDrive/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED'
H5_list_subset = utils.get_h5s(to_pred_h5s)
h5_file = H5_list_subset[-1]
real_bool = image_tools.get_h5_key_and_concatenate(h5_file, '[0, 1]- (no touch, touch)')
# '/Volumes/GoogleDrive/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED/ANM234232_140120_AH1030_AAAA_a_ALT_LABELS.h5'

vals = image_tools.get_h5_key_and_concatenate(sg.file_save_name, 'all_inds')
vals = real_bool[vals]
utils.add_to_h5(sg.file_save_name, 'labels', vals, overwrite_if_exists=True)


utils.get_class_info(sg)
