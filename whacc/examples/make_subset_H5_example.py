# from PSM_testbed.WhACC import subset_H5_generator
from PSM_testbed.WhACC import utils
from PSM_testbed.WhACC.subset_h5_generator import subset_h5_generator
H5_list = utils.get_h5s('/Users/phil/Dropbox/Autocurator/testing_data/MP4s/')

sg = subset_h5_generator(H5_list[0], 'labels')

sg.get_example_segments(seg_len_look_dist=10,
                             min_y=.9,
                             max_y=1,
                             num_to_sample=10,
                             num_high_prob_past_max_y=10,
                             start_and_stop_pole_times = [2000, 3000])

sg.plot_all_onset_or_offset(sg.onset_list, fig_size = [10, 10])
sg.plot_all_onset_or_offset(sg.offset_list, fig_size = [10, 10])

# import numpy as np
# import h5py
# frame_inds = [1, 2, 3]
# h5_img_key = 'images'
# im_stack = None
# for k in frame_inds:
#     with h5py.File(sg.h5_img_file, 'r') as h:
#         x = np.asarray(h[h5_img_key][k])
#         if im_stack is None:
#             im_stack = x
#         else:
#             im_stack = np.hstack((im_stack, x))
