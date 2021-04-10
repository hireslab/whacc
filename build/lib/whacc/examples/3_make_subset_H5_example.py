# from whacc import subset_H5_generator
from whacc import utils
from whacc.subset_h5_generator import subset_h5_generator
H5_list = utils.get_h5s('/Users/phil/Dropbox/Autocurator/testing_data/MP4s/')

sg = subset_h5_generator(H5_list[0], 'labels')

sg.get_example_segments(seg_len_before_touch=10,
                        seg_len_after_touch=10,
                        min_y=.6,
                        max_y=.9,
                        num_to_sample=10,
                        min_seg_size=6,
                        start_and_stop_pole_times = [1000, 2500])

sg.plot_all_onset_or_offset(sg.onset_list, fig_size = [10, 10])
sg.plot_all_onset_or_offset(sg.offset_list, fig_size = [10, 10])


import matplotlib.pyplot as plt
import numpy as np
# plot the labels to see how they look. ideal is a lot near 0 or 1 and little in between
plt.figure()
plt.hist(sg.labels, 100)

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



sg.save_subset_h5_file()
