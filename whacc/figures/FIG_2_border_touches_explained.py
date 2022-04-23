from whacc import utils, image_tools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

y = np.zeros(2001)
ymax = .1
y[290:310] = ymax
y[350:355] = ymax
y[570:600] = ymax
shift_zero_by = 190
fig = plt.figure(figsize=(14, 4))
colors = ['r', 'g', 'k'] # can change these as needed
for i, borders in enumerate([3, 80]):
    b = utils.inds_around_inds(y, borders * 2 + 1)
    tmp1, _ = utils.group_consecutives(b)
    for k in tmp1:
        k = np.asarray(k)-shift_zero_by
        plt.plot(k, [ymax +.03 + (i/40)]*len(k), colors[i])
plt.plot(y[shift_zero_by:], colors[i+1])
plt.ylim([-.1, 1])
plt.xlim([0, 550])

custom_lines = [Line2D([0], [0], color=colors[0], lw=2),
                Line2D([0], [0], color=colors[1], lw=2),
                Line2D([0], [0], color=colors[2], lw=2)]

plt.legend(custom_lines, ['3 border', '80 border', 'touch trace'])



#
#
# h5_in = '/Users/phil/Desktop/holy_test_set_10_percent_3lag.h5'
# y = image_tools.get_h5_key_and_concatenate(h5_in, '[0, 1]- (no touch, touch)')
#
# h5_in = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/3lag/train_3lag.h5'
# y = image_tools.get_h5_key_and_concatenate(h5_in, 'labels')
#
# h5_in = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/small_h5s/3lag/small_test_3lag.h5'
# y = image_tools.get_h5_key_and_concatenate(h5_in, 'labels')
#
# utils.print_h5_keys(h5_in)
# """
# of course everything will be covered by 80 border because that is what the holy set is made up of!!
# replace the file with one of min or jons original data (ideally one with out any shift issues)
# """
#
# y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
#
# y = np.zeros(2001)
# ymax = .1
# y[290:310] = ymax
# y[350:355] = ymax
# y[570:600] = ymax
#
#
# plt.figure(figsize=(14, 4))
# colors = ['r', 'g', 'b', 'm']
# for i, borders in enumerate([3, 80]):
#
#     b = utils.inds_around_inds(y, borders * 2 + 1)
#     tmp1, _ = utils.group_consecutives(b)
#     for k in tmp1:
#         plt.plot(k, [ymax +.03 + (i/40)]*len(k), colors[i])
# plt.plot(y, 'k')
# plt.ylim([-.1, 1])
#
# from matplotlib.lines import Line2D
# custom_lines = [Line2D([0], [0], color='m', lw=2),
#                 Line2D([0], [0], color='b', lw=2),
#                 Line2D([0], [0], color='g', lw=2),
#                 Line2D([0], [0], color='r', lw=2),
#                 Line2D([0], [0], color='k', lw=2)]
#
#
# # fig, ax = plt.subplots()
# plt.legend(custom_lines, ['80 border', '40 border', '20 border', '3 border', 'touch trace'])
#
#
# x = np.zeros(201)
# x[90:110] = 1
# N = 161
# assert N / 2 != round(N / 2), 'N must be an odd number so that there are equal number of points on each side'
# cumsum = np.cumsum(np.insert(x, 0, 0))
# a = (cumsum[N:] - cumsum[:-N]) / float(N)
# a = np.where(a > 0)[0] + ((N - 1) / 2)
# a.astype('int')
#
#

##############
