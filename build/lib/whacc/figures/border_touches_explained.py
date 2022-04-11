
from whacc import utils, image_tools
import matplotlib.pyplot as plt
import numpy as np
h5_in = '/Users/phil/Desktop/holy_test_set_10_percent_3lag.h5'
utils.print_h5_keys(h5_in)
y = image_tools.get_h5_key_and_concatenate(h5_in, '[0, 1]- (no touch, touch)')
"""
of course everything will be covered by 80 border because that is what the holy set is made up of!!
replace the file with one of min or jons original data (ideally one with out any shift issues)
"""
plt.figure(figsize=(14, 4))
colors = ['r', 'g', 'b', 'm']
for i, borders in enumerate([3, 20, 40, 80]):
    b = utils.inds_around_inds(y, borders * 2 + 1)
    tmp1, _ = utils.group_consecutives(b)
    for k in tmp1:
        plt.plot(k, [1.03 + (i/40)]*len(k), colors[i])
plt.xlim(2000, 4000)
plt.plot(y, 'k')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='m', lw=2),
                Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], color='k', lw=2)]


# fig, ax = plt.subplots()
plt.legend(custom_lines, ['80 border', '40 border', '20 border', '3 border', 'touch trace'])
