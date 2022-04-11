import copy

from whacc import utils, image_tools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as spio
import scipy.stats as stats
#
# def get_points_edge(arr_in, edge = 31):
#     tmp1 = (edge//2) - np.arange(edge)
#     tmp2 = np.where(arr_in)[0]
#     out = (tmp2[:, None] - tmp1[None, :])#.astype('float32')
#     return out
#
# def extract_array(array_in, ind_in):
#     out = np.ones_like(ind_in)*np.nan
#     place_ind0 = np.nanargmin(np.abs(ind_in), axis=1)
#     place_ind1 = np.nanargmax(np.abs(ind_in), axis=1)
#     for i, (i1, i2, ind) in enumerate(zip(place_ind0, place_ind1, ind_in)):
#         out[i, i1:i2+1] = array_in[ind[i1]:ind[i2]+1]
#     return out
#
# def get_onset_or_offset(arr_in, neg1or1 = 1):
#     # neg1or1 = 1 # 1 is onset -1 is offset
#     arr_in = 1*(arr_in>.5)
#     touch_onset =  (1*(np.hstack(([0], np.diff(arr_in))) == neg1or1)).astype('float')
#     return touch_onset
#
# def get_stacked_spike_arrays(touch_onset, poisson_array, edge = 11):
#     ind_in = get_points_edge(touch_onset, edge = edge)
#     out = extract_array(poisson_array, ind_in)
#     return out
#
# def make_poisson(on_or_offset, low = 3, high = 10):
#     tmp = copy.deepcopy(on_or_offset)
#     print(low, high)
#     tmp[tmp == 0] = low
#     tmp[tmp == 1] = high
#     print(tmp)
#     poisson_array_out = np.random.poisson(tmp)
#     return poisson_array_out
#
#
# noise_add = 0
# neg1or1 = 1
# onset_real_tmp1 = get_onset_or_offset(a[0], neg1or1 = neg1or1)
# # onset_real_tmp += .1
# plt.plot(onset_real_tmp1, 'r--', alpha=1)
# filt = np.asarray([100, 40, 30, 20, 10, 4, 3, 1, 1])/40
# assert len(filt)%2 == 1, 'must be odd'
#
# onset_real_tmp = np.pad(onset_real_tmp1, len(filt)-1)
# onset_real_tmp[onset_real_tmp==0] = 0.1
# onset_real = np.convolve(onset_real_tmp, filt, mode='valid')
# for  k in range(10):
#     spike_array_source = np.random.poisson(onset_real)
#     if noise_add>0:
#         spike_array_source+=  np.random.poisson(noise_add, len(onset_real))
#     spike_array_source = spike_array_source[:len(onset_real_tmp1)]
#     # poisson_array = make_poisson(onset_real, low = 100,high = 100)
#     plt.plot(spike_array_source, 'b-', alpha=0.2)
# # can cut to length of the original onset_real_tmp1
# plt.plot(onset_real[:len(onset_real_tmp1)], 'k-', alpha=1)
#
#
# def create_error_array(touch_mat, operation, onset, center, num_to_operate = 1):
#     touch_mat = copy.deepcopy(touch_mat)
#     if operation == 1:#deduct 2
#         op_name = 'deduct'
#         touch_mat[:num_to_operate, onset:onset+2] = 0
#     elif operation==2:#append 2
#         op_name = 'append'
#         touch_mat[:num_to_operate, onset-2:onset] = 1
#     elif operation==3:#split 3 at center
#         op_name = 'split'
#         touch_mat[:num_to_operate, center-1:center+2] = 0
#
#     elif operation==4: #ghost edge at last 3 places
#         op_name = 'ghost'
#         touch_mat[:num_to_operate, -3:] = 1
#     elif operation==5:#miss
#         op_name = 'miss'
#         touch_mat[-num_to_operate:, :] = 0 # reversed this so that it is different than join cause
#     #     it was literally the exact same plot and you couldnt see join
#     elif operation==6:#join
#         op_name = 'join'
#         touch_mat[:num_to_operate, center:] = 1
#         touch_mat[1:num_to_operate+1, :center] = 1
#
#
#     else:
#         assert False, 'operation number not valid'
#     return touch_mat.flatten(), op_name
#
# def make_fake_touch_array(num_touches = 10, len_touch = 11, in_touch_array_len = 31):
#     # num_touches = 10; len_touch = 11; in_touch_array_len = 31
#     assert in_touch_array_len%2 ==1, 'must be odd'
#     assert len_touch%2 ==1, 'must be odd'
#
#     center = in_touch_array_len//2
#     edge_len = len_touch//2
#     onset =  center - edge_len
#
#     t = np.zeros([num_touches, in_touch_array_len])
#     t[:, onset:onset+len_touch] = 1
#     return t, onset, center
# """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
#
#
t, onset, center = make_fake_touch_array(num_touches = 4, len_touch = 11, in_touch_array_len = 31)
plt.plot((1.1*-1)+t.flatten(), '-', label = 'real touch')
for k in range(6):
    t_mod, op_name = create_error_array(t, k+1, onset, center, num_to_operate = 1)
    plt.plot((1.1*k)+t_mod, '-', label=op_name)
plt.xlim([plt.xlim()[0], plt.xlim()[1]*1.5])
plt.legend()
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""



def poissonSpikeGen(fr, tSim, nTrials):
  # the spike matrix spikeMat and a time vector tVec,
  # a vector describing the time stamps for each column of the 2D matrix spikeMat.

  dt = 1/1000; # s
  tSim = tSim*dt
  nBins = np.floor(tSim/dt);
  spikeMat = np.random.rand(int(nTrials), int(nBins)) < fr*dt;
  tVec = np.arange(0,tSim, dt)
  return spikeMat, tVec
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
def get_SNR(a, win_size = 10, bl_move = 5):
    center = a.shape[1]//2
    signal = a[:, center:center+win_size]
    baseline = a[:, center-win_size-bl_move:center-bl_move]
    return signal.var()/baseline.var()

def get_sd_for_signal(a, win_size = 10, bl_move = 5, return_inds = False):
    center = a.shape[1]//2
    signal = a[:, center:center+win_size]
    baseline = a[:, center-win_size-bl_move:center-bl_move]
    # print(signal.shape)
    out = (signal - np.nanmean(baseline)) / np.std(baseline)
    if return_inds:
        return np.nanmean(out), [center, center+win_size, center-win_size-bl_move, center-bl_move]
    else:
        return np.nanmean(out)

def get_sd_for_signal(a, win_size = 10, bl_move = 5, return_inds = False):
    center = a.shape[1]//2
    signal = a[:, center:center+win_size]
    baseline = a[:, center-win_size-bl_move:center-bl_move]
    # print(signal.shape)
    out = (signal - np.nanmean(baseline)) / np.std(baseline)
    if return_inds:
        return np.nanmean(out), [center, center+win_size, center-win_size-bl_move, center-bl_move]
    else:
        return np.nanmean(out)


def exponential_decay(a, b, N):
    # a, b: exponential decay parameter
    # N: number of samples
    return a * (1-b) ** np.arange(N)
high = 1
len1 = 11
tmp1 = exponential_decay(high, 0.2, len1)
tmp2 = exponential_decay(high, 0.1, len1)
plt.plot(tmp1)
plt.plot(tmp2)
plt.plot(tmp1+tmp2)

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
filt = tmp1+tmp2
# make filter and poisson array
# filt = np.asarray([100, 40, 30, 20, 10, 4, 3, 1, 1])/40
# filt = np.asarray([100, 90, 70, 60, 20, 4, 3, 1, 1])/40
# filt = np.repeat(filt, 3).flatten()
# filt = exponential_decay(1, 0.1, 21)
# filt = exponential_decay(2.5, 0.05, 41)
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# 8*100 should be about the correct amount of touches for 100 trials about 8 per trial


def y_square(l, r):
    y = list(plt.ylim())
    return (l, y[0]), r-l, y[1]-y[0]

num_to_operate = 500
num_touches = 1500
baseline_p_num = .1
signal_p_num = .5
width_signal_to_extract = 51
max_peak_probability = 0.99
in_touch_array_len = 101


t, onset, center = make_fake_touch_array(num_touches = num_touches, len_touch = 31, in_touch_array_len = in_touch_array_len)
# noise_add = 1 # just for baseline adding there si still noise cause sampled form low alpha poisson
onset_real_tmp1 = get_onset_or_offset(t.flatten(), neg1or1 = 1)# 1 for onset
all_t = {'real touch':onset_real_tmp1.flatten()}
for k in range(6): # save all onsets from altered touch arrays
    t_mod, op_name = create_error_array(t, k+1, onset, center, num_to_operate = num_to_operate)
    t_mod_onsets = get_onset_or_offset(t_mod.flatten(), neg1or1 = 1)# 1 for onset
    all_t[op_name] = t_mod_onsets

assert len(filt)%2 == 1, 'must be odd'

onset_real_tmp = np.pad(onset_real_tmp1, len(filt)-1)
onset_real_tmp[onset_real_tmp==0] = baseline_p_num
onset_real_tmp[onset_real_tmp==1] = signal_p_num

onset_real = np.convolve(onset_real_tmp, filt, mode='valid')
# tmp1 = onset_real - np.min(onset_real)
# onset_real = max_peak_probability*(onset_real/np.max(onset_real))
# onset_real = onset_real_tmp[:len(onset_real_tmp1)]

# plt.plot(onset_real_tmp1[:100]+0.1)
# plt.plot(onset_real[:100])



# tmp1 = []
# for k in onset_real[:in_touch_array_len]:
#     [tmp2, _] = poissonSpikeGen(k, 1, num_touches)
#     tmp1.append(tmp2.flatten())
#
# spike_array_source = np.hstack(tmp1)


spike_array_source = onset_real

# spike_array_source = np.random.poisson(onset_real)

# if noise_add>0:
#     spike_array_source+=  np.random.poisson(noise_add, len(onset_real))

spike_array_source = 1*spike_array_source[:len(onset_real_tmp1)]
onset_real_final = onset_real[:len(onset_real_tmp1)]
# poisson_array = onset_real # set this to see just the pure lamda values of each type
final_spikes = []
fig = plt.figure()
for i, k in enumerate(all_t):

    out = get_stacked_spike_arrays(all_t[k], spike_array_source, edge = width_signal_to_extract)
    # plt.plot(np.std(out, axis  =0))
    # print(out.shape, k)
    y = np.nanmean(out, axis = 0)
    x_vals = np.arange(len(y))-len(y)//2
    # sem =stats.sem(out, axis = 0)
    # if i==0:
    #     subtract_me = 0
    # y=y-subtract_me#+(i/5)
    # print(np.mean(sem))
    # print(np.std(out, axis  =0))


    sd, inds_2_plot = get_sd_for_signal(out, win_size = 10, bl_move = 5, return_inds = True)
    sd = np.std(out, axis  =0)
    snr = get_SNR(out)

    tmp1 = plt.plot(x_vals, y, label = k)
    # plt.fill_between(x_vals, y-sd, y+sd, color=tmp1[0]._color, alpha=0.2)


    final_spikes.append(out)
plt.legend()



inds_2_plot = np.asarray(inds_2_plot)-len(y)//2
plt.gca().add_patch(plt.Rectangle(*y_square(*inds_2_plot[:2]),fill=True, color='g', alpha=0.5, figure=fig, zorder=-1000, edgecolor=None))
plt.gca().add_patch(plt.Rectangle(*y_square(*inds_2_plot[2:]),fill=True, color='r', alpha=0.5, figure=fig, zorder=-1000, edgecolor=None))



from matplotlib.patches import Rectangle

# def foo_square(fig, l, r, alpha, color):
#     y = list(plt.ylim())
#     print((l, y[0]), r-l, y[1]-y[0])
#     plt.gca().add_patch(plt.Rectangle((l, y[0]), r-l, y[1]-y[0], fill=True,
#                                       color='g', alpha=0.5, zorder=1000,
#                                   transform=fig.transFigure, figure=fig))


def get_sd_for_signal(a, win_size = 10, bl_move = 5):
    center = a.shape[1]//2
    signal = a[:, center:center+win_size]
    baseline = a[:, center-win_size-bl_move:center-bl_move]
    # print(signal.shape)
    print(np.nanmean(signal), np.nanmean(baseline), np.std(baseline))
    out = (signal - np.nanmean(baseline)) / np.std(baseline)
    return np.nanmean(out)


tmp1 = np.zeros(1000)
tmp1[:500] = 1
np.std(tmp1)




zscore = get_sd_for_signal(out, win_size = 10, bl_move = 5)


a = out;  win_size = 10; bl_move = 5

center = a.shape[1]//2
signal = a[:, center:center+win_size]
baseline = a[:, center-win_size-bl_move:center-bl_move]


signal.var()/baseline.var()

mbl = np.nanmean(baseline, axis = 1)
msig = np.nanmean(signal, axis = 1)
(signal - mbl[:, None])/mbl[:, None]

plt.plot(np.nanmean(a, axis = 0))

np.nanmean(signal - mbl[:, None])

# here peak response is   max (mean ( signal-mean(Baseline in that trial) / mean (baseline in that trial) ) ) in a certain time frame


foo_square(fig, inds_2_plot[0], inds_2_plot[1], .5, 'r')
plt.show()
inds_2_plot



fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(np.cumsum(np.random.randn(100)))
fig.patches.extend([plt.Rectangle((0.25,0.5),0.25,0.25,
                                  fill=True, color='g', alpha=0.5, zorder=1000,
                                  transform=fig.transFigure, figure=fig)])



ylim = list(plt.ylim()) ; ylim[0]  = 0 ; plt.ylim(ylim)





for k in final_spikes:
    print(get_sd_for_signal(k))



plt.plot(final_spikes[0][:100, :])


plt.plot(spike_array_source)
plt.plot(onset_real_final)

# plt.plot(onset_real_tmp)
# plt.plot(onset_real)

"""
make smoothing windows of some ms after the touch onset 
"""
all_out = []
num_count = 100
for _ in range(num_count):
    for k in a:
        onset_tmp = get_onset_or_offset(k, neg1or1 = neg1or1)
        out = get_stacked_spike_arrays(onset_tmp, spike_array_source)
        all_out.append(np.nanmean(out, axis = 0))
tmp1 = np.asarray(all_out)
# tmp1 = np.tile((np.tile(np.arange(7), [11, 1]).T), [2, 1])
tmp2 = np.reshape(tmp1, [num_count, 7, -1])
tmp3 = np.nanmean(tmp2, axis = 0)
plt.plot(tmp3.T)

plt.legend(['normal', 'deduct', 'append', 'split', 'ghost', 'miss','join' ])
# _____

plt.plot(tmp1)
plt.plot(onset_real)
# _____
"""
not getting the results I expected with these fake neurons because they are too close together, also missing a touch doesn;t 
affect the peak because all peaks are the same so an average doesnt affect it
IRL peaks are variable tho, also SE you'll see a difference there it will be less tight
"""


touch_onset =  (1*(np.hstack(([0], np.diff(arr_in))) == 1)).astype('float')
touch_onset =  (1*(np.hstack((np.diff(arr_in), [0])) == neg1or1)).astype('float')

get_points_edge(touch_onset, edge = 11)
# find min and max and locations, pop them in a nan init array


#

# def get_points_edge(arr_in, edge = 31, neg1or1 = 1):
#     touch_onset = 1*(np.hstack(([0], np.diff(arr_in))) == neg1or1)
#     touch_onset = np.pad(touch_onset, edge)
#     print(touch_onset.shape)
#     x = inds_around_inds(touch_onset, edge)
#     print(x)
#     return x#np.reshape(x, [-1, edge])


x = get_points_edge(a[0, :], 101)

arr_in = a[0, :]; neg1or1 = 1
touch_onset = np.hstack(([0], np.diff(arr_in))) == neg1or1




np.pad([1, 2, 1], 1)
