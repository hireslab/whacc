import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import h5py
from whacc import image_tools
from whacc import utils
import copy
from matplotlib import cm
from tqdm import tqdm
import time
import os
import pdb


class basic_metrics():
    def __init__(self, y, yhat_float, num_thresh=50):
        self.thresh = np.linspace(0, 1, num=num_thresh)
        self.y = y
        self.yhat = yhat_float
        self.TN = []
        self.FP = []
        self.FN = []
        self.TP = []
        self.Sensitivity = []
        self.Specificity = []
        self.Precision = []
        self.G_Mean = []
        self.Accuracy = []

        for k in self.thresh:
            yhat = (self.yhat > k) * 1
            cm = np.asarray(tf.math.confusion_matrix(self.y, yhat) / len(self.y)).flatten()
            self.TN.append(cm[0])
            self.FP.append(cm[1])
            self.FN.append(cm[2])
            self.TP.append(cm[3])
            self.Sensitivity.append(self.TP[-1] / (self.TP[-1] + self.FN[-1]))
            self.Specificity.append(self.TN[-1] / (self.TN[-1] + self.FP[-1]))
            self.Precision.append(self.TP[-1] / (self.TP[-1] + self.FP[-1]))
            self.Accuracy.append((self.TP[-1] + self.TN[-1]) / (self.TP[-1] + self.TN[-1] + self.FP[-1] + self.FN[-1]))
            self.G_Mean.append(np.sqrt(self.Sensitivity[-1] * self.Specificity[-1]))

    def best_geo_mean_ind(self):
        max_inds = np.where(self.G_Mean == np.max(self.G_Mean))[0]  # get inds of max g mean
        if len(max_inds) == 1:
            return max_inds[0]
        print(11111)
        groups, _ = utils.group_consecutives(max_inds)  # group those continious segments
        group_lens = [len(k) for k in groups]  # get the lengths of each segment
        max_len_group_ind = np.argmax(
            [group_lens])  # find the max length of those (default to the first one if there are more than one)
        ind_tmp = np.floor(group_lens[max_len_group_ind] / 2).astype('int')  # find teh middle point in that segment
        best_geo_mean_ind = groups[max_len_group_ind][
            ind_tmp]  # use the inds to get the middle number of the largest segment
        return best_geo_mean_ind


# def find_closest(real, predicted, axis=0):
#     """
#     helper function of "get_error_arrays" takes in the indices of a bool list and finds the closest matching values to
#     both the right and left in separate arrays
#     Parameters
#     ----------
#     real : inds of bool list
#     predicted : inds of bool list
#     axis : int
#
#     Returns
#     -------
#
#     """
#     c = real[:, None] - predicted
#     closest_left = np.max(np.ma.masked_array(c, mask=[c > 0]), axis=axis)
#     closest_left.fill_value = 999999
#     closest_right = np.min(np.ma.masked_array(c, mask=[c < 0]), axis=axis)
#     closest_right.fill_value = -999999
#     return closest_left, closest_right
#
#
# def get_error_arrays(real_bool, predicted_bool):
#     """
#     used to get 2 matrices, the first is 2 columns of length sum((real_bool==1)*1) the left column is the distance to
#     the closest 1 TO THE LEFT in predicted_bool for each 1 in real_bool, the right column is the closest to the right.
#     the second matrix is the same but for 0 instead of one, columns for this matrix are of length sum((real_bool==1)*0)
#     predicted bool
#     Parameters
#     ----------
#     real_bool :
#     predicted_bool :
#
#     Returns
#     -------
#     """
#     real = np.where(real_bool == 1)[0]
#     predicted = np.where(predicted_bool == 1)[0]
#     left1, right1 = find_closest(real, predicted)
#
#     real = np.where(real_bool == 0)[0]
#     predicted = np.where(predicted_bool == 0)[0]
#     left0, right0 = find_closest(real, predicted)
#
#     L1 = left1.data.astype('float64')
#     L1[L1 == 999999] = np.nan
#     R1 = right1.data.astype('float64')
#     R1[R1 == -999999] = np.nan
#     L0 = left0.data.astype('float64')
#     L0[L0 == 999999] = np.nan
#     R0 = right0.data.astype('float64')
#     R0[R0 == -999999] = np.nan
#     # pdb.set_trace()
#     out_1 = np.vstack((L1, R1))
#     out_0 = np.vstack((L0, R0))
#     print('inds are L1, R1 and  L0, R0')
#     return np.transpose(out_1), np.transpose(out_0)


class pole_plot():
    def __init__(self, img_h5_file, pred_val=None, true_val=None, threshold=0.5, len_plot=10, current_frame=0,
                 figsize=[10, 10]):
        """
        Examples
        ________
        a = analysis.pole_plot(
            '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/master_train_test1.h5',
            pred_val = [0,0,0,0,0,0,0,.2,.4,.5,.6,.7,.8,.8,.6,.4,.2,.1,0,0,0,0],
            true_val = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
            len_plot = 10)ax

        a.plot_it()
        """
        if true_val is None:
            if 'labels' in utils.print_h5_keys(img_h5_file, return_list=True, do_print=False):
                true_val = image_tools.get_h5_key_and_concatenate([img_h5_file])
        self.isnotebook = utils.isnotebook()
        self.img_h5_file = img_h5_file
        self.pred_val = np.asarray(pred_val)
        self.true_val = np.asarray(true_val)
        self.threshold = threshold
        self.len_plot = len_plot
        self.current_frame = current_frame
        self.figsize = figsize
        self.fig_created = False
        try:
            tmp3 = np.unique(np.concatenate((self.pred_val, self.true_val)))
            self.range_labels = list(range([np.nanmin(tmp3), np.nanmax(tmp3) + 1]))
            self.ylims = [np.nanmin(tmp3) - .5, np.nanmax(tmp3) + .5]
        except:
            self.range_labels = [0, 1]
            self.ylims = [-.5, 1.5]

        try:
            self.pred_val_bool = (1 * (self.pred_val > threshold)).flatten()
        except:
            self.pred_val_bool = np.asarray(None)

    def plot_it(self):

        if self.fig_created is False or self.isnotebook:  # we need to create a new fig every time if we are in colab or jupyter
            self.fig, self.axs = plt.subplots(2, figsize=self.figsize)
            self.fig_created = True
            plt.subplots_adjust(hspace=.001)
        self.axs[0].clear()
        self.axs[1].clear()
        self.fig.suptitle('Touch prediction')
        s1 = self.current_frame
        s2 = self.current_frame + self.len_plot
        # plt.axis('off')
        with h5py.File(self.img_h5_file, 'r') as h:
            self.current_imgs = image_tools.img_unstacker(h['images'][s1:s2], s2 - s1)
            # plt.imshow(self.current_imgs)
            self.axs[0].imshow(self.current_imgs)
        leg = []
        # axs[1].plot([None])
        if len(self.pred_val.shape) != 0:
            plt.plot(self.pred_val[s1:s2].flatten(), 'k-')
            leg.append('pred')
        # if len(self.pred_val_bool.shape) != 0:
        #     plt.plot(self.pred_val_bool[s1:s2].flatten(), '.g', markersize=10)
        #     leg.append('bool_pred')
        if len(self.true_val.shape) != 0:
            tmp1 = self.true_val[s1:s2].flatten()
            plt.scatter(range(len(tmp1)), tmp1, s=80, facecolors='none', edgecolors='r')
            leg.append('actual')
        if leg:
            plt.legend(leg, )
            plt.ylim(self.ylims)
            # plt.ylim([-.2, 1.2])

    def next(self):
        self.current_frame = self.current_frame + self.len_plot
        self.plot_it()

    def move(self, move_val):
        self.current_frame = self.current_frame + move_val
        self.plot_it()


class error_analysis():
    def __init__(self, real_bool, pred_bool, frame_num_array=None):
        frame_num_array = list(frame_num_array.astype(int))
        self.real = real_bool
        self.pred = pred_bool
        self.type_list = ['ghost', 'ghost', 'append', 'miss', 'miss', 'deduct', 'join', 'split']
        self.group_inds_neg = []
        self.group_inds_pos = []
        self.error_neg = []
        self.error_pos = []
        self.neg_add_to = []
        self.pos_add_to = []

        self.all_error_type = []
        self.all_errors = []
        self.all_error_nums = []
        if frame_num_array is None:
            frame_num_array = [len(self.pred)]

        for i, (i1, i2) in enumerate(self.loop_segments(frame_num_array)):  # separate the trials
            d = self.get_diff_array(self.real[i1:i2], self.pred[i1:i2])  # TP = 2, TN = 0, FP = -1, FN = 1

            R_neg, P_neg, X_neg, group_inds_neg = self.get_error_segments_plus_border(d, -1)
            cnt = -1
            self.group_inds_neg += group_inds_neg
            for each_x in X_neg:
                # cnt+=1
                # if cnt == 11:
                #     pdb.set_trace()
                self.error_neg.append(self.get_error_type(each_x))
                self.neg_add_to.append(i1)

            R_pos, P_pos, X_pos, group_inds_pos = self.get_error_segments_plus_border(d, 1)
            self.group_inds_pos += group_inds_pos

            for each_x in X_pos:
                self.error_pos.append(self.get_error_type(each_x))
                self.pos_add_to.append(i1)
        self.get_all_errors_sorted_final()

    def get_all_errors_sorted_final(self):
        for k, i, et in zip(self.group_inds_neg, self.neg_add_to, self.error_neg):
            self.all_errors.append(list(k + i))
            self.all_error_type.append(self.type_list[et])
            self.all_error_nums.append(et)
        for k, i in zip(self.group_inds_pos, self.pos_add_to):
            self.all_errors.append(list(k + i))
            self.all_error_type.append(self.type_list[et])
            self.all_error_nums.append(et)

        # i = np.argsort([k[0] for k in self.all_errors])
        # self.all_errors = [self.all_errors[k] for k in i]
        # self.all_error_type = [self.all_error_type[k] for k in i]
        # self.all_error_nums = [self.all_error_nums[k] for k in i]

    @staticmethod
    def loop_segments(frame_num_array):
        frame_num_array = [0] + list(frame_num_array)
        frame_num_array = np.cumsum(frame_num_array)
        return zip(list(frame_num_array[:-1]), list(frame_num_array[1:]))

    @staticmethod
    def one_sided_get_type(x):
        if np.isnan(x[0]) and np.isnan(x[-1]):
            if x[1] == -1:  # full trial ghost touch
                return -7
            elif x[1] == 1:  # full trial miss touch
                return -4
        if np.isnan(x[0]):  # edge case (error being parsed is @ index 0 or -1)
            return -1
        #                                          ghost.   ghost.   append.  miss.   miss.    deduct.
        ind = np.where(np.all(x[:2] == np.asarray([[0, -1], [1, -1], [2, -1], [0, 1], [-1, 1], [2, 1]]), axis=1))[0][0]
        return ind

    @staticmethod
    def type_parser(x):
        minx = np.min(x)
        maxx = np.max(x)
        sort_vals = [minx, maxx]
        if np.any(np.all(sort_vals == np.asarray([[1, 1], [1, 2], [2, 2]]), axis=1)):  # joins
            return 6
        elif maxx == minx and maxx == 5:  # splits
            return 7
        else:
            return maxx  # everything else is already the correct index return the max

    def get_error_type(self, x):
        return self.type_parser([self.one_sided_get_type(x), self.one_sided_get_type(np.flip(x))])

    @staticmethod
    def get_diff_array(real_bool, pred_bool):
        diff_array = np.asarray(real_bool) - np.asarray(pred_bool)
        diff_array = diff_array + (real_bool + pred_bool == 2) * 2  # TP = 2, TN = 0, FP = -1, FN = 1
        return diff_array

    def get_error_segments_plus_border(self, d, error_num_one_or_neg_one):
        group_inds, _ = utils.group_consecutives(np.where(d == error_num_one_or_neg_one)[0])
        R = []
        P = []
        X = []
        for k in group_inds:
            R_ind = k[-1] + 2
            L_tmp = k[0] - 1
            L_ind = np.max([L_tmp, 0])
            #  for this error segment (plus edges) for ...
            r = self.real[L_ind:R_ind]  # real,
            p = self.pred[L_ind:R_ind]  # predicted and
            x = d[L_ind:R_ind]  # the diff array (all match types).

            if L_tmp < 0:  # edge case to the left
                r = np.append(np.nan, r)
                p = np.append(np.nan, p)
                x = np.append(np.nan, x)
            if R_ind > len(d):  # edge case to the right
                r = np.append(r, np.nan)
                p = np.append(p, np.nan)
                x = np.append(x, np.nan)
            R.append(r)
            P.append(p)
            X.append(x)

        return R, P, X, group_inds


def name_filt(all_data, filt_string):
    keep_filt = []
    all_data[0].keys()
    for k in range(len(all_data)):
        if filt_string in all_data[k]['full_name']:
            keep_filt.append(k)

    return keep_filt


def name_filt_remove(all_data, filt_string):
    keep_filt = []
    all_data[0].keys()
    for k in range(len(all_data)):
        if filt_string not in all_data[k]['full_name']:
            keep_filt.append(k)

    return keep_filt


def performance_filter(all_data, key_name, greater_than=.8, less_than_or_qual_to=1):
    keep_filt = []
    for i, a in enumerate(all_data):
        log_ind = np.where(key_name == a['logs_names'])[0][0]
        val_list = a['all_logs'][:, log_ind]
        test1 = np.logical_and(val_list > greater_than, val_list <= less_than_or_qual_to).any()
        if test1:
            keep_filt.append(i)
    return keep_filt


def plot_acc(all_data, all_keep_filt, figsize=[15, 10]):
    all_leg = []
    plt.figure(figsize=figsize)
    all_colors = []
    for i in all_keep_filt:
        a = all_data[i]
        key_name = 'acc_val'
        log_ind = np.where(key_name == a['logs_names'])[0][0]

        p = plt.plot(a['all_logs'][:, log_ind])
        all_colors.append(p[0].get_color())
        all_leg.append(a['full_name'] + '_val')

        key_name = 'acc_test'
        log_ind = np.where(key_name == a['logs_names'])[0][0]
        plt.plot(a['all_logs'][:, log_ind], '--', color=all_colors[-1])
        all_leg.append(a['full_name'] + '_test')

    plt.ylim([.8, 1])
    plt.legend(all_leg)
    plt.ylabel('Accuracy', fontsize=22)
    plt.xlabel('Epochs', fontsize=22)


def plot_touch_segment_with_array_blocks(actual_h5_img_file, in_list_of_arrays=[], touch_number=0, border=4, height=20,
                                         img_width=61, color_list=None, cmap_col='inferno'):
    if color_list is None:
        color_list=[0, .5, .2, .3, .75, .85]

    if in_list_of_arrays == []:
        print('no input arrays, returning...')
        return
    color_dict = dict()
    cmap = cm.get_cmap(cmap_col)

    for i, k1 in enumerate(color_list):
        color_dict[i] = np.asarray(cmap(k1)[:-1]) * 255

    in_list_of_arrays = copy.deepcopy(in_list_of_arrays)
    tmp1, tmp2 = utils.group_consecutives(np.where(in_list_of_arrays[0] != 0)[0])

    inds = list(range(tmp1[touch_number][0] - border, tmp1[touch_number][-1] + 1 + border * 2))

    for i, k in enumerate(in_list_of_arrays):
        k = k.astype(float)
        if i == 0:
            tmp1 = np.tile(np.repeat(k[inds], img_width, axis=0), (height, 1))
        else:
            tmp1 = np.vstack((tmp1, np.tile(np.repeat(k[inds], img_width, axis=0), (height, 1))))
    tmp1 = np.stack((tmp1,) * 3, axis=-1)

    for kk in np.unique(tmp1.astype(int)):
        tmp3 = np.where(tmp1 == kk)
        for i1, i2 in zip(tmp3[0], tmp3[1]):
            tmp1[i1, i2, :] = color_dict[kk]

    tmp1 = tmp1.astype(int)
    with h5py.File(actual_h5_img_file, 'r') as h:
        tmp2 = image_tools.img_unstacker(h['images'][inds[0]:inds[-1] + 1], num_frames_wide=len(inds))
        tmp2 = np.vstack((tmp1, tmp2))
        return tmp2


# tmp2 = plot_touch_segment_with_array_blocks(actual_h5_img_file, in_list_of_arrays=[real, pred_m, pred_v],
#                                             touch_number=202, border=4, height=20, img_width=61,
#                                             color_list=[0, .5, .2, .3, .85, .95], cmap_col='inferno')
# plt.figure(figsize=(20, 10))
# plt.imshow(tmp2)
