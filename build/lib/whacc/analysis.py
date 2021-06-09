import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import h5py
import copy
from tqdm import tqdm
import time
import os


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


class distance_metric():
    def __init__(self, real, predicted, axis=0, frame_num_array=None):
        self.real = real
        self.predicted = predicted
        self.axis = axis
        if frame_num_array is None:
            frame_num_array = [len(predicted)]
        # -----------
        self.pos_match = []
        self.neg_match = []
        self.pred_group_errors_1 = []
        self.pred_group_errors_0 = []
        self.group_inds_1 = []
        self.group_inds_0 = []
        # need to allow for going thorugh a list of lists where each list is a seperate trial
        for i1, i2 in self.loop_segments(frame_num_array):
            pos_match, neg_match = self.get_error_arrays(real[i1:i2], predicted[i1:i2], axis)
            pred_group_errors_1, group_inds_1 = self.pred_group_errors(predicted[i1:i2], pos_match, 1)
            pred_group_errors_0, group_inds_0 = self.pred_group_errors(predicted[i1:i2], neg_match, 0)
            group_inds_1 = group_inds_1 + i1
            group_inds_0 = group_inds_0 + i1

            self.pos_match = self.pos_match + list(pos_match)
            self.neg_match = self.neg_match + list(neg_match)
            self.pred_group_errors_1 = self.pred_group_errors_1 + list(pred_group_errors_1)
            self.pred_group_errors_0 = self.pred_group_errors_0 + list(pred_group_errors_0)
            self.group_inds_1 = self.group_inds_1 + list(group_inds_1)
            self.group_inds_0 = self.group_inds_0 + list(group_inds_0)
        # weird this is the only way I could make these into lists of lists instead of lists of arrays
        for i, (k1, k2) in enumerate(zip(self.group_inds_1, self.group_inds_0)):
            self.group_inds_1[i] = list(k1)
            self.group_inds_1[i] = list(k2)

        self.pred_group_errors_1_min = [np.nanmin(np.abs(k)) for k in self.pred_group_errors_1]
        self.pred_group_errors_1_max = [np.nanmax(np.abs(k)) for k in self.pred_group_errors_1]

        self.pred_group_errors_0_min = [np.nanmin(np.abs(k)) for k in self.pred_group_errors_0]
        self.pred_group_errors_0_max = [np.nanmax(np.abs(k)) for k in self.pred_group_errors_0]

        self.error_0a_splits, self.error_0b_misses, self.error_0c_deducts = self.class_type(self.pred_group_errors_0)
        self.error_1a_ghosts, self.error_1b_joins, self.error_1c_appends = self.class_type(self.pred_group_errors_1)

    @staticmethod
    def loop_segments(frame_num_array):
        frame_num_array = [0] + frame_num_array
        frame_num_array = np.cumsum(frame_num_array)
        return zip(list(frame_num_array[:-1]), list(frame_num_array[1:]))

    @staticmethod
    def find_closest(real_ind, predicted_ind, axis=0):
        c = real_ind[:, None] - predicted_ind

        if not list(c):  # on of the arrays inds is an empty list
            tmp1 = np.ma.masked_array(np.asarray([99999] * len(real_ind) + [99999] * len(predicted_ind)))
            return tmp1, tmp1

        closest_left = np.max(np.ma.masked_array(c, mask=[c > 0]), axis=axis)
        closest_left.fill_value = 999999

        closest_right = np.min(np.ma.masked_array(c, mask=[c < 0]), axis=axis)
        closest_right.fill_value = 999999  # used to be negative 999999

        return closest_left, closest_right

    def get_error_arrays(self, real, predicted, axis):

        real_ind = np.where(real == 1)[0]
        predicted_ind = np.where(predicted == 1)[0]
        left1, right1 = self.find_closest(real_ind, predicted_ind, axis=axis)

        real_ind = np.where(real == 0)[0]
        predicted_ind = np.where(predicted == 0)[0]
        left0, right0 = self.find_closest(real_ind, predicted_ind, axis=axis)

        L1 = left1.data.astype('float64')
        L1[np.abs(L1) > 99999] = np.nan
        R1 = right1.data.astype('float64')
        R1[np.abs(R1) > 99999] = np.nan
        L0 = left0.data.astype('float64')
        L0[np.abs(L0) > 99999] = np.nan
        R0 = right0.data.astype('float64')
        R0[np.abs(R0) > 99999] = np.nan

        out_1 = np.vstack((L1, R1))
        out_0 = np.vstack((L0, R0))

        return np.transpose(out_1), np.transpose(out_0)

    @staticmethod
    def pred_group_errors(predicted, in_match, match_type):
        """
    pred_bool the prediction bool
    in_match: either neg_match or pos_match
    match_type: either 1 or 0 depending on which one we are interested in. 0 for neg_match
    1 for pos_match
    """

        _, group_inds = utils.group_consecutives(np.where(predicted == match_type)[0])
        out = []
        for k in group_inds:
            out.append([in_match[kk, :][np.nanargmin(np.abs(in_match[kk, :]))] for kk in k])
        return out, group_inds

    @staticmethod
    def class_type(pred_group_errors_tmp):
        """
    logic: only one class can exist for each array
    all MISS_JOIN's have 2 ones... return
    all APPEND's are NOT MISS_JOIN that have a 0 and a 1 in them... return
    all SPLIT_GHOST's are remainder that are not MISS_JOIN or APPEND... return
    """
        a_SPLIT_GHOST = [False] * len(pred_group_errors_tmp)
        a_MISS_JOIN = [False] * len(pred_group_errors_tmp)
        a_APPEND = [False] * len(pred_group_errors_tmp)
        # logic: only one class can exist for each array
        for i, k in enumerate(pred_group_errors_tmp):
            a_MISS_JOIN[i] = np.sum(np.abs(k) == 1) == 2  # all MISS_JOIN's have 2 ones
            if not a_MISS_JOIN[i]:
                only_one_one = np.sum(np.abs(k) == 1) == 1
                at_least_one_zero = np.sum(np.abs(k) == 1) >= 1
                if only_one_one and at_least_one_zero:
                    a_APPEND[i] = True  # all APPEND's are NOT MISS_JOIN that have a 0 and a 1 in them
                else:
                    a_SPLIT_GHOST[i] = True  # all SPLIT_GHOST's are remainder that are not MISS_JOIN's or APPEND's
        return np.asarray(a_SPLIT_GHOST), np.asarray(a_MISS_JOIN), np.asarray(a_APPEND)
