# import numpy as np
# from whacc import utils
#
#
# class error_analysis():
#     def __init__(self, real_bool, pred_bool, frame_num_array=None):
#         self.real = real_bool
#         self.pred = pred_bool
#         self.type_list = ['ghost', 'ghost', 'append', 'miss', 'miss', 'deduct', 'join', 'split']
#         self.group_inds_neg = []
#         self.group_inds_pos = []
#         self.error_neg = []
#         self.error_pos = []
#         if frame_num_array is None:
#             frame_num_array = [len(self.pred)]
#
#         for i, (i1, i2) in enumerate(self.loop_segments(frame_num_array)):  # separate the trials
#             d = self.get_diff_array(self.real[i1:i2], self.pred[i1:i2])
#
#             R_neg, P_neg, X_neg, group_inds_neg = self.get_error_segments_plus_border(d, -1)
#             self.group_inds_neg += group_inds_neg
#             for each_x in X_neg:
#                 self.error_neg.append(self.get_error_type(each_x))
#
#             R_pos, P_pos, X_pos, group_inds_pos = self.get_error_segments_plus_border(d, 1)
#             self.group_inds_pos += group_inds_pos
#             for each_x in X_pos:
#                 self.error_pos.append(self.get_error_type(each_x))
#
#     @staticmethod
#     def loop_segments(frame_num_array):
#         frame_num_array = [0] + frame_num_array
#         frame_num_array = np.cumsum(frame_num_array)
#         return zip(list(frame_num_array[:-1]), list(frame_num_array[1:]))
#     @staticmethod
#     def one_sided_get_type(x):
#         if np.isnan(x[0]) and np.isnan(x[-1]):
#             if x[1] == -1:  # full trial ghost touch
#                 return -7
#             elif x[1] == 1:  # full trial miss touch
#                 return -4
#         if np.isnan(x[0]):  # edge case (error being parsed is @ index 0 or -1)
#             return -1
#         #                                          ghost.   ghost.   append.  miss.   miss.    deduct.
#         ind = np.where(np.all(x[:2] == np.asarray([[0, -1], [1, -1], [2, -1], [0, 1], [-1, 1], [2, 1]]), axis=1))[0][0]
#         return ind
#     @staticmethod
#     def type_parser(x):
#         minx = np.min(x)
#         maxx = np.max(x)
#         sort_vals = [minx, maxx]
#         if np.any(np.all(sort_vals == np.asarray([[1, 1], [1, 2], [2, 2]]), axis=1)):  # joins
#             return 6
#         elif maxx == minx and maxx == 5:  # splits
#             return 7
#         else:
#             return maxx  # everything else is already the correct index return the max
#
#     def get_error_type(self, x):
#         return self.type_parser([self.one_sided_get_type(x), self.one_sided_get_type(np.flip(x))])
#     @staticmethod
#     def get_diff_array(real_bool, pred_bool):
#         diff_array = np.asarray(real_bool) - np.asarray(pred_bool)
#         diff_array = diff_array + (real_bool + pred_bool == 2) * 2  # TP = 2, TN = 0, FP = -1, FN = 1
#         return diff_array
#     @staticmethod
#     def get_error_segments_plus_border(d, error_num_one_or_neg_one):
#         group_inds, _ = utils.group_consecutives(np.where(d == error_num_one_or_neg_one)[0])
#         R = []
#         P = []
#         X = []
#         for k in group_inds:
#             R_ind = k[-1] + 2
#             L_tmp = k[0] - 1
#             L_ind = np.max([L_tmp, 0])
#             #  for this error segment (plus edges) for ...
#             r = real_bool[L_ind:R_ind]  # real,
#             p = pred_bool[L_ind:R_ind]  # predicted and
#             x = d[L_ind:R_ind]  # the diff array (all match types).
#
#             if L_tmp < 0:  # edge case to the left
#                 r = np.append(np.nan, r)
#                 p = np.append(np.nan, p)
#                 x = np.append(np.nan, x)
#             if R_ind > len(d):  # edge case to the right
#                 r = np.append(r, np.nan)
#                 p = np.append(p, np.nan)
#                 x = np.append(x, np.nan)
#             R.append(r)
#             P.append(p)
#             X.append(x)
#
#         return R, P, X, group_inds
#
#
#
# from whacc import analysis
# import numpy as np
# real_bool = np.asarray([0, 0, 1, 1, 1, 1, 1])
# pred_bool = np.asarray([0, 1, 1, 0, 0, 0, 0])
# a = analysis.error_analysis(real_bool, pred_bool,[3,4])

from whacc import transfer_learning as tl
from whacc import image_tools
from whacc import utils

H5_list_to_train = ['/Users/phil/Dropbox/Colab data/RETRAIN_H5_data/RETRAIN_AH0407_160613_JC1003_AAAC.h5']
testing_h5_list = ['/Users/phil/Dropbox/Colab data/H5_data/AH0407_160613_JC1003_AAAC-006.h5']
testing_h5_list = H5_list_to_train


bd = '/Users/phil/Downloads/'
split_h5_files = image_tools.split_h5(H5_list_to_train, [8, 3], temp_base_name=[bd + 'training_set', bd + 'validation_set'],
                                      add_numbers_to_name=False)

validation_h5_list = [split_h5_files[1]]
training_h5_list = [split_h5_files[0]]

# all_models = utils.get_model_list("/Users/phil/Dropbox/HIRES_LAB/curation_for_auto_curator/model_iterations/model_iterations")
all_models = utils.get_model_list("/Users/phil/Dropbox/Colab data/model_iterations/")

batch_size = 50

# import tensorflow as tf
# model2 = tf.keras.models.load_model(all_models[12])

save_and_plot_history_1 = tl.save_and_plot_history(image_tools.ImageBatchGenerator(batch_size, testing_h5_list),
                                                   key_name='labels', plot_metric_inds=[1, 2])

training_info = tl.foo_start_running(all_models[12], training_h5_list, validation_h5_list, dropout=None, epochs=1000,
                                     batch_size=batch_size, learning_rate=10 ** -4, patience=10, monitor='val_loss',
                                     key_name='labels', verbose_=1, save_name=None, model_save_dir=None,
                                     save_best_only=False, add_callback_list=[save_and_plot_history_1])
