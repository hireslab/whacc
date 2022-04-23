# # # import os
# # # import glob
# # #
# # # import cv2
# # # import numpy as np
# # # import time
# # # import re
# # # import h5py
# # # import matplotlib.pyplot as plt
# # # from functools import partial
# # # from tqdm import tqdm
# # # from PIL import Image
# # # from whacc.image_tools import h5_iterative_creator
# # # from sklearn.preprocessing import normalize
# # # from whacc import utils, image_tools, analysis
# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # # import copy
# # # from scipy.signal import medfilt
# # #
# # #
# # # def track_h5(template_image, h5_file, match_method='cv2.TM_CCOEFF', ind_list=None):
# # #     with h5py.File(h5_file, 'r') as h5:
# # #         if isinstance(template_image, int):  # if termplate is an ind to the images in the h5
# # #             template_image = h5['images'][template_image, ...]
# # #         elif len(template_image.shape) == 2:
# # #             template_image = np.repeat(template_image[:, :, None], 3, axis=2)
# # #
# # #         if ind_list is None:
# # #             ind_list = range(len(h5['labels'][:]))
# # #         # width and height of img_stacks will be that of template (61x61)
# # #         max_match_val = []
# # #         try:
# # #             method_ = eval(match_method)
# # #         except:
# # #             method_ = match_method
# # #         max_match_val = []
# # #         for frame_i in tqdm(ind_list):
# # #             img = h5['images'][frame_i, ...]
# # #             # Apply template Matching
# # #             if isinstance(method_, str):
# # #                 print('NOOOOOOOOOOOOOOO')
# # #                 if method_ == 'calc_diff':
# # #                     max_val = np.sum(np.abs(img.flatten() - template_image.flatten()))
# # #                 elif method_ == 'mse':
# # #                     max_val = np.mean((img.flatten() - template_image.flatten()) ** 2)
# # #             else:
# # #                 res = cv2.matchTemplate(img, template_image, method_)
# # #                 min_val, max_val, min_loc, top_left = cv2.minMaxLoc(res)
# # #             max_match_val.append(max_val)
# # #             # top_left = np.flip(np.asarray(top_left))
# # #     return max_match_val, template_image
# # #
# # #
# # # x = '/Users/phil/Downloads/test_pole_tracker/AH0667x170317.h5'
# # # utils.print_h5_keys(x)
# # # max_val_stack = image_tools.get_h5_key_and_concatenate(x, 'max_val_stack')
# # # locations_x_y = image_tools.get_h5_key_and_concatenate(x, 'locations_x_y')
# # # trial_nums_and_frame_nums = image_tools.get_h5_key_and_concatenate(x, 'trial_nums_and_frame_nums')
# # # template_img = image_tools.get_h5_key_and_concatenate(x, 'template_img')
# # # frame_nums = trial_nums_and_frame_nums[1, :].astype(int)
# # # trial_nums = trial_nums_and_frame_nums[0, :].astype(int)
# # # asdfasdfasdf
# # # method = 'TM_CCOEFF_NORMED'
# # # ind_list = None
# # # max_match_val_new, template_image_out = track_h5(template_img, x, match_method='cv2.' + method, ind_list=ind_list)
# # #
# # # method = 'calc_diff'
# # # ind_list = None
# # # template_img = 2000
# # # max_match_val_new, template_image_out = track_h5(template_img, x, match_method=method, ind_list=ind_list)
# # #
# # # method = 'mse'
# # # ind_list = None
# # # template_img = 2000
# # # max_match_val_new, template_image_out = track_h5(template_img, x, match_method=method, ind_list=ind_list)
# # #
# # # for k1, k2 in utils.loop_segments(frame_nums):
# # #     plt.plot(max_match_val_new[k1:k2], linewidth=.3)
# # # plt.legend(trial_nums)
# # #
# # # match_list = ['TM_SQDIFF_NORMED', 'TM_CCORR_NORMED', 'TM_CCOEFF_NORMED', 'TM_SQDIFF', 'TM_CCORR', 'TM_CCOEFF']
# # #
# # # h5_file = '/Users/phil/Downloads/test_pole_tracker/AH0667x170317.h5'
# # # meth_dict = dict()
# # # meth_dict['h5_file'] = h5_file
# # # ind_list = None
# # # for template_image_ind in [0, 2000]:
# # #     for method in match_list:
# # #         max_match_val_new, template_image = track_h5(template_image_ind, h5_file, match_method='cv2.' + method,
# # #                                                      ind_list=ind_list)
# # #         meth_dict['ind_' + str(template_image_ind) + '_' + method] = max_match_val_new
# # #
# # # for method in match_list:
# # #     max_match_val_new, template_image = track_h5(template_img, h5_file, match_method='cv2.' + method, ind_list=ind_list)
# # #     meth_dict['ind_template_img_' + method] = max_match_val_new
# # #
# # # fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=False)
# # # ax_list = fig.axes
# # # cnt = -1
# # # for k in meth_dict:
# # #     if 'h5_file' not in k and 'NORM' in k:
# # #         cnt += 1
# # #         if len(ax_list) == cnt:
# # #             cnt = 0
# # #             fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=False)
# # #             ax_list = fig.axes
# # #         ax1 = ax_list[cnt]
# # #         ax1.set_title(k)
# # #         # plt.title(k)
# # #         for k1, k2 in utils.loop_segments(frame_nums):
# # #             try:
# # #                 x = np.asarray(meth_dict[k][k1:k2])
# # #                 # ax1.plot(x-x[0],linewidth=.3, alpha = 1)
# # #                 ax1.plot(x, linewidth=.3, alpha=1)
# # #             except:
# # #                 break
# # # plt.legend(trial_nums)
# # #
# # # # plt.imshow(image_tools.get_h5_key_and_concatenate(h5_file, 'images'))
# # #
# # # a = analysis.pole_plot('/Users/phil/Downloads/test_pole_tracker/AH0667x170317.h5')
# # #
# # # a.current_frame = 0
# # # a.plot_it()
# # #
# # # a.current_frame = 1000
# # # a.plot_it()
# # # """"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # #
# # # h5_file = '/Users/phil/Downloads/test_pole_tracker/AH0667x170317.h5'
# # # method = 'cv2.TM_CCOEFF'
# # # method = 'cv2.TM_CCOEFF_NORMED'
# # # method = 'cv2.TM_SQDIFF_NORMED'
# # # ind_list = None
# # # template_image_ind = 2000  # know this is a good starting point with no whiskers in it
# # # ls = np.asarray(utils.loop_segments(frame_nums, returnaslist=True))
# # # all_maxes = []
# # # trial_inds = range(len(frame_nums))
# # # self_references_frame_compares = np.zeros(np.sum(frame_nums))
# # # max_match_all = []
# # # max_match_all2 = []
# # # trial_ind_all = []
# # # template_img_all = []
# # # template_image_ind_all = []
# # # for k in range(len(frame_nums)):
# # #     template_image_ind_all.append(template_image_ind)
# # #     max_match, template_img = track_h5(int(template_image_ind), h5_file, match_method=method, ind_list=ind_list)
# # #     template_img_all.append(template_img)
# # #     max_match_all.append(np.asarray(max_match))
# # #     max_match_all2.append(np.asarray(max_match))
# # #     trial_ind = np.where(template_image_ind < np.cumsum(frame_nums))[0][0]
# # #     trial_ind_all.append(trial_ind)
# # #     self_references_frame_compares[ls[0, trial_ind]:ls[1, trial_ind]] = max_match[ls[0, trial_ind]:ls[1, trial_ind]]
# # #     if k == len(frame_nums)-1:
# # #         break
# # #     for kt in trial_ind_all:
# # #         for kk in max_match_all:
# # #             kk[ls[0, kt]:ls[1, kt]] = np.nan
# # #             kk[ls[0, kt]:ls[1, kt]] = np.nan
# # #     _val = -99999999999
# # #     _ind = -99999999999
# # #     for kk in max_match_all:
# # #         tmp1 = np.nanmax(kk)
# # #         tmp2 = np.nanargmax(kk)
# # #         if tmp1 > _val:
# # #             _val = copy.deepcopy(tmp1)
# # #             _ind = copy.deepcopy(tmp2)
# # #     # template_image_ind = copy.deepcopy(_ind)
# # #     template_image_ind = template_image_ind+4000
# # #
# # # kernel_size = 1
# # # for k1, k2 in utils.loop_segments(frame_nums):
# # #     plt.plot(medfilt(self_references_frame_compares[k1:k2], kernel_size=kernel_size), linewidth = 0.3)
# # # plt.legend(range(len(frame_nums)))
# # # plt.title(method)
# # #
# # #
# # # # pred_bool_smoothed = medfilt(copy.deepcopy(pred_bool_temp), kernel_size=kernel_size)
# # #
# # # fig, ax = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=False)
# # # ax_list = fig.axes
# # # for i, k in enumerate(ax_list):
# # #     k.imshow(template_img_all[i])
# # #     k.set_title(template_image_ind_all[i])
# # #
# # #
# # #
# # # x = np.mean(np.asarray(max_match_all2), axis = 0)
# # # kernel_size = 1
# # # for k1, k2 in utils.loop_segments(frame_nums):
# # #     plt.plot(medfilt(x[k1:k2], kernel_size=kernel_size), linewidth = 0.3)
# # # plt.legend(range(len(frame_nums)))
# # # plt.title(method)
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # #
# # # h5_file = '/Users/phil/Dropbox/HIRES_LAB/curation_for_auto_curator/Data/Jon/AH0667/170317/AH0667x170317.h5'
# # # trial_nums_and_frame_nums = image_tools.get_h5_key_and_concatenate(h5_file, 'trial_nums_and_frame_nums')
# # # frame_nums = trial_nums_and_frame_nums[1, :].astype(int)
# # #
# # # method = 'cv2.TM_CCORR_NORMED'
# # # frame_to_compare = 2000
# # # testing_frames_start = 1250
# # # testing_frames_len = 50
# # #
# # # method = 'cv2.TM_CCOEFF'#console regular# best
# # # frame_to_compare = 1
# # # testing_frames_start = 1250
# # # testing_frames_len = 50
# # #
# # # # method = 'cv2.TM_CCOEFF' #console (1)
# # # # frame_to_compare = 2000
# # # # testing_frames_start = 1250
# # # # testing_frames_len = 50
# # #
# # # ind_list = None
# # # all_tests = []
# # # for ktrial, _ in utils.loop_segments(frame_nums):
# # #     template_image_ind = frame_to_compare+ktrial
# # #     max_match_all = []
# # #     for k1, k2 in utils.loop_segments(frame_nums):
# # #         ind_list = np.arange(testing_frames_start, testing_frames_start+testing_frames_len) + k1
# # #         max_match, template_img = track_h5(int(template_image_ind), h5_file, match_method=method, ind_list=ind_list)
# # #         max_match = np.asarray(max_match).astype(float)
# # #         max_match_all.append(max_match-max_match[0])
# # #     all_tests.append(np.asarray(max_match_all).flatten())
# # #
# # # all_var = []
# # # for i, k in enumerate(all_tests):
# # #     addto = (10**6)*i*2
# # #     k = k[(k>np.quantile(k,0.1)) & (k<np.quantile(k,0.9))]
# # #     plt.plot(k+addto, '.', markersize = 0.3)
# # #     # plt.plot(k+addto, '-k', linewidth = 0.05)
# # #     all_var.append(np.var(k))
# # # plt.legend(np.argsort(all_var))
# # #
# # # plt.figure()
# # # for i, k in enumerate(all_tests):
# # #     addto = (10**6)*i*2
# # #     plt.plot(k+addto, '.', markersize = 0.3)
# # #
# # #
# # #
# # #
# # #
# # # k1, k2 = utils.loop_segments(frame_nums, returnaslist=True)
# # # template_image_ind = frame_to_compare+k1[np.argmin(all_var)]
# # #
# # # ind_list = None
# # # max_match, template_img = track_h5(int(template_image_ind), h5_file, match_method=method, ind_list=ind_list)
# # #
# # # locations_x_y = image_tools.get_h5_key_and_concatenate(h5_file, 'locations_x_y')
# # # tmp1 = np.argsort(locations_x_y[:, 0][2000::4000])
# # #
# # # k1, k2 = utils.loop_segments(frame_nums, returnaslist = True)
# # #
# # # for i, k in enumerate(tmp1):
# # #     addto = i*10**6
# # #     plt.plot(np.asarray(max_match[k1[k]:k2[k]])+addto, linewidth=0.3)
# # #
# # #
# # # for i, k in enumerate(tmp1):
# # #     addto = i*10**-3
# # #     plt.plot(np.asarray(max_match[k1[k]:k2[k]])+addto, linewidth=0.3)
# # #
# # #
# # #
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # # k1, k2 = utils.loop_segments(frame_nums, returnaslist=True)
# # # all_var_inds = np.argsort(all_var)
# # # all_max = []
# # # for ii in range(4):
# # #     template_image_ind = frame_to_compare+k1[all_var_inds[ii]]
# # #     ind_list = None
# # #     max_match, template_img = track_h5(int(template_image_ind), h5_file, match_method=method, ind_list=ind_list)
# # #     all_max.append(max_match)
# # #
# # #
# # # max_match_mean = np.nanmean(np.asarray(all_max), axis = 0)
# # # tmp1 = np.argsort(locations_x_y[:, 0][2000::4000])
# # # for i, k in enumerate(tmp1):
# # #     addto = i*10**6
# # #     plt.plot(np.asarray(max_match_mean[k1[k]:k2[k]])+addto, linewidth=0.3)
# # #
# # #
# # # x = np.asarray(all_max)
# # # for i, k in enumerate(tmp1):
# # #     addto = i*10**6
# # #     x1  = x[0][k1[k]:k2[k]]+addto
# # #     x2  = x[1][k1[k]:k2[k]]+addto
# # #     plt.plot(x1-x2, linewidth=0.3)
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # #
# # #
# # # from IPython.utils import io
# # #
# # # vit = dict()
# # # vit['all_acc_before_no_pole_mask'] = []
# # # vit['all_acc_after_no_pole_mask'] = []
# # # vit['all_acc_before'] = []
# # # vit['all_acc_after'] = []
# # # vit['h5_img_file'] = []
# # # vit['h5_img_file_full_dir'] = []
# # #
# # # vit['m_name'] = m_names
# # # vit['L_key']= label_key
# # # vit['vm_name'] = vit_m_names
# # # vit['h5_img_file_full_dir']= to_pred_h5s
# # # for k in vit['h5_img_file_full_dir']:
# # #   vit['h5_img_file'].append(os.path.basename(k))
# # #
# # # for h5_img_file in to_pred_h5s:
# # #   in_range = image_tools.get_h5_key_and_concatenate([h5_img_file], 'in_range')
# # #   tmp1 = []
# # #   tmp2 = []
# # #   tmp3 = []
# # #   tmp4 = []
# # #   for iii, (vm_name, m_name, L_key) in enumerate(tzip(vit_m_names, m_names, label_key)):
# # #     pred_m_raw = image_tools.get_h5_key_and_concatenate([h5_img_file], key_name=m_name)
# # #     pred_v = image_tools.get_h5_key_and_concatenate([h5_img_file], key_name=vm_name)
# # #     real = image_tools.get_h5_key_and_concatenate([h5_img_file], key_name=L_key)
# # #     if pred_m_raw.shape[1] ==1:
# # #       pred_m = ((pred_m_raw>0.5)*1).flatten()
# # #     else:
# # #       pred_m = np.argmax(pred_m_raw, axis = 1)# turn into integers instead of percentages
# # #
# # #     # get everything back to binary (if possible)
# # #     with io.capture_output() as captured:#prevents crazy printing
# # #
# # #       pred_m_bool = utils.convert_labels_back_to_binary(pred_m, L_key)
# # #       real_bool = utils.convert_labels_back_to_binary(real, L_key)
# # #       pred_v_bool = utils.convert_labels_back_to_binary(pred_v, L_key)
# # #     if real_bool is None: # convert labels will return None if it cant convert
# # #     #it back to the normal format. i.e. only onset or only offsets...
# # #       tmp1.append(0)
# # #       tmp2.append(0)
# # #       tmp3.append(0)
# # #       tmp4.append(0)
# # #     else:
# # #       tmp1.append(np.mean(real_bool == pred_m_bool))
# # #       tmp2.append(np.mean(real_bool == pred_v_bool))
# # #       tmp3.append(np.mean(real_bool*in_range == pred_m_bool*in_range))
# # #       tmp4.append(np.mean(real_bool*in_range == pred_v_bool*in_range))
# # #
# # #   vit['all_acc_before_no_pole_mask'].append(tmp1)
# # #   vit['all_acc_after_no_pole_mask'].append(tmp2)
# # #   vit['all_acc_before'].append(tmp3)
# # #   vit['all_acc_after'].append(tmp4)
# # #   # vit['m_name'] = list(dict.fromkeys(vit['m_name']))
# # #   # vit['L_key'] = list(dict.fromkeys(vit['L_key']))
# # #   # vit['h5_img_file_full_dir'].append(h5_img_file)
# # #
# # #
# # # vit2 = copy.deepcopy(vit)
# # # ############################################################################################################
# # # ############################################################################################################
# # # ############################################################################################################
# # # ############################################################################################################
# # # ############################################################################################################
# # # ############################################################################################################
# # # ############################################################################################################
# # # ############################################################################################################
# # #
# # # # def grab_frames(x, lstm_len, end_frame_num):
# # # #     # if
# # #
# # # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/regular/train_regular.h5'
# # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/small_h5s/3lag/small_test_3lag.h5'
# # # utils.print_h5_keys(h5)
# # # # (None, 5, 96, 96, 3)
# # # lstm_len = 5
# # # assert lstm_len%2 == 1
# # # with h5py.File(h5, 'r') as h:
# # #     print(len(h['images']))
# # #     batch_size = 20
# # #     print(h['images'][0:batch_size].shape)
# # #     x = h['images'][20:20 + batch_size]
# # #     x2 = x[:, None, ...]
# # #     print(x2.shape)
# # #     print(x2[:-5, ...].shape)
# # #     # print(x[4:-1, ...].shape)
# # #     x3 = np.stack(( x[:-4, ...],  x[1:-3, ...], x[2:-2, ...], x[3:-1, ...], x[4:, ...]), axis = 1)
# # #     print(x3.shape)
# # #     print(x.shape)
# # #     # print(x[4:-1, ...].shape)
# # #     y = x3[0]
# # #     y2 = y
# # #
# # #     plt.figure()
# # #     for i, k in enumerate(y):
# # #         plt.subplot(3,2, i+1)
# # #         plt.imshow(k)
# # #
# # #
# # #
# # # # finished extractor........
# # # lstm_len = 5
# # # assert lstm_len%2 == 1, "number of images must be odd"
# # # batch_size = 10
# # # chunk_num = 24
# # #
# # # lstm_len//2
# # #
# # # with h5py.File(h5, 'r') as h:
# # #     b = lstm_len//2
# # #     tot_len = h['images'].shape[0]
# # #     i1 = chunk_num * batch_size - b
# # #     i2 = chunk_num * batch_size + batch_size + b
# # #     edge_left_trigger = abs(min(i1, 0))
# # #     edge_right_trigger = abs(min(tot_len-i2, 0))
# # #     x = h['images'][max(i1, 0):min(i2, tot_len)]
# # #     print(x.shape)
# # #     if edge_left_trigger+edge_right_trigger>0: # in case of edge cases
# # #         pad_shape = list(x.shape); pad_shape[0] = edge_left_trigger+edge_right_trigger
# # #         pad = np.zeros(pad_shape).astype('uint8')
# # #         if edge_left_trigger>edge_right_trigger:
# # #             x = np.concatenate((pad, x), axis = 0)
# # #         else:
# # #             x = np.concatenate((x, pad), axis = 0)
# # #
# # #     s = list(x.shape)
# # #     s.insert(1, lstm_len)
# # #     out = np.zeros(s).astype('uint8')
# # #
# # #
# # #     for i in range(lstm_len):
# # #         i1 = max(0, b-i)
# # #         i2 = min(s[0], s[0]+b-i)
# # #         i3 = max(0, i-b)
# # #         i4 = min(s[0], s[0]+i-b)
# # #         print('take ', i3,' to ',  i4, ' and place in ', i1,' to ', i2)
# # #         out[i1:i2, i, ...] = x[i3:i4, ...]
# # #     out = out[b:s[0]-b, ...]
# # #
# # #     y = out[-1]
# # #     print(out.shape)
# # #     plt.figure()
# # #     for i, k in enumerate(y):
# # #         plt.subplot(3,2, i+1)
# # #         # plt.figure()
# # #         plt.imshow(k)
# # #
# # #
# # #
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # #
# # #
# # # # finished extractor........
# # #
# # #
# # #
# # # lstm_len = 5
# # #
# # # batch_size = 10
# # # chunk_num = 24
# # # with h5py.File(h5, 'r') as h:
# # #     b = lstm_len//2
# # #     tot_len = h['images'].shape[0]
# # #     i1 = chunk_num * batch_size - b
# # #     i2 = chunk_num * batch_size + batch_size + b
# # #     edge_left_trigger = abs(min(i1, 0))
# # #     edge_right_trigger = abs(min(tot_len-i2, 0))
# # #     x = h['images'][max(i1, 0):min(i2, tot_len)]
# # #     print(x.shape)
# # #     if edge_left_trigger+edge_right_trigger>0: # in case of edge cases
# # #         pad_shape = list(x.shape); pad_shape[0] = edge_left_trigger+edge_right_trigger
# # #         pad = np.zeros(pad_shape).astype('uint8')
# # #         if edge_left_trigger>edge_right_trigger:
# # #             x = np.concatenate((pad, x), axis = 0)
# # #         else:
# # #             x = np.concatenate((x, pad), axis = 0)
# # #
# # #     s = list(x.shape)
# # #     s.insert(1, lstm_len)
# # #     out = np.zeros(s).astype('uint8')
# # #
# # #
# # #     for i in range(lstm_len):
# # #         i1 = max(0, b-i)
# # #         i2 = min(s[0], s[0]+b-i)
# # #         i3 = max(0, i-b)
# # #         i4 = min(s[0], s[0]+i-b)
# # #         print('take ', i3,' to ',  i4, ' and place in ', i1,' to ', i2)
# # #         out[i1:i2, i, ...] = x[i3:i4, ...]
# # #     out = out[b:s[0]-b, ...]
# # #
# # #     y = out[-1]
# # #     print(out.shape)
# # #     plt.figure()
# # #     for i, k in enumerate(y):
# # #         plt.subplot(3,2, i+1)
# # #         # plt.figure()
# # #         plt.imshow(k)
# # #
# # #
# # #
# # #
# # # import tensorflow as tf
# # # from tensorflow import keras
# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # # import h5py
# # # import copy
# # # import time
# # # import os
# # # from whacc import utils
# # # import pdb
# # #
# # # def reset_to_first_frame_for_each_file_ind(file_inds_for_H5_extraction):
# # #     """reset_to_first_frame_for_each_file_ind - uses the output of batch_size_file_ind_selector
# # #     to determine when to reset the index for each individual H5 file. using the above example
# # #     the out put would be [0, 0, 2, 2, 2, 5, 5], each would be subtracted from the indexing to
# # #     set the position of the index to 0 for each new H5 file.
# # #
# # #     Parameters
# # #     ----------
# # #     file_inds_for_H5_extraction :
# # #
# # #
# # #     Returns
# # #     -------
# # #
# # #
# # #     """
# # #     subtract_for_index = []
# # #     for k, elem in enumerate(file_inds_for_H5_extraction):
# # #         tmp1 = np.diff(file_inds_for_H5_extraction)
# # #         tmp1 = np.where(tmp1 != 0)
# # #         tmp1 = np.append(-1, tmp1[0]) + 1
# # #         subtract_for_index.append(tmp1[np.int(file_inds_for_H5_extraction[k])])
# # #     return subtract_for_index
# # #
# # #
# # # def batch_size_file_ind_selector(num_in_each, batch_size):
# # #     """batch_size_file_ind_selector - needed for ImageBatchGenerator to know which H5 file index
# # #     to use depending on the iteration number used in __getitem__ in the generator.
# # #     this all depends on the variable batch size.
# # #
# # #     Example: the output of the following...
# # #     batch_size_file_ind_selector([4000, 4001, 3999], [2000])
# # #     would be [0, 0, 1, 1, 1, 2, 2] which means that there are 2 chunks in the first
# # #     H5 file, 3 in the second and 2 in the third based on chunk size of 2000
# # #
# # #     Parameters
# # #     ----------
# # #     num_in_each :
# # #         param batch_size:
# # #     batch_size :
# # #
# # #
# # #     Returns
# # #     -------
# # #
# # #
# # #     """
# # #     break_into = np.ceil(np.array(num_in_each) / batch_size)
# # #     extract_inds = np.array([])
# # #     for k, elem in enumerate(break_into):
# # #         tmp1 = np.array(np.ones(np.int(elem)) * k)
# # #         extract_inds = np.concatenate((extract_inds, tmp1), axis=0)
# # #     return extract_inds
# # #
# # # def get_total_frame_count(h5_file_list):
# # #     """
# # #
# # #     Parameters
# # #     ----------
# # #     h5_file_list :
# # #
# # #
# # #     Returns
# # #     -------
# # #
# # #
# # #     """
# # #     total_frame_count = []
# # #     for H5_file in h5_file_list:
# # #         H5 = h5py.File(H5_file, 'r')
# # #         images = H5['images']
# # #         total_frame_count.append(images.shape[0])
# # #
# # #     return total_frame_count
# # #
# # #
# # # class ImageBatchGenerator_LSTM(keras.utils.Sequence):
# # #     """ """
# # #
# # #     def __init__(self, lstm_len, batch_size, h5_file_list, label_key = 'labels', IMG_SIZE = 96):
# # #         assert lstm_len%2 == 1, "number of images must be odd"
# # #         h5_file_list = utils.make_list(h5_file_list, suppress_warning=True)
# # #         num_frames_in_all_H5_files = get_total_frame_count(h5_file_list)
# # #         file_inds_for_H5_extraction = batch_size_file_ind_selector(
# # #             num_frames_in_all_H5_files, batch_size)
# # #         subtract_for_index = reset_to_first_frame_for_each_file_ind(
# # #             file_inds_for_H5_extraction)
# # #         # self.to_fit = to_fit #set to True to return XY and False to return X
# # #         self.label_key = label_key
# # #         self.batch_size = batch_size
# # #         self.H5_file_list = h5_file_list
# # #         self.num_frames_in_all_H5_files = num_frames_in_all_H5_files
# # #         self.file_inds_for_H5_extraction = file_inds_for_H5_extraction
# # #         self.subtract_for_index = subtract_for_index
# # #         self.IMG_SIZE = IMG_SIZE
# # #         self.lstm_len = lstm_len
# # #
# # #     def __getitem__(self, num_2_extract):
# # #         h = self.H5_file_list
# # #         i = self.file_inds_for_H5_extraction
# # #         H5_file = h[np.int(i[num_2_extract])]
# # #         num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
# # #         with h5py.File(H5_file, 'r') as h:
# # #
# # #
# # #             b = self.lstm_len//2
# # #             tot_len = h['images'].shape[0]
# # #             assert tot_len-b>self.batch_size, "reduce batch size to be less than total length of images minus floor(lstm_len) - 1, MAX->" + str(tot_len-b-1)
# # #             i1 = num_2_extract_mod * self.batch_size - b
# # #             i2 = num_2_extract_mod * self.batch_size + self.batch_size + b
# # #             edge_left_trigger = abs(min(i1, 0))
# # #             edge_right_trigger = abs(min(tot_len-i2, 0))
# # #             x = h['images'][max(i1, 0):min(i2, tot_len)]
# # #             if edge_left_trigger+edge_right_trigger>0: # in case of edge cases
# # #                 pad_shape = list(x.shape)
# # #                 pad_shape[0] = edge_left_trigger+edge_right_trigger
# # #                 pad = np.zeros(pad_shape).astype('float32')
# # #                 if edge_left_trigger>edge_right_trigger:
# # #                     x = np.concatenate((pad, x), axis = 0)
# # #                 else:
# # #                     x = np.concatenate((x, pad), axis = 0)
# # #             x = self.image_transform(x)
# # #
# # #             s = list(x.shape)
# # #             s.insert(1, self.lstm_len)
# # #             out = np.zeros(s).astype('float32')
# # #
# # #             for i in range(self.lstm_len):
# # #                 i1 = max(0, b-i)
# # #                 i2 = min(s[0], s[0]+b-i)
# # #                 i3 = max(0, i-b)
# # #                 i4 = min(s[0], s[0]+i-b)
# # #                 # print('take ', i3,' to ',  i4, ' and place in ', i1,' to ', i2)
# # #                 out[i1:i2, i, ...] = x[i3:i4, ...]
# # #             out = out[b:s[0]-b, ...]
# # #             i1 = num_2_extract_mod * self.batch_size
# # #             i2 = num_2_extract_mod * self.batch_size + self.batch_size
# # #
# # #             raw_Y = h[self.label_key][i1:i2]
# # #             return out, raw_Y
# # #
# # #     def __len__(self):
# # #         return len(self.file_inds_for_H5_extraction)
# # #
# # #     def getXandY(self, num_2_extract):
# # #         """
# # #
# # #         Parameters
# # #         ----------
# # #         num_2_extract :
# # #
# # #
# # #         Returns
# # #         -------
# # #
# # #         """
# # #         rgb_tensor, raw_Y = self.__getitem__(num_2_extract)
# # #         return rgb_tensor, raw_Y
# # #     def getXandY_NOT_LSTM_FORMAT(self, num_2_extract):
# # #
# # #         h = self.H5_file_list
# # #         i = self.file_inds_for_H5_extraction
# # #         H5_file = h[np.int(i[num_2_extract])]
# # #         num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
# # #         with h5py.File(H5_file, 'r') as h:
# # #
# # #
# # #             b = self.lstm_len//2
# # #             tot_len = h['images'].shape[0]
# # #             assert tot_len-b>self.batch_size, "reduce batch size to be less than total length of images minus floor(lstm_len) - 1, MAX->" + str(tot_len-b-1)
# # #             i1 = num_2_extract_mod * self.batch_size - b
# # #             i2 = num_2_extract_mod * self.batch_size + self.batch_size + b
# # #             edge_left_trigger = abs(min(i1, 0))
# # #             edge_right_trigger = abs(min(tot_len-i2, 0))
# # #             x = h['images'][max(i1, 0):min(i2, tot_len)]
# # #             if edge_left_trigger+edge_right_trigger>0: # in case of edge cases
# # #                 pad_shape = list(x.shape)
# # #                 pad_shape[0] = edge_left_trigger+edge_right_trigger
# # #                 pad = np.zeros(pad_shape).astype('float32')
# # #                 if edge_left_trigger>edge_right_trigger:
# # #                     x = np.concatenate((pad, x), axis = 0)
# # #                 else:
# # #                     x = np.concatenate((x, pad), axis = 0)
# # #             x = self.image_transform(x)
# # #             out = x
# # #             raw_Y = h[self.label_key][i1:i2]
# # #             return out, raw_Y
# # #     def image_transform(self, raw_X):
# # #         """input num_of_images x H x W, image input must be grayscale
# # #         MobileNetV2 requires certain image dimensions
# # #         We use N x 61 x 61 formated images
# # #         self.IMG_SIZE is a single number to change the images into, images must be square
# # #
# # #         Parameters
# # #         ----------
# # #         raw_X :
# # #
# # #
# # #         Returns
# # #         -------
# # #
# # #
# # #         """
# # #         if len(raw_X.shape) == 4 and raw_X.shape[3] == 3:
# # #             rgb_batch = copy.deepcopy(raw_X)
# # #         else:
# # #             rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
# # #         rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
# # #         rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2
# # #         rgb_tensor = tf.image.resize(rgb_tensor, (self.IMG_SIZE, self.IMG_SIZE))  # resizing
# # #         self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)
# # #         return rgb_tensor
# # #
# # #
# # # import datetime
# # #
# # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/small_h5s/3lag/small_test_3lag.h5'
# # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/regular/train_regular.h5'
# # # lstm_len = 5;batch_size = 1000; h5_file_list = [h5];
# # #
# # # G = ImageBatchGenerator_LSTM(lstm_len, batch_size, h5_file_list, label_key = 'labels', IMG_SIZE = 96)
# # # start = datetime.now()
# # # for k in range(G.__len__()):
# # #     x, y = G.__getitem__(k)
# # # print(datetime.now() - start)
# # #
# # # x, y = G.getXandY(1)
# # # x = (x+1)/2
# # # y = x[488]
# # # plt.figure()
# # # for i, k in enumerate(y):
# # #
# # #     print(all(k.flatten()==0))
# # #     plt.subplot(3,2, i+1)
# # #
# # #     # plt.figure()
# # #     plt.imshow(k)
# # #
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # # real_bool = image_tools.get_h5_key_and_concatenate(h5_file, '[0, 1]- (no touch, touch)')
# # # trial_nums_and_frame_nums = image_tools.get_h5_key_and_concatenate(h5_file, 'trial_nums_and_frame_nums')
# # # frame_nums = trial_nums_and_frame_nums[1, :].astype(int)
# # # in_range = image_tools.get_h5_key_and_concatenate(h5_file, 'in_range')
# # #
# # # real_bool[np.invert(in_range.astype(bool))] = -1
# # #
# # # kernel_size = 7
# # # pred_bool_temp = image_tools.get_h5_key_and_concatenate(h5_file, key_name)
# # # pred_bool_smoothed = foo_arg_max_and_smooth(pred_bool_temp, kernel_size, threshold, key_name, L_type_split_ind = 5)
# # # # pred_bool_smoothed = foo_arg_max_and_smooth(pred_bool_temp, kernel_size, threshold, L_type_split_ind = 5)
# # # pred_bool_smoothed[np.invert(in_range.astype(bool))] = -1
# # #
# # # r = real_bool
# # # p = pred_bool_smoothed
# # #
# # #
# # #
# # # a = analysis.error_analysis(r, p, frame_num_array=frame_nums)
# # #
# # #
# # #
# # #
# # # np.unique(a.coded_array)
# # #
# # # from whacc import analysis
# # # import numpy as np
# # # r = np.asarray([0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,0])
# # # p = np.asarray([0,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,1,1,0,1,0])
# # #
# # # # r = np.asarray([0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1])
# # # # p = np.asarray([0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,0])
# # #
# # # a = analysis.error_analysis(r, p)
# # # # ['miss', 'append', 'deduct', 'split', 'ghost']
# # # # [[1, 2], [4, 5], [11], [19], [25, 26]]
# # # from whacc import utils
# # # utils.get_class_info(a,end_prev_len = 4000)
# # # """
# # # onset graph
# # # find appends/deducts
# # #
# # # all_errors_sorted --> are the inds to each error in -->all_error_type_sorted
# # # """
# # #
# # # aet = np.asarray(a.all_error_type_sorted)
# # # cirt_error_inds = np.where(np.logical_and(aet != 'deduct',  aet != 'append'))[0]
# # # aet[cirt_error_inds]
# # # print(len(cirt_error_inds))
# # # plt.hist(aet[cirt_error_inds])
# # #
# # #
# # # append_var = np.where(np.asarray(a.all_error_type_sorted) == 'append')[0]
# # # deduct_var = np.where(np.asarray(a.all_error_type_sorted) == 'deduct')[0]
# # # sided_append_or_deduct = np.diff(np.asarray(a.which_side_test_sorted)).flatten()
# # #
# # # # onset_append_inds = append_var[sided_append_or_deduct[append_var]>0]
# # # # offset_append_inds = append_var[sided_append_or_deduct[append_var]<0]
# # # #
# # # # onset_deduct_inds = deduct_var[sided_append_or_deduct[deduct_var]>0]
# # # # offset_deduct_inds = deduct_var[sided_append_or_deduct[deduct_var]<0]
# # # #
# # # # onset_inds_real = utils.search_sequence_numpy(r, np.asarray([0,1]))+1
# # # # offset_inds_pred = utils.search_sequence_numpy(r, np.asarray([1,0]))
# # #
# # #
# # # # error_lengths = [len(k) for k in a.all_errors_sorted]
# # #
# # #
# # # # correct_onset_inds_sorted = np.intersect1d(a.onset_inds_real, a.onset_inds_pred)
# # # # correct_offset_inds_sorted = np.intersect1d(a.offset_inds_real, a.offset_inds_pred)
# # # #
# # # # onset_distance = np.concatenate((np.asarray(a.error_lengths_sorted)[a.onset_append_inds], np.asarray(a.error_lengths_sorted)[a.onset_deduct_inds]*-1))
# # # # offset_distance = np.concatenate((np.asarray(a.error_lengths_sorted)[a.offset_append_inds], np.asarray(a.error_lengths_sorted)[a.offset_deduct_inds]*-1))
# # # #
# # # # onset_distance = np.concatenate((onset_distance, np.zeros_like(correct_onset_inds_sorted)))
# # # # offset_distance = np.concatenate((offset_distance, np.zeros_like(correct_offset_inds_sorted)))
# # # #
# # #
# # #
# # # import matplotlib.pyplot as plt
# # # # plt.hist(a.onset_distance, bins = np.linspace(-10, 10))
# # #
# # # plt.figure()
# # # bins = np.arange(-7, 7)+.5
# # # plt.hist(np.clip(a.onset_distance, bins[0], bins[-1]), bins=bins)
# # # plt.xlabel('distance from human defined onset'); plt.ylabel('count')
# # #
# # # plt.figure()
# # # bins = np.arange(-1, 7)+.5
# # # tmp1 = plt.hist(np.clip(np.abs(a.onset_distance), bins[0], bins[-1]), bins=bins)
# # # # plt.figure()
# # # cum_dist = np.cumsum(tmp1[0]/np.sum(tmp1[0]))
# # # plt.xlabel('absolute distance from human defined onset')
# # # ax1 = plt.gca()
# # # ax2 = ax1.twinx()
# # # ax2.plot(bins[:-1]+.5, cum_dist, '-k')
# # # plt.ylim([0.5, 1])
# # # ax1.set_ylabel('count', color='b')
# # # ax2.set_ylabel('cumulative distribution of total number of onsets', color='k')
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # # from whacc import image_tools, utils
# # # import numpy as np
# # # import datetime
# # # import matplotlib.pyplot as plt
# # #
# # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/small_h5s/3lag/small_test_3lag.h5'
# # # imgs = image_tools.get_h5_key_and_concatenate(h5, 'images')
# # # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/regular/train_regular.h5'
# # # lstm_len = 5;batch_size = 100; h5_file_list = [h5];
# # #
# # # G = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, h5_file_list, label_key = 'labels', IMG_SIZE = 96)
# # # x, y = G.__getitem__(0)
# # # type(x[0][0][0][0][0])
# # #
# # #
# # # start = datetime.now()
# # # for k in range(G.__len__()):
# # #     x, y = G.__getitem__(k)
# # # print(datetime.now() - start)
# # #
# # # x, y = G.getXandY(1)
# # # x = (x+1)/2
# # # y = x[488]
# # # plt.figure()
# # # for i, k in enumerate(y):
# # #
# # #     print(all(k.flatten()==0))
# # #     plt.subplot(3,2, i+1)
# # #
# # #     # plt.figure()
# # #     plt.imshow(k)
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # # from whacc import image_tools, utils
# # # import numpy as np
# # # import datetime
# # # import matplotlib.pyplot as plt
# # #
# # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/small_h5s/3lag/small_test_3lag.h5'
# # # h5 = '/Users/phil/Desktop/content/ALL_RETRAIN_H5_data____temp/3lag/small_train_3lag.h5'
# # # imgs = image_tools.get_h5_key_and_concatenate(h5, 'images')
# # # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/regular/train_regular.h5'
# # # lstm_len = 5;batch_size = 100; h5_file_list = [h5];
# # #
# # # ImageBatchGenerator_LSTM2(lstm_len, batch_size, h5_file_list, label_key = 'labels', IMG_SIZE = 96)
# # # x, y = G.__getitem__(0)
# # # utils.np_stats(x)
# # #
# # # G = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, h5_file_list, label_key = 'labels', IMG_SIZE = 96)
# # # x, y = G.__getitem__(0)
# # # utils.np_stats(x)
# # #
# # # base_dir = '/Users/phil/Desktop/content/'
# # # cnt = 0
# # # for h5 in utils.lister_it(utils.get_h5s(base_dir), keep_strings='____temp'):  # only go through the non converted h5s
# # #     if cnt == 0:
# # #         cnt = 1
# # #         # utils.print_h5_keys(h5)
# # #         h5_out = ''.join(h5.split('____temp'))
# # #         print(h5, h5_out)
# # #         convert_h5_to_LSTM_h5(h5, h5_out, IMG_SIZE=96)
# # #         utils.copy_over_all_non_image_keys(h5, h5_out)
# # #
# # # x = image_tools.get_h5_key_and_concatenate(h5_out, 'images')
# # # utils.np_stats(x)
# # #
# # #
# # #
# # #
# # # al_gen = image_tools.ImageBatchGenerator_simple(100, '/Users/phil/Desktop/content/ALL_RETRAIN_H5_data/3lag/small_train_3lag.h5', label_key='labels', IMG_SIZE = 96)
# # # x, y = al_gen.__getitem__(0)
# # # utils.np_stats(x)
# # #
# # # np.min(x), np.max(x), np.mean(x), x.shape
# # # np.min(x2), np.max(x2), np.mean(x2), x2.shape
# # #
# # # x2 = x[40, 3, ...]
# # # # x2 = np.asarray(x2)
# # # plt.imshow(x2)
# # #
# # #
# # #
# # # def npstats(in_arr):
# # #     print('min', np.min(in_arr))
# # #     print('max', np.max(in_arr))
# # #     print('mean', np.mean(in_arr))
# # #     print('shape', in_arr.shape)
# # #     print('len of unique', len(np.unique(in_arr)))
# # #
# # #
# # #
# # #
# # #
# # # def image_transform(self, raw_X):
# # #     rgb_tensor = tf.cast(raw_X, tf.float32)  # convert to tf tensor with float32 dtypes
# # #     rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2
# # #     return rgb_tensor
# # #
# # #
# # #
# # #
# # # from whacc import image_tools, utils
# # # import h5py
# # #
# # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/3lag/val_3lag.h5'
# # # with h5py.File(h5, 'r') as h:
# # #     utils.np_stats(h['images'][:100])
# # #     print(h['images'][:].shape)
# # #
# # #
# # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/3lag/val_3lag.h5'
# # # with h5py.File(h5, 'r') as h:
# # #     utils.np_stats(h['images'][:100])
# # #     print(h['images'][:].shape)
# # #
# # #
# #
# #
# # #######
# # from whacc import image_tools
# #
# # """
# # set label_index_to_lstm_len to shift arrays
# #
# # set the number to be whatever you want so can set to nan or -9999 to then remove those edge trials automatically
# # """
# #
# # from whacc.image_tools import *
# # import pdb
# #
# # class ImageBatchGenerator_feature_array(keras.utils.Sequence):
# #     """ """
# #
# #     def __init__(self, lstm_len, batch_size, h5_file_list, label_key='labels', feature_len=2048,
# #                  label_index_to_lstm_len=None, edge_value=-999):
# #         assert lstm_len % 2 == 1, "number of images must be odd"
# #         if label_index_to_lstm_len is None:
# #             label_index_to_lstm_len = lstm_len // 2  # in the middle
# #         h5_file_list = utils.make_list(h5_file_list, suppress_warning=True)
# #         num_frames_in_all_H5_files = get_total_frame_count(h5_file_list)
# #         file_inds_for_H5_extraction = batch_size_file_ind_selector(
# #             num_frames_in_all_H5_files, batch_size)
# #         subtract_for_index = reset_to_first_frame_for_each_file_ind(
# #             file_inds_for_H5_extraction)
# #         # self.to_fit = to_fit #set to True to return XY and False to return X
# #         self.label_key = label_key
# #         self.batch_size = batch_size
# #         self.H5_file_list = h5_file_list
# #         self.num_frames_in_all_H5_files = num_frames_in_all_H5_files
# #         self.file_inds_for_H5_extraction = file_inds_for_H5_extraction
# #         self.subtract_for_index = subtract_for_index
# #         self.label_index_to_lstm_len = label_index_to_lstm_len
# #         self.lstm_len = lstm_len
# #         self.feature_len = feature_len
# #         self.edge_value = edge_value
# #
# #         self.get_frame_edges()
# #
# #     def get_frame_edges(self):
# #         self.all_edges_list = []
# #         b = self.lstm_len // 2
# #
# #         s = [b * 2, self.lstm_len, self.feature_len]
# #         for H5_file in self.H5_file_list:
# #             with h5py.File(H5_file,
# #                            'r') as h:  # 0(0, 1) 1(0) 3998(-1) 3999(-2, -1) ...  4000(0, 1) 4001(0) 7998(-1) 7999(-2, -1) #0,0     0,1     1,0     3998,-1     3999,-2     3999,-1
# #                 full_edges_mask = np.ones(s)
# #                 edge_ind = np.flip(np.arange(1, b + 1))
# #                 for i in np.arange(1, b + 1):
# #                     full_edges_mask[i - 1, :edge_ind[i - 1], ...] = np.zeros_like(
# #                         full_edges_mask[i - 1, :edge_ind[i - 1], ...])
# #                     full_edges_mask[-i, -edge_ind[i - 1]:, ...] = np.zeros_like(
# #                         full_edges_mask[-i, -edge_ind[i - 1]:, ...])
# #                 all_edges = []
# #                 for i1, i2 in utils.loop_segments(h['frame_nums']):  # 0, 1, 3998, 3999 ; 4000, 4001, 7998, 7999; ...
# #                     edges = (np.asarray([[i1], [i2 - b]]) + np.arange(0, b).T).flatten()
# #                     all_edges.append(edges)
# #                 all_edges = np.asarray(all_edges)
# #             self.all_edges_list.append(all_edges)
# #             # pdb.set_trace()
# #             full_edges_mask = full_edges_mask.astype(int)
# #             self.full_edges_mask = full_edges_mask == 0
# #
# #
# #     def __getitem__(self, num_2_extract):
# #         h = self.H5_file_list
# #         i = self.file_inds_for_H5_extraction
# #         all_edges = self.all_edges_list[np.int(i[num_2_extract])]
# #         H5_file = h[np.int(i[num_2_extract])]
# #         num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
# #         with h5py.File(H5_file, 'r') as h:
# #             b = self.lstm_len // 2
# #             tot_len = h['images'].shape[0]
# #             assert tot_len - b > self.batch_size, "reduce batch size to be less than total length of images minus floor(lstm_len) - 1, MAX->" + str(
# #                 tot_len - b - 1)
# #             i1 = num_2_extract_mod * self.batch_size - b
# #             i2 = num_2_extract_mod * self.batch_size + self.batch_size + b
# #             edge_left_trigger = abs(min(i1, 0))
# #             edge_right_trigger = min(abs(min(tot_len - i2, 0)), b)
# #             x = h['images'][max(i1, 0):min(i2, tot_len)]
# #             if edge_left_trigger + edge_right_trigger > 0:  # in case of edge cases
# #                 pad_shape = list(x.shape)
# #                 pad_shape[0] = edge_left_trigger + edge_right_trigger
# #                 pad = np.zeros(pad_shape).astype('float32')
# #                 if edge_left_trigger > edge_right_trigger:
# #                     x = np.concatenate((pad, x), axis=0)
# #                 else:
# #                     x = np.concatenate((x, pad), axis=0)
# #             x = self.image_transform(x)
# #             s = list(x.shape)
# #             s.insert(1, self.lstm_len)
# #             out = np.zeros(s).astype('float32')  # before was uint8
# #
# #             for i in range(self.lstm_len):
# #                 i1 = max(0, b - i)
# #                 i2 = min(s[0], s[0] + b - i)
# #                 i3 = max(0, i - b)
# #                 i4 = min(s[0], s[0] + i - b)
# #                 # print('take ', i3,' to ',  i4, ' and place in ', i1,' to ', i2)
# #                 out[i1:i2, i, ...] = x[i3:i4, ...]
# #             out = out[b:s[0] - b, ...]
# #             i1 = num_2_extract_mod * self.batch_size
# #             i2 = num_2_extract_mod * self.batch_size + self.batch_size
# #             raw_Y = h[self.label_key][i1:i2]
# #             # black out edges from frame to frame
# #             adjust_these_edge_frames = np.intersect1d(all_edges.flatten(), np.arange(i1, i2))
# #             for atef in adjust_these_edge_frames:
# #                 # mask_ind = np.where(atef == all_edges)[1][0]
# #                 # out[atef] = out[atef] * (self.full_edges_mask[mask_ind]
# #
# #                 mask_ind = np.where(atef == all_edges)[1][0]
# #                 mask_ = self.full_edges_mask[mask_ind]
# #                 # pdb.set_trace()
# #                 out[atef - i1][mask_] = self.edge_value
# #             return out, raw_Y
# #
# #     # gray mask(set array to -1 not 0 ),DONE
# #     # doesnt fill the edges as expected, DONE
# #     # outputs format 0-255 not -1 to 1 DONE this was just custom code not from image_tools
# #     # need to test this with uneven frames
# #     def __len__(self):
# #         return len(self.file_inds_for_H5_extraction)
# #
# #     def getXandY(self, num_2_extract):
# #         """
# #
# #         Parameters
# #         ----------
# #         num_2_extract :
# #
# #
# #         Returns
# #         -------
# #
# #         """
# #         rgb_tensor, raw_Y = self.__getitem__(num_2_extract)
# #         return rgb_tensor, raw_Y
# #
# #     def image_transform(self, raw_X):
# #         """input num_of_images x H x W, image input must be grayscale
# #         MobileNetV2 requires certain image dimensions
# #         We use N x 61 x 61 formated images
# #         self.IMG_SIZE is a single number to change the images into, images must be square
# #
# #         Parameters
# #         ----------
# #         raw_X :
# #
# #
# #         Returns
# #         -------
# #
# #
# #         """
# #         rgb_batch = copy.deepcopy(raw_X)
# #         # if len(raw_X.shape) == 4 and raw_X.shape[3] == 3:
# #         #     rgb_batch = copy.deepcopy(raw_X)
# #         # else:
# #         #     rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
# #         rgb_tensor = rgb_batch
# #         # rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
# #         # rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2 #commented before
# #         # rgb_tensor = tf.image.resize(rgb_tensor, (self.feature_len))  # resizing
# #         # rgb_tensor = tf.cast(rgb_tensor, np.uint8)# un commented before
# #         self.IMG_SHAPE = (self.feature_len)
# #         return rgb_tensor
# #
# #
# # h5 = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing_features_data/feature_data/regular_80_border/DATA_FULL/3lag/holy_test_set_10_percent_3lag.h5'
# #
# # a = ImageBatchGenerator_feature_array(7, 100, h5, )
# # x, y = a.__getitem__(0)
# #
# # utils.np_stats(x), utils.np_stats(y)
# # # import os
# # # import glob
# # #
# # # import cv2
# # # import numpy as np
# # # import time
# # # import re
# # # import h5py
# # # import matplotlib.pyplot as plt
# # # from functools import partial
# # # from tqdm import tqdm
# # # from PIL import Image
# # # from whacc.image_tools import h5_iterative_creator
# # # from sklearn.preprocessing import normalize
# # # from whacc import utils, image_tools, analysis
# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # # import copy
# # # from scipy.signal import medfilt
# # #
# # #
# # # def track_h5(template_image, h5_file, match_method='cv2.TM_CCOEFF', ind_list=None):
# # #     with h5py.File(h5_file, 'r') as h5:
# # #         if isinstance(template_image, int):  # if termplate is an ind to the images in the h5
# # #             template_image = h5['images'][template_image, ...]
# # #         elif len(template_image.shape) == 2:
# # #             template_image = np.repeat(template_image[:, :, None], 3, axis=2)
# # #
# # #         if ind_list is None:
# # #             ind_list = range(len(h5['labels'][:]))
# # #         # width and height of img_stacks will be that of template (61x61)
# # #         max_match_val = []
# # #         try:
# # #             method_ = eval(match_method)
# # #         except:
# # #             method_ = match_method
# # #         max_match_val = []
# # #         for frame_i in tqdm(ind_list):
# # #             img = h5['images'][frame_i, ...]
# # #             # Apply template Matching
# # #             if isinstance(method_, str):
# # #                 print('NOOOOOOOOOOOOOOO')
# # #                 if method_ == 'calc_diff':
# # #                     max_val = np.sum(np.abs(img.flatten() - template_image.flatten()))
# # #                 elif method_ == 'mse':
# # #                     max_val = np.mean((img.flatten() - template_image.flatten()) ** 2)
# # #             else:
# # #                 res = cv2.matchTemplate(img, template_image, method_)
# # #                 min_val, max_val, min_loc, top_left = cv2.minMaxLoc(res)
# # #             max_match_val.append(max_val)
# # #             # top_left = np.flip(np.asarray(top_left))
# # #     return max_match_val, template_image
# # #
# # #
# # # x = '/Users/phil/Downloads/test_pole_tracker/AH0667x170317.h5'
# # # utils.print_h5_keys(x)
# # # max_val_stack = image_tools.get_h5_key_and_concatenate(x, 'max_val_stack')
# # # locations_x_y = image_tools.get_h5_key_and_concatenate(x, 'locations_x_y')
# # # trial_nums_and_frame_nums = image_tools.get_h5_key_and_concatenate(x, 'trial_nums_and_frame_nums')
# # # template_img = image_tools.get_h5_key_and_concatenate(x, 'template_img')
# # # frame_nums = trial_nums_and_frame_nums[1, :].astype(int)
# # # trial_nums = trial_nums_and_frame_nums[0, :].astype(int)
# # # asdfasdfasdf
# # # method = 'TM_CCOEFF_NORMED'
# # # ind_list = None
# # # max_match_val_new, template_image_out = track_h5(template_img, x, match_method='cv2.' + method, ind_list=ind_list)
# # #
# # # method = 'calc_diff'
# # # ind_list = None
# # # template_img = 2000
# # # max_match_val_new, template_image_out = track_h5(template_img, x, match_method=method, ind_list=ind_list)
# # #
# # # method = 'mse'
# # # ind_list = None
# # # template_img = 2000
# # # max_match_val_new, template_image_out = track_h5(template_img, x, match_method=method, ind_list=ind_list)
# # #
# # # for k1, k2 in utils.loop_segments(frame_nums):
# # #     plt.plot(max_match_val_new[k1:k2], linewidth=.3)
# # # plt.legend(trial_nums)
# # #
# # # match_list = ['TM_SQDIFF_NORMED', 'TM_CCORR_NORMED', 'TM_CCOEFF_NORMED', 'TM_SQDIFF', 'TM_CCORR', 'TM_CCOEFF']
# # #
# # # h5_file = '/Users/phil/Downloads/test_pole_tracker/AH0667x170317.h5'
# # # meth_dict = dict()
# # # meth_dict['h5_file'] = h5_file
# # # ind_list = None
# # # for template_image_ind in [0, 2000]:
# # #     for method in match_list:
# # #         max_match_val_new, template_image = track_h5(template_image_ind, h5_file, match_method='cv2.' + method,
# # #                                                      ind_list=ind_list)
# # #         meth_dict['ind_' + str(template_image_ind) + '_' + method] = max_match_val_new
# # #
# # # for method in match_list:
# # #     max_match_val_new, template_image = track_h5(template_img, h5_file, match_method='cv2.' + method, ind_list=ind_list)
# # #     meth_dict['ind_template_img_' + method] = max_match_val_new
# # #
# # # fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=False)
# # # ax_list = fig.axes
# # # cnt = -1
# # # for k in meth_dict:
# # #     if 'h5_file' not in k and 'NORM' in k:
# # #         cnt += 1
# # #         if len(ax_list) == cnt:
# # #             cnt = 0
# # #             fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=False)
# # #             ax_list = fig.axes
# # #         ax1 = ax_list[cnt]
# # #         ax1.set_title(k)
# # #         # plt.title(k)
# # #         for k1, k2 in utils.loop_segments(frame_nums):
# # #             try:
# # #                 x = np.asarray(meth_dict[k][k1:k2])
# # #                 # ax1.plot(x-x[0],linewidth=.3, alpha = 1)
# # #                 ax1.plot(x, linewidth=.3, alpha=1)
# # #             except:
# # #                 break
# # # plt.legend(trial_nums)
# # #
# # # # plt.imshow(image_tools.get_h5_key_and_concatenate(h5_file, 'images'))
# # #
# # # a = analysis.pole_plot('/Users/phil/Downloads/test_pole_tracker/AH0667x170317.h5')
# # #
# # # a.current_frame = 0
# # # a.plot_it()
# # #
# # # a.current_frame = 1000
# # # a.plot_it()
# # # """"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # #
# # # h5_file = '/Users/phil/Downloads/test_pole_tracker/AH0667x170317.h5'
# # # method = 'cv2.TM_CCOEFF'
# # # method = 'cv2.TM_CCOEFF_NORMED'
# # # method = 'cv2.TM_SQDIFF_NORMED'
# # # ind_list = None
# # # template_image_ind = 2000  # know this is a good starting point with no whiskers in it
# # # ls = np.asarray(utils.loop_segments(frame_nums, returnaslist=True))
# # # all_maxes = []
# # # trial_inds = range(len(frame_nums))
# # # self_references_frame_compares = np.zeros(np.sum(frame_nums))
# # # max_match_all = []
# # # max_match_all2 = []
# # # trial_ind_all = []
# # # template_img_all = []
# # # template_image_ind_all = []
# # # for k in range(len(frame_nums)):
# # #     template_image_ind_all.append(template_image_ind)
# # #     max_match, template_img = track_h5(int(template_image_ind), h5_file, match_method=method, ind_list=ind_list)
# # #     template_img_all.append(template_img)
# # #     max_match_all.append(np.asarray(max_match))
# # #     max_match_all2.append(np.asarray(max_match))
# # #     trial_ind = np.where(template_image_ind < np.cumsum(frame_nums))[0][0]
# # #     trial_ind_all.append(trial_ind)
# # #     self_references_frame_compares[ls[0, trial_ind]:ls[1, trial_ind]] = max_match[ls[0, trial_ind]:ls[1, trial_ind]]
# # #     if k == len(frame_nums)-1:
# # #         break
# # #     for kt in trial_ind_all:
# # #         for kk in max_match_all:
# # #             kk[ls[0, kt]:ls[1, kt]] = np.nan
# # #             kk[ls[0, kt]:ls[1, kt]] = np.nan
# # #     _val = -99999999999
# # #     _ind = -99999999999
# # #     for kk in max_match_all:
# # #         tmp1 = np.nanmax(kk)
# # #         tmp2 = np.nanargmax(kk)
# # #         if tmp1 > _val:
# # #             _val = copy.deepcopy(tmp1)
# # #             _ind = copy.deepcopy(tmp2)
# # #     # template_image_ind = copy.deepcopy(_ind)
# # #     template_image_ind = template_image_ind+4000
# # #
# # # kernel_size = 1
# # # for k1, k2 in utils.loop_segments(frame_nums):
# # #     plt.plot(medfilt(self_references_frame_compares[k1:k2], kernel_size=kernel_size), linewidth = 0.3)
# # # plt.legend(range(len(frame_nums)))
# # # plt.title(method)
# # #
# # #
# # # # pred_bool_smoothed = medfilt(copy.deepcopy(pred_bool_temp), kernel_size=kernel_size)
# # #
# # # fig, ax = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=False)
# # # ax_list = fig.axes
# # # for i, k in enumerate(ax_list):
# # #     k.imshow(template_img_all[i])
# # #     k.set_title(template_image_ind_all[i])
# # #
# # #
# # #
# # # x = np.mean(np.asarray(max_match_all2), axis = 0)
# # # kernel_size = 1
# # # for k1, k2 in utils.loop_segments(frame_nums):
# # #     plt.plot(medfilt(x[k1:k2], kernel_size=kernel_size), linewidth = 0.3)
# # # plt.legend(range(len(frame_nums)))
# # # plt.title(method)
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # #
# # # h5_file = '/Users/phil/Dropbox/HIRES_LAB/curation_for_auto_curator/Data/Jon/AH0667/170317/AH0667x170317.h5'
# # # trial_nums_and_frame_nums = image_tools.get_h5_key_and_concatenate(h5_file, 'trial_nums_and_frame_nums')
# # # frame_nums = trial_nums_and_frame_nums[1, :].astype(int)
# # #
# # # method = 'cv2.TM_CCORR_NORMED'
# # # frame_to_compare = 2000
# # # testing_frames_start = 1250
# # # testing_frames_len = 50
# # #
# # # method = 'cv2.TM_CCOEFF'#console regular# best
# # # frame_to_compare = 1
# # # testing_frames_start = 1250
# # # testing_frames_len = 50
# # #
# # # # method = 'cv2.TM_CCOEFF' #console (1)
# # # # frame_to_compare = 2000
# # # # testing_frames_start = 1250
# # # # testing_frames_len = 50
# # #
# # # ind_list = None
# # # all_tests = []
# # # for ktrial, _ in utils.loop_segments(frame_nums):
# # #     template_image_ind = frame_to_compare+ktrial
# # #     max_match_all = []
# # #     for k1, k2 in utils.loop_segments(frame_nums):
# # #         ind_list = np.arange(testing_frames_start, testing_frames_start+testing_frames_len) + k1
# # #         max_match, template_img = track_h5(int(template_image_ind), h5_file, match_method=method, ind_list=ind_list)
# # #         max_match = np.asarray(max_match).astype(float)
# # #         max_match_all.append(max_match-max_match[0])
# # #     all_tests.append(np.asarray(max_match_all).flatten())
# # #
# # # all_var = []
# # # for i, k in enumerate(all_tests):
# # #     addto = (10**6)*i*2
# # #     k = k[(k>np.quantile(k,0.1)) & (k<np.quantile(k,0.9))]
# # #     plt.plot(k+addto, '.', markersize = 0.3)
# # #     # plt.plot(k+addto, '-k', linewidth = 0.05)
# # #     all_var.append(np.var(k))
# # # plt.legend(np.argsort(all_var))
# # #
# # # plt.figure()
# # # for i, k in enumerate(all_tests):
# # #     addto = (10**6)*i*2
# # #     plt.plot(k+addto, '.', markersize = 0.3)
# # #
# # #
# # #
# # #
# # #
# # # k1, k2 = utils.loop_segments(frame_nums, returnaslist=True)
# # # template_image_ind = frame_to_compare+k1[np.argmin(all_var)]
# # #
# # # ind_list = None
# # # max_match, template_img = track_h5(int(template_image_ind), h5_file, match_method=method, ind_list=ind_list)
# # #
# # # locations_x_y = image_tools.get_h5_key_and_concatenate(h5_file, 'locations_x_y')
# # # tmp1 = np.argsort(locations_x_y[:, 0][2000::4000])
# # #
# # # k1, k2 = utils.loop_segments(frame_nums, returnaslist = True)
# # #
# # # for i, k in enumerate(tmp1):
# # #     addto = i*10**6
# # #     plt.plot(np.asarray(max_match[k1[k]:k2[k]])+addto, linewidth=0.3)
# # #
# # #
# # # for i, k in enumerate(tmp1):
# # #     addto = i*10**-3
# # #     plt.plot(np.asarray(max_match[k1[k]:k2[k]])+addto, linewidth=0.3)
# # #
# # #
# # #
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # # k1, k2 = utils.loop_segments(frame_nums, returnaslist=True)
# # # all_var_inds = np.argsort(all_var)
# # # all_max = []
# # # for ii in range(4):
# # #     template_image_ind = frame_to_compare+k1[all_var_inds[ii]]
# # #     ind_list = None
# # #     max_match, template_img = track_h5(int(template_image_ind), h5_file, match_method=method, ind_list=ind_list)
# # #     all_max.append(max_match)
# # #
# # #
# # # max_match_mean = np.nanmean(np.asarray(all_max), axis = 0)
# # # tmp1 = np.argsort(locations_x_y[:, 0][2000::4000])
# # # for i, k in enumerate(tmp1):
# # #     addto = i*10**6
# # #     plt.plot(np.asarray(max_match_mean[k1[k]:k2[k]])+addto, linewidth=0.3)
# # #
# # #
# # # x = np.asarray(all_max)
# # # for i, k in enumerate(tmp1):
# # #     addto = i*10**6
# # #     x1  = x[0][k1[k]:k2[k]]+addto
# # #     x2  = x[1][k1[k]:k2[k]]+addto
# # #     plt.plot(x1-x2, linewidth=0.3)
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # #
# # #
# # # from IPython.utils import io
# # #
# # # vit = dict()
# # # vit['all_acc_before_no_pole_mask'] = []
# # # vit['all_acc_after_no_pole_mask'] = []
# # # vit['all_acc_before'] = []
# # # vit['all_acc_after'] = []
# # # vit['h5_img_file'] = []
# # # vit['h5_img_file_full_dir'] = []
# # #
# # # vit['m_name'] = m_names
# # # vit['L_key']= label_key
# # # vit['vm_name'] = vit_m_names
# # # vit['h5_img_file_full_dir']= to_pred_h5s
# # # for k in vit['h5_img_file_full_dir']:
# # #   vit['h5_img_file'].append(os.path.basename(k))
# # #
# # # for h5_img_file in to_pred_h5s:
# # #   in_range = image_tools.get_h5_key_and_concatenate([h5_img_file], 'in_range')
# # #   tmp1 = []
# # #   tmp2 = []
# # #   tmp3 = []
# # #   tmp4 = []
# # #   for iii, (vm_name, m_name, L_key) in enumerate(tzip(vit_m_names, m_names, label_key)):
# # #     pred_m_raw = image_tools.get_h5_key_and_concatenate([h5_img_file], key_name=m_name)
# # #     pred_v = image_tools.get_h5_key_and_concatenate([h5_img_file], key_name=vm_name)
# # #     real = image_tools.get_h5_key_and_concatenate([h5_img_file], key_name=L_key)
# # #     if pred_m_raw.shape[1] ==1:
# # #       pred_m = ((pred_m_raw>0.5)*1).flatten()
# # #     else:
# # #       pred_m = np.argmax(pred_m_raw, axis = 1)# turn into integers instead of percentages
# # #
# # #     # get everything back to binary (if possible)
# # #     with io.capture_output() as captured:#prevents crazy printing
# # #
# # #       pred_m_bool = utils.convert_labels_back_to_binary(pred_m, L_key)
# # #       real_bool = utils.convert_labels_back_to_binary(real, L_key)
# # #       pred_v_bool = utils.convert_labels_back_to_binary(pred_v, L_key)
# # #     if real_bool is None: # convert labels will return None if it cant convert
# # #     #it back to the normal format. i.e. only onset or only offsets...
# # #       tmp1.append(0)
# # #       tmp2.append(0)
# # #       tmp3.append(0)
# # #       tmp4.append(0)
# # #     else:
# # #       tmp1.append(np.mean(real_bool == pred_m_bool))
# # #       tmp2.append(np.mean(real_bool == pred_v_bool))
# # #       tmp3.append(np.mean(real_bool*in_range == pred_m_bool*in_range))
# # #       tmp4.append(np.mean(real_bool*in_range == pred_v_bool*in_range))
# # #
# # #   vit['all_acc_before_no_pole_mask'].append(tmp1)
# # #   vit['all_acc_after_no_pole_mask'].append(tmp2)
# # #   vit['all_acc_before'].append(tmp3)
# # #   vit['all_acc_after'].append(tmp4)
# # #   # vit['m_name'] = list(dict.fromkeys(vit['m_name']))
# # #   # vit['L_key'] = list(dict.fromkeys(vit['L_key']))
# # #   # vit['h5_img_file_full_dir'].append(h5_img_file)
# # #
# # #
# # # vit2 = copy.deepcopy(vit)
# # # ############################################################################################################
# # # ############################################################################################################
# # # ############################################################################################################
# # # ############################################################################################################
# # # ############################################################################################################
# # # ############################################################################################################
# # # ############################################################################################################
# # # ############################################################################################################
# # #
# # # # def grab_frames(x, lstm_len, end_frame_num):
# # # #     # if
# # #
# # # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/regular/train_regular.h5'
# # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/small_h5s/3lag/small_test_3lag.h5'
# # # utils.print_h5_keys(h5)
# # # # (None, 5, 96, 96, 3)
# # # lstm_len = 5
# # # assert lstm_len%2 == 1
# # # with h5py.File(h5, 'r') as h:
# # #     print(len(h['images']))
# # #     batch_size = 20
# # #     print(h['images'][0:batch_size].shape)
# # #     x = h['images'][20:20 + batch_size]
# # #     x2 = x[:, None, ...]
# # #     print(x2.shape)
# # #     print(x2[:-5, ...].shape)
# # #     # print(x[4:-1, ...].shape)
# # #     x3 = np.stack(( x[:-4, ...],  x[1:-3, ...], x[2:-2, ...], x[3:-1, ...], x[4:, ...]), axis = 1)
# # #     print(x3.shape)
# # #     print(x.shape)
# # #     # print(x[4:-1, ...].shape)
# # #     y = x3[0]
# # #     y2 = y
# # #
# # #     plt.figure()
# # #     for i, k in enumerate(y):
# # #         plt.subplot(3,2, i+1)
# # #         plt.imshow(k)
# # #
# # #
# # #
# # # # finished extractor........
# # # lstm_len = 5
# # # assert lstm_len%2 == 1, "number of images must be odd"
# # # batch_size = 10
# # # chunk_num = 24
# # #
# # # lstm_len//2
# # #
# # # with h5py.File(h5, 'r') as h:
# # #     b = lstm_len//2
# # #     tot_len = h['images'].shape[0]
# # #     i1 = chunk_num * batch_size - b
# # #     i2 = chunk_num * batch_size + batch_size + b
# # #     edge_left_trigger = abs(min(i1, 0))
# # #     edge_right_trigger = abs(min(tot_len-i2, 0))
# # #     x = h['images'][max(i1, 0):min(i2, tot_len)]
# # #     print(x.shape)
# # #     if edge_left_trigger+edge_right_trigger>0: # in case of edge cases
# # #         pad_shape = list(x.shape); pad_shape[0] = edge_left_trigger+edge_right_trigger
# # #         pad = np.zeros(pad_shape).astype('uint8')
# # #         if edge_left_trigger>edge_right_trigger:
# # #             x = np.concatenate((pad, x), axis = 0)
# # #         else:
# # #             x = np.concatenate((x, pad), axis = 0)
# # #
# # #     s = list(x.shape)
# # #     s.insert(1, lstm_len)
# # #     out = np.zeros(s).astype('uint8')
# # #
# # #
# # #     for i in range(lstm_len):
# # #         i1 = max(0, b-i)
# # #         i2 = min(s[0], s[0]+b-i)
# # #         i3 = max(0, i-b)
# # #         i4 = min(s[0], s[0]+i-b)
# # #         print('take ', i3,' to ',  i4, ' and place in ', i1,' to ', i2)
# # #         out[i1:i2, i, ...] = x[i3:i4, ...]
# # #     out = out[b:s[0]-b, ...]
# # #
# # #     y = out[-1]
# # #     print(out.shape)
# # #     plt.figure()
# # #     for i, k in enumerate(y):
# # #         plt.subplot(3,2, i+1)
# # #         # plt.figure()
# # #         plt.imshow(k)
# # #
# # #
# # #
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # #
# # #
# # # # finished extractor........
# # #
# # #
# # #
# # # lstm_len = 5
# # #
# # # batch_size = 10
# # # chunk_num = 24
# # # with h5py.File(h5, 'r') as h:
# # #     b = lstm_len//2
# # #     tot_len = h['images'].shape[0]
# # #     i1 = chunk_num * batch_size - b
# # #     i2 = chunk_num * batch_size + batch_size + b
# # #     edge_left_trigger = abs(min(i1, 0))
# # #     edge_right_trigger = abs(min(tot_len-i2, 0))
# # #     x = h['images'][max(i1, 0):min(i2, tot_len)]
# # #     print(x.shape)
# # #     if edge_left_trigger+edge_right_trigger>0: # in case of edge cases
# # #         pad_shape = list(x.shape); pad_shape[0] = edge_left_trigger+edge_right_trigger
# # #         pad = np.zeros(pad_shape).astype('uint8')
# # #         if edge_left_trigger>edge_right_trigger:
# # #             x = np.concatenate((pad, x), axis = 0)
# # #         else:
# # #             x = np.concatenate((x, pad), axis = 0)
# # #
# # #     s = list(x.shape)
# # #     s.insert(1, lstm_len)
# # #     out = np.zeros(s).astype('uint8')
# # #
# # #
# # #     for i in range(lstm_len):
# # #         i1 = max(0, b-i)
# # #         i2 = min(s[0], s[0]+b-i)
# # #         i3 = max(0, i-b)
# # #         i4 = min(s[0], s[0]+i-b)
# # #         print('take ', i3,' to ',  i4, ' and place in ', i1,' to ', i2)
# # #         out[i1:i2, i, ...] = x[i3:i4, ...]
# # #     out = out[b:s[0]-b, ...]
# # #
# # #     y = out[-1]
# # #     print(out.shape)
# # #     plt.figure()
# # #     for i, k in enumerate(y):
# # #         plt.subplot(3,2, i+1)
# # #         # plt.figure()
# # #         plt.imshow(k)
# # #
# # #
# # #
# # #
# # # import tensorflow as tf
# # # from tensorflow import keras
# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # # import h5py
# # # import copy
# # # import time
# # # import os
# # # from whacc import utils
# # # import pdb
# # #
# # # def reset_to_first_frame_for_each_file_ind(file_inds_for_H5_extraction):
# # #     """reset_to_first_frame_for_each_file_ind - uses the output of batch_size_file_ind_selector
# # #     to determine when to reset the index for each individual H5 file. using the above example
# # #     the out put would be [0, 0, 2, 2, 2, 5, 5], each would be subtracted from the indexing to
# # #     set the position of the index to 0 for each new H5 file.
# # #
# # #     Parameters
# # #     ----------
# # #     file_inds_for_H5_extraction :
# # #
# # #
# # #     Returns
# # #     -------
# # #
# # #
# # #     """
# # #     subtract_for_index = []
# # #     for k, elem in enumerate(file_inds_for_H5_extraction):
# # #         tmp1 = np.diff(file_inds_for_H5_extraction)
# # #         tmp1 = np.where(tmp1 != 0)
# # #         tmp1 = np.append(-1, tmp1[0]) + 1
# # #         subtract_for_index.append(tmp1[np.int(file_inds_for_H5_extraction[k])])
# # #     return subtract_for_index
# # #
# # #
# # # def batch_size_file_ind_selector(num_in_each, batch_size):
# # #     """batch_size_file_ind_selector - needed for ImageBatchGenerator to know which H5 file index
# # #     to use depending on the iteration number used in __getitem__ in the generator.
# # #     this all depends on the variable batch size.
# # #
# # #     Example: the output of the following...
# # #     batch_size_file_ind_selector([4000, 4001, 3999], [2000])
# # #     would be [0, 0, 1, 1, 1, 2, 2] which means that there are 2 chunks in the first
# # #     H5 file, 3 in the second and 2 in the third based on chunk size of 2000
# # #
# # #     Parameters
# # #     ----------
# # #     num_in_each :
# # #         param batch_size:
# # #     batch_size :
# # #
# # #
# # #     Returns
# # #     -------
# # #
# # #
# # #     """
# # #     break_into = np.ceil(np.array(num_in_each) / batch_size)
# # #     extract_inds = np.array([])
# # #     for k, elem in enumerate(break_into):
# # #         tmp1 = np.array(np.ones(np.int(elem)) * k)
# # #         extract_inds = np.concatenate((extract_inds, tmp1), axis=0)
# # #     return extract_inds
# # #
# # # def get_total_frame_count(h5_file_list):
# # #     """
# # #
# # #     Parameters
# # #     ----------
# # #     h5_file_list :
# # #
# # #
# # #     Returns
# # #     -------
# # #
# # #
# # #     """
# # #     total_frame_count = []
# # #     for H5_file in h5_file_list:
# # #         H5 = h5py.File(H5_file, 'r')
# # #         images = H5['images']
# # #         total_frame_count.append(images.shape[0])
# # #
# # #     return total_frame_count
# # #
# # #
# # # class ImageBatchGenerator_LSTM(keras.utils.Sequence):
# # #     """ """
# # #
# # #     def __init__(self, lstm_len, batch_size, h5_file_list, label_key = 'labels', IMG_SIZE = 96):
# # #         assert lstm_len%2 == 1, "number of images must be odd"
# # #         h5_file_list = utils.make_list(h5_file_list, suppress_warning=True)
# # #         num_frames_in_all_H5_files = get_total_frame_count(h5_file_list)
# # #         file_inds_for_H5_extraction = batch_size_file_ind_selector(
# # #             num_frames_in_all_H5_files, batch_size)
# # #         subtract_for_index = reset_to_first_frame_for_each_file_ind(
# # #             file_inds_for_H5_extraction)
# # #         # self.to_fit = to_fit #set to True to return XY and False to return X
# # #         self.label_key = label_key
# # #         self.batch_size = batch_size
# # #         self.H5_file_list = h5_file_list
# # #         self.num_frames_in_all_H5_files = num_frames_in_all_H5_files
# # #         self.file_inds_for_H5_extraction = file_inds_for_H5_extraction
# # #         self.subtract_for_index = subtract_for_index
# # #         self.IMG_SIZE = IMG_SIZE
# # #         self.lstm_len = lstm_len
# # #
# # #     def __getitem__(self, num_2_extract):
# # #         h = self.H5_file_list
# # #         i = self.file_inds_for_H5_extraction
# # #         H5_file = h[np.int(i[num_2_extract])]
# # #         num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
# # #         with h5py.File(H5_file, 'r') as h:
# # #
# # #
# # #             b = self.lstm_len//2
# # #             tot_len = h['images'].shape[0]
# # #             assert tot_len-b>self.batch_size, "reduce batch size to be less than total length of images minus floor(lstm_len) - 1, MAX->" + str(tot_len-b-1)
# # #             i1 = num_2_extract_mod * self.batch_size - b
# # #             i2 = num_2_extract_mod * self.batch_size + self.batch_size + b
# # #             edge_left_trigger = abs(min(i1, 0))
# # #             edge_right_trigger = abs(min(tot_len-i2, 0))
# # #             x = h['images'][max(i1, 0):min(i2, tot_len)]
# # #             if edge_left_trigger+edge_right_trigger>0: # in case of edge cases
# # #                 pad_shape = list(x.shape)
# # #                 pad_shape[0] = edge_left_trigger+edge_right_trigger
# # #                 pad = np.zeros(pad_shape).astype('float32')
# # #                 if edge_left_trigger>edge_right_trigger:
# # #                     x = np.concatenate((pad, x), axis = 0)
# # #                 else:
# # #                     x = np.concatenate((x, pad), axis = 0)
# # #             x = self.image_transform(x)
# # #
# # #             s = list(x.shape)
# # #             s.insert(1, self.lstm_len)
# # #             out = np.zeros(s).astype('float32')
# # #
# # #             for i in range(self.lstm_len):
# # #                 i1 = max(0, b-i)
# # #                 i2 = min(s[0], s[0]+b-i)
# # #                 i3 = max(0, i-b)
# # #                 i4 = min(s[0], s[0]+i-b)
# # #                 # print('take ', i3,' to ',  i4, ' and place in ', i1,' to ', i2)
# # #                 out[i1:i2, i, ...] = x[i3:i4, ...]
# # #             out = out[b:s[0]-b, ...]
# # #             i1 = num_2_extract_mod * self.batch_size
# # #             i2 = num_2_extract_mod * self.batch_size + self.batch_size
# # #
# # #             raw_Y = h[self.label_key][i1:i2]
# # #             return out, raw_Y
# # #
# # #     def __len__(self):
# # #         return len(self.file_inds_for_H5_extraction)
# # #
# # #     def getXandY(self, num_2_extract):
# # #         """
# # #
# # #         Parameters
# # #         ----------
# # #         num_2_extract :
# # #
# # #
# # #         Returns
# # #         -------
# # #
# # #         """
# # #         rgb_tensor, raw_Y = self.__getitem__(num_2_extract)
# # #         return rgb_tensor, raw_Y
# # #     def getXandY_NOT_LSTM_FORMAT(self, num_2_extract):
# # #
# # #         h = self.H5_file_list
# # #         i = self.file_inds_for_H5_extraction
# # #         H5_file = h[np.int(i[num_2_extract])]
# # #         num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
# # #         with h5py.File(H5_file, 'r') as h:
# # #
# # #
# # #             b = self.lstm_len//2
# # #             tot_len = h['images'].shape[0]
# # #             assert tot_len-b>self.batch_size, "reduce batch size to be less than total length of images minus floor(lstm_len) - 1, MAX->" + str(tot_len-b-1)
# # #             i1 = num_2_extract_mod * self.batch_size - b
# # #             i2 = num_2_extract_mod * self.batch_size + self.batch_size + b
# # #             edge_left_trigger = abs(min(i1, 0))
# # #             edge_right_trigger = abs(min(tot_len-i2, 0))
# # #             x = h['images'][max(i1, 0):min(i2, tot_len)]
# # #             if edge_left_trigger+edge_right_trigger>0: # in case of edge cases
# # #                 pad_shape = list(x.shape)
# # #                 pad_shape[0] = edge_left_trigger+edge_right_trigger
# # #                 pad = np.zeros(pad_shape).astype('float32')
# # #                 if edge_left_trigger>edge_right_trigger:
# # #                     x = np.concatenate((pad, x), axis = 0)
# # #                 else:
# # #                     x = np.concatenate((x, pad), axis = 0)
# # #             x = self.image_transform(x)
# # #             out = x
# # #             raw_Y = h[self.label_key][i1:i2]
# # #             return out, raw_Y
# # #     def image_transform(self, raw_X):
# # #         """input num_of_images x H x W, image input must be grayscale
# # #         MobileNetV2 requires certain image dimensions
# # #         We use N x 61 x 61 formated images
# # #         self.IMG_SIZE is a single number to change the images into, images must be square
# # #
# # #         Parameters
# # #         ----------
# # #         raw_X :
# # #
# # #
# # #         Returns
# # #         -------
# # #
# # #
# # #         """
# # #         if len(raw_X.shape) == 4 and raw_X.shape[3] == 3:
# # #             rgb_batch = copy.deepcopy(raw_X)
# # #         else:
# # #             rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
# # #         rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
# # #         rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2
# # #         rgb_tensor = tf.image.resize(rgb_tensor, (self.IMG_SIZE, self.IMG_SIZE))  # resizing
# # #         self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)
# # #         return rgb_tensor
# # #
# # #
# # # import datetime
# # #
# # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/small_h5s/3lag/small_test_3lag.h5'
# # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/regular/train_regular.h5'
# # # lstm_len = 5;batch_size = 1000; h5_file_list = [h5];
# # #
# # # G = ImageBatchGenerator_LSTM(lstm_len, batch_size, h5_file_list, label_key = 'labels', IMG_SIZE = 96)
# # # start = datetime.now()
# # # for k in range(G.__len__()):
# # #     x, y = G.__getitem__(k)
# # # print(datetime.now() - start)
# # #
# # # x, y = G.getXandY(1)
# # # x = (x+1)/2
# # # y = x[488]
# # # plt.figure()
# # # for i, k in enumerate(y):
# # #
# # #     print(all(k.flatten()==0))
# # #     plt.subplot(3,2, i+1)
# # #
# # #     # plt.figure()
# # #     plt.imshow(k)
# # #
# # # """$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# # # real_bool = image_tools.get_h5_key_and_concatenate(h5_file, '[0, 1]- (no touch, touch)')
# # # trial_nums_and_frame_nums = image_tools.get_h5_key_and_concatenate(h5_file, 'trial_nums_and_frame_nums')
# # # frame_nums = trial_nums_and_frame_nums[1, :].astype(int)
# # # in_range = image_tools.get_h5_key_and_concatenate(h5_file, 'in_range')
# # #
# # # real_bool[np.invert(in_range.astype(bool))] = -1
# # #
# # # kernel_size = 7
# # # pred_bool_temp = image_tools.get_h5_key_and_concatenate(h5_file, key_name)
# # # pred_bool_smoothed = foo_arg_max_and_smooth(pred_bool_temp, kernel_size, threshold, key_name, L_type_split_ind = 5)
# # # # pred_bool_smoothed = foo_arg_max_and_smooth(pred_bool_temp, kernel_size, threshold, L_type_split_ind = 5)
# # # pred_bool_smoothed[np.invert(in_range.astype(bool))] = -1
# # #
# # # r = real_bool
# # # p = pred_bool_smoothed
# # #
# # #
# # #
# # # a = analysis.error_analysis(r, p, frame_num_array=frame_nums)
# # #
# # #
# # #
# # #
# # # np.unique(a.coded_array)
# # #
# # # from whacc import analysis
# # # import numpy as np
# # # r = np.asarray([0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,0])
# # # p = np.asarray([0,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,1,1,0,1,0])
# # #
# # # # r = np.asarray([0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1])
# # # # p = np.asarray([0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,0])
# # #
# # # a = analysis.error_analysis(r, p)
# # # # ['miss', 'append', 'deduct', 'split', 'ghost']
# # # # [[1, 2], [4, 5], [11], [19], [25, 26]]
# # # from whacc import utils
# # # utils.get_class_info(a,end_prev_len = 4000)
# # # """
# # # onset graph
# # # find appends/deducts
# # #
# # # all_errors_sorted --> are the inds to each error in -->all_error_type_sorted
# # # """
# # #
# # # aet = np.asarray(a.all_error_type_sorted)
# # # cirt_error_inds = np.where(np.logical_and(aet != 'deduct',  aet != 'append'))[0]
# # # aet[cirt_error_inds]
# # # print(len(cirt_error_inds))
# # # plt.hist(aet[cirt_error_inds])
# # #
# # #
# # # append_var = np.where(np.asarray(a.all_error_type_sorted) == 'append')[0]
# # # deduct_var = np.where(np.asarray(a.all_error_type_sorted) == 'deduct')[0]
# # # sided_append_or_deduct = np.diff(np.asarray(a.which_side_test_sorted)).flatten()
# # #
# # # # onset_append_inds = append_var[sided_append_or_deduct[append_var]>0]
# # # # offset_append_inds = append_var[sided_append_or_deduct[append_var]<0]
# # # #
# # # # onset_deduct_inds = deduct_var[sided_append_or_deduct[deduct_var]>0]
# # # # offset_deduct_inds = deduct_var[sided_append_or_deduct[deduct_var]<0]
# # # #
# # # # onset_inds_real = utils.search_sequence_numpy(r, np.asarray([0,1]))+1
# # # # offset_inds_pred = utils.search_sequence_numpy(r, np.asarray([1,0]))
# # #
# # #
# # # # error_lengths = [len(k) for k in a.all_errors_sorted]
# # #
# # #
# # # # correct_onset_inds_sorted = np.intersect1d(a.onset_inds_real, a.onset_inds_pred)
# # # # correct_offset_inds_sorted = np.intersect1d(a.offset_inds_real, a.offset_inds_pred)
# # # #
# # # # onset_distance = np.concatenate((np.asarray(a.error_lengths_sorted)[a.onset_append_inds], np.asarray(a.error_lengths_sorted)[a.onset_deduct_inds]*-1))
# # # # offset_distance = np.concatenate((np.asarray(a.error_lengths_sorted)[a.offset_append_inds], np.asarray(a.error_lengths_sorted)[a.offset_deduct_inds]*-1))
# # # #
# # # # onset_distance = np.concatenate((onset_distance, np.zeros_like(correct_onset_inds_sorted)))
# # # # offset_distance = np.concatenate((offset_distance, np.zeros_like(correct_offset_inds_sorted)))
# # # #
# # #
# # #
# # # import matplotlib.pyplot as plt
# # # # plt.hist(a.onset_distance, bins = np.linspace(-10, 10))
# # #
# # # plt.figure()
# # # bins = np.arange(-7, 7)+.5
# # # plt.hist(np.clip(a.onset_distance, bins[0], bins[-1]), bins=bins)
# # # plt.xlabel('distance from human defined onset'); plt.ylabel('count')
# # #
# # # plt.figure()
# # # bins = np.arange(-1, 7)+.5
# # # tmp1 = plt.hist(np.clip(np.abs(a.onset_distance), bins[0], bins[-1]), bins=bins)
# # # # plt.figure()
# # # cum_dist = np.cumsum(tmp1[0]/np.sum(tmp1[0]))
# # # plt.xlabel('absolute distance from human defined onset')
# # # ax1 = plt.gca()
# # # ax2 = ax1.twinx()
# # # ax2.plot(bins[:-1]+.5, cum_dist, '-k')
# # # plt.ylim([0.5, 1])
# # # ax1.set_ylabel('count', color='b')
# # # ax2.set_ylabel('cumulative distribution of total number of onsets', color='k')
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # # from whacc import image_tools, utils
# # # import numpy as np
# # # import datetime
# # # import matplotlib.pyplot as plt
# # #
# # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/small_h5s/3lag/small_test_3lag.h5'
# # # imgs = image_tools.get_h5_key_and_concatenate(h5, 'images')
# # # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/regular/train_regular.h5'
# # # lstm_len = 5;batch_size = 100; h5_file_list = [h5];
# # #
# # # G = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, h5_file_list, label_key = 'labels', IMG_SIZE = 96)
# # # x, y = G.__getitem__(0)
# # # type(x[0][0][0][0][0])
# # #
# # #
# # # start = datetime.now()
# # # for k in range(G.__len__()):
# # #     x, y = G.__getitem__(k)
# # # print(datetime.now() - start)
# # #
# # # x, y = G.getXandY(1)
# # # x = (x+1)/2
# # # y = x[488]
# # # plt.figure()
# # # for i, k in enumerate(y):
# # #
# # #     print(all(k.flatten()==0))
# # #     plt.subplot(3,2, i+1)
# # #
# # #     # plt.figure()
# # #     plt.imshow(k)
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # # from whacc import image_tools, utils
# # # import numpy as np
# # # import datetime
# # # import matplotlib.pyplot as plt
# # #
# # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/test_data/small_h5s/3lag/small_test_3lag.h5'
# # # h5 = '/Users/phil/Desktop/content/ALL_RETRAIN_H5_data____temp/3lag/small_train_3lag.h5'
# # # imgs = image_tools.get_h5_key_and_concatenate(h5, 'images')
# # # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/regular/train_regular.h5'
# # # lstm_len = 5;batch_size = 100; h5_file_list = [h5];
# # #
# # # ImageBatchGenerator_LSTM2(lstm_len, batch_size, h5_file_list, label_key = 'labels', IMG_SIZE = 96)
# # # x, y = G.__getitem__(0)
# # # utils.np_stats(x)
# # #
# # # G = image_tools.ImageBatchGenerator_LSTM(lstm_len, batch_size, h5_file_list, label_key = 'labels', IMG_SIZE = 96)
# # # x, y = G.__getitem__(0)
# # # utils.np_stats(x)
# # #
# # # base_dir = '/Users/phil/Desktop/content/'
# # # cnt = 0
# # # for h5 in utils.lister_it(utils.get_h5s(base_dir), keep_strings='____temp'):  # only go through the non converted h5s
# # #     if cnt == 0:
# # #         cnt = 1
# # #         # utils.print_h5_keys(h5)
# # #         h5_out = ''.join(h5.split('____temp'))
# # #         print(h5, h5_out)
# # #         convert_h5_to_LSTM_h5(h5, h5_out, IMG_SIZE=96)
# # #         utils.copy_over_all_non_image_keys(h5, h5_out)
# # #
# # # x = image_tools.get_h5_key_and_concatenate(h5_out, 'images')
# # # utils.np_stats(x)
# # #
# # #
# # #
# # #
# # # al_gen = image_tools.ImageBatchGenerator_simple(100, '/Users/phil/Desktop/content/ALL_RETRAIN_H5_data/3lag/small_train_3lag.h5', label_key='labels', IMG_SIZE = 96)
# # # x, y = al_gen.__getitem__(0)
# # # utils.np_stats(x)
# # #
# # # np.min(x), np.max(x), np.mean(x), x.shape
# # # np.min(x2), np.max(x2), np.mean(x2), x2.shape
# # #
# # # x2 = x[40, 3, ...]
# # # # x2 = np.asarray(x2)
# # # plt.imshow(x2)
# # #
# # #
# # #
# # # def npstats(in_arr):
# # #     print('min', np.min(in_arr))
# # #     print('max', np.max(in_arr))
# # #     print('mean', np.mean(in_arr))
# # #     print('shape', in_arr.shape)
# # #     print('len of unique', len(np.unique(in_arr)))
# # #
# # #
# # #
# # #
# # #
# # # def image_transform(self, raw_X):
# # #     rgb_tensor = tf.cast(raw_X, tf.float32)  # convert to tf tensor with float32 dtypes
# # #     rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2
# # #     return rgb_tensor
# # #
# # #
# # #
# # #
# # # from whacc import image_tools, utils
# # # import h5py
# # #
# # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border/data/3lag/val_3lag.h5'
# # # with h5py.File(h5, 'r') as h:
# # #     utils.np_stats(h['images'][:100])
# # #     print(h['images'][:].shape)
# # #
# # #
# # # h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/3lag/val_3lag.h5'
# # # with h5py.File(h5, 'r') as h:
# # #     utils.np_stats(h['images'][:100])
# # #     print(h['images'][:].shape)
# # #
# # #
# #
# #
# # #######
# # from whacc import image_tools
# #       # out = tf.cast(out, tf.float32)
#
# """
# set label_index_to_lstm_len to shift arrays
#
# set the number to be whatever you want so can set to nan or -9999 to then remove those edge trials automatically
# """
#
# from whacc.image_tools import *
# import pdb
#
#
# class ImageBatchGenerator_feature_array(keras.utils.Sequence):
#     """ """
#
#     def __init__(self, lstm_len, batch_size, h5_file_list, label_key='labels', feature_len=2048,
#                  label_index_to_lstm_len=None, edge_value=-1):
#         assert lstm_len % 2 == 1, "number of images must be odd"
#         if label_index_to_lstm_len is None:
#             label_index_to_lstm_len = lstm_len // 2  # in the middle
#         h5_file_list = utils.make_list(h5_file_list, suppress_warning=True)
#         num_frames_in_all_H5_files = get_total_frame_count(h5_file_list)
#         file_inds_for_H5_extraction = batch_size_file_ind_selector(
#             num_frames_in_all_H5_files, batch_size)
#         subtract_for_index = reset_to_first_frame_for_each_file_ind(
#             file_inds_for_H5_extraction)
#         # self.to_fit = to_fit #set to True to return XY and False to return X
#         self.label_key = label_key
#         self.batch_size = batch_size
#         self.H5_file_list = h5_file_list
#         self.num_frames_in_all_H5_files = num_frames_in_all_H5_files
#         self.file_inds_for_H5_extraction = file_inds_for_H5_extraction
#         self.subtract_for_index = subtract_for_index
#         self.label_index_to_lstm_len = label_index_to_lstm_len
#         self.lstm_len = lstm_len
#         self.feature_len = feature_len
#         self.edge_value = edge_value
#
#         self.get_frame_edges()
#         # self.full_edges_mask = self.full_edges_mask - (self.lstm_len // 2 - self.label_index_to_lstm_len)
#
#     def __getitem__(self, num_2_extract):
#         h = self.H5_file_list
#         i = self.file_inds_for_H5_extraction
#         all_edges = self.all_edges_list[np.int(i[num_2_extract])]
#         H5_file = h[np.int(i[num_2_extract])]
#         num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
#         with h5py.File(H5_file, 'r') as h:
#             b = self.lstm_len // 2
#             tot_len = h['images'].shape[0]
#             assert tot_len - b > self.batch_size, "reduce batch size to be less than total length of images minus floor(lstm_len) - 1, MAX->" + str(
#                 tot_len - b - 1)
#             i1 = num_2_extract_mod * self.batch_size - b
#             i2 = num_2_extract_mod * self.batch_size + self.batch_size + b
#             edge_left_trigger = abs(min(i1, 0))
#             edge_right_trigger = min(abs(min(tot_len - i2, 0)), b)
#             x = h['images'][max(i1, 0):min(i2, tot_len)]
#             if edge_left_trigger + edge_right_trigger > 0:  # in case of edge cases
#                 pad_shape = list(x.shape)
#                 pad_shape[0] = edge_left_trigger + edge_right_trigger
#                 pad = np.zeros(pad_shape).astype('float32')
#                 if edge_left_trigger > edge_right_trigger:
#                     x = np.concatenate((pad, x), axis=0)
#                 else:
#                     x = np.concatenate((x, pad), axis=0)
#             # x = self.image_transform(x)
#
#             s = list(x.shape)
#             s.insert(1, self.lstm_len)
#             out = np.zeros(s).astype('float32')  # before was uint8
#             # out = tf.cast(out, tf.float32)
#             Z = self.label_index_to_lstm_len - self.lstm_len // 2
#             # Z = 0
#             # pdb.set_trace()
#
#             for i in range(self.lstm_len):
#                 i = i + Z
#                 i1 = max(0, b - i)
#                 i2 = min(s[0], s[0] + b - i)
#                 i3 = max(0, i - b)
#                 i4 = min(s[0], s[0] + i - b)
#                 # print('take ', i3,' to ',  i4, ' and place in ', i1,' to ', i2)       lstm -->5 batch size --> 20
#                 print(x[i3:i4, ...].shape)
#                 print(out[i1:i2, i, ...][:, 0])
#                 # asdf
#                 out[i1:i2, i, ...] = x[i3:i4, ...]  # out --> (24, 5, 3) input ot out --> (22, 3)
#
#             out = out[b:s[0] - b, ...]
#             i1 = num_2_extract_mod * self.batch_size
#             i2 = num_2_extract_mod * self.batch_size + self.batch_size
#             raw_Y = h[self.label_key][i1:i2]
#             # black out edges from frame to frame
#             adjust_these_edge_frames = np.intersect1d(all_edges.flatten(), np.arange(i1, i2))
#             # pdb.set_trace()
#             for atef in adjust_these_edge_frames:
#                 # mask_ind = np.where(atef == all_edges)[1][0]
#                 # out[atef] = out[atef] * (self.full_edges_mask[mask_ind]
#
#                 mask_ind = np.where(atef == all_edges)[1][0]
#                 mask_ = self.full_edges_mask[mask_ind]
#                 print(mask_, atef)
#                 # pdb.set_trace()
#                 mask_ = mask_ == 1
#                 out[atef - i1][mask_] = self.edge_value
#
#             return out, raw_Y
#
#     # gray mask(set array to -1 not 0 ),DONE
#     # doesnt fill the edges as expected, DONE
#     # outputs format 0-255 not -1 to 1 DONE this was just custom code not from image_tools
#     # need to test this with uneven frames
#     def __len__(self):
#         return len(self.file_inds_for_H5_extraction)
#
#     def getXandY(self, num_2_extract):
#         """
#
#         Parameters
#         ----------
#         num_2_extract :
#
#
#         Returns
#         -------
#
#         """
#         rgb_tensor, raw_Y = self.__getitem__(num_2_extract)
#         return rgb_tensor, raw_Y
#
#     def image_transform(self, raw_X):
#         """input num_of_images x H x W, image input must be grayscale
#         MobileNetV2 requires certain image dimensions
#         We use N x 61 x 61 formated images
#         self.IMG_SIZE is a single number to change the images into, images must be square
#
#         Parameters
#         ----------
#         raw_X :
#
#
#         Returns
#         -------
#
#
#         """
#         rgb_batch = copy.deepcopy(raw_X)
#         # if len(raw_X.shape) == 4 and raw_X.shape[3] == 3:
#         #     rgb_batch = copy.deepcopy(raw_X)
#         # else:
#         #     rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
#         rgb_tensor = rgb_batch
#         # rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
#         # rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2 #commented before
#         # rgb_tensor = tf.image.resize(rgb_tensor, (self.feature_len))  # resizing
#         # rgb_tensor = tf.cast(rgb_tensor, np.uint8)# un commented before
#         self.IMG_SHAPE = (self.feature_len)
#         return rgb_tensor
#
#     def get_frame_edges(self):
#         self.all_edges_list = []
#         b = self.lstm_len // 2
#
#         s = [b * 2, self.lstm_len, self.feature_len]
#         for H5_file in self.H5_file_list:
#             with h5py.File(H5_file,
#                            'r') as h:  # 0(0, 1) 1(0) 3998(-1) 3999(-2, -1) ...  4000(0, 1) 4001(0) 7998(-1) 7999(-2, -1) #0,0     0,1     1,0     3998,-1     3999,-2     3999,-1
#                 full_edges_mask = np.ones(s)
#                 shift_by = 0
#                 # edge_ind = np.flip(np.arange(1, b + 1))-shift_by
#                 # edge_ind2 = np.arange(1, b + 1)+shift_by
#                 tmp1 = np.arange(1, self.lstm_len)
#                 front_edge = tmp1[:self.label_index_to_lstm_len]
#                 back_edge = tmp1[:self.lstm_len - self.label_index_to_lstm_len - 1]
#                 # pdb.set_trace()
#
#                 edge_ind = np.flip(front_edge)
#                 for i in front_edge:
#                     print(i - 1, ':', edge_ind[i - 1])
#                     print(full_edges_mask[i - 1, :edge_ind[i - 1], ...].shape)
#                     print('\n')
#                     full_edges_mask[i - 1, :edge_ind[i - 1], ...] = np.zeros_like(
#                         full_edges_mask[i - 1, :edge_ind[i - 1], ...])
#
#                 edge_ind = np.flip(back_edge)
#                 for i in back_edge:
#                     print(-i, -edge_ind[i - 1], ':')
#                     print(full_edges_mask[-i, -edge_ind[i - 1]:, ...].shape)
#                     print('\n')
#                     full_edges_mask[-i, -edge_ind[i - 1]:, ...] = np.zeros_like(
#                         full_edges_mask[-i, -edge_ind[i - 1]:, ...])
#
#                 # pdb.set_trace()
#                 all_edges = []
#                 for i1, i2 in utils.loop_segments(h['frame_nums']):  # 0, 1, 3998, 3999 ; 4000, 4001, 7998, 7999; ...
#                     edges = (np.asarray([[i1], [i2 - b]]) + np.arange(0, b).T).flatten()
#                     all_edges.append(edges)
#                 all_edges = np.asarray(all_edges)
#             self.all_edges_list.append(all_edges)
#             # pdb.set_trace()
#             full_edges_mask = full_edges_mask.astype(int)
#             self.full_edges_mask = full_edges_mask == 0
#
#             # self.full_edges_mask = self.full_edges_mask - (self.lstm_len // 2 - self.label_index_to_lstm_len)
#
#
# h5 = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing_features_data/feature_data/regular_80_border/DATA_FULL/3lag/holy_test_set_10_percent_3lag.h5'
# h5 = '/Users/phil/Desktop/temp.h5'
# a = ImageBatchGenerator_feature_array(5, 20, h5, label_key='labels', feature_len=3,
#                                       label_index_to_lstm_len=2 - 1, edge_value=-1)
# x, y = a.__getitem__(0)
# utils.np_stats(x)
# s = x.shape;
# x2 = np.reshape(x, (s[0], s[1] * s[2]))
# plt.figure()
# plt.imshow(x2)
#
# a = ImageBatchGenerator_feature_array(5, 20, h5, label_key='labels', feature_len=3,
#                                       label_index_to_lstm_len=2 - 0, edge_value=-1)
# x, y = a.__getitem__(0)
# utils.np_stats(x)
# s = x.shape;
# x2 = np.reshape(x, (s[0], s[1] * s[2]))
# plt.figure()
# plt.imshow(x2)
#
# asdfsadf
# import h5py
#
# d1 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# d2 = [.5, .5, .5, 5, .5, .5, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#       0, 0]
# d3 = [.2, .2, .2, .3, .4, .8, .6, .2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#       0, 0, 0]
# data = np.asarray([d1, d2, d3]).T  #
# plt.imshow(data)
# plt.ylabel('time');
# plt.xlabel('features')
# frame_nums = np.asarray([10, 10, 10, 8])
# with h5py.File('/Users/phil/Desktop/temp.h5', 'w') as h:
#     h.create_dataset('images', data=data)
#     h.create_dataset('labels', data=d1)
#     h.create_dataset('frame_nums', data=frame_nums)
#
# utils.np_stats(x), utils.np_stats(y)
#
# x, y = a.__getitem__(0)
# print(x[0, :, :10])
# x, y = a.__getitem__(1)
# print(x[0, :, :10])
#
# x, y = a.__getitem__(a.__len__() - 1)
# print(x[0, :, :10])
# x, y = a.__getitem__(a.__len__() - 2)
# print(x[0, :, :10])
# x, y = a.__getitem__(a.__len__() - 3)
# print(x[0, :, :10])
# x, y = a.__getitem__(a.__len__() - 4)
# print(x[0, :, :10])
#
# import matplotlib.pyplot as plt
#
# x2 = (x + 1) / 2
# plt.imshow(x2[0, :, :20])
#
# x2 = (x + 1) / 2
# plt.imshow(x2[:, 0, :20])
# plt.figure()
# plt.imshow(x2[:, 1, :20])
# plt.figure()
# plt.imshow(x2[:, 2, :20])
# plt.figure()
# plt.imshow(x2[:, 3, :20])
#
# x2 = (x + 1) / 2
# x2 = np.moveaxis(x2, (0, 1, 2), (1, 2, 0))
#
# x2 = (x + 1) / 2
#
# x2 = (x + 1) / 2
# x2 = np.reshape(x2, (100, 7 * 2048))
# fig, ax = plt.subplots(nrows=1, ncols=10, sharex=True, sharey=False)
# for i, sp in enumerate(fig.axes):
#     sp.imshow(x2)
#
# plt.imshow(x2[:, :400])
# plt.figure()
# plt.plot(y)
#
# plt.figure()
# plt.plot(np.mean(x2, axis=1))
#
# x2 = (x + 1) / 2
# x2 = np.moveaxis(x2, (0, 1, 2), (1, 2, 0))
# fig, ax = plt.subplots(nrows=1, ncols=30, sharex=True, sharey=False)
# for i, sp in enumerate(fig.axes):
#     sp.imshow(x2[i])
#
# safasdf
#
# import matplotlib.pyplot as plt
#
# x2 = (x + 1) / 2
# plt.imshow(x2[:, 0, :20])
# plt.figure()
# plt.imshow(x2[:, 1, :20])
# plt.figure()
# plt.imshow(x2[:, 2, :20])
#
# x2 = (x + 1) / 2
# x2 = np.moveaxis(x2, (0, 1, 2), (1, 2, 0))
#
# x2 = (x + 1) / 2
#
# x2 = (x + 1) / 2
# x2 = np.reshape(x2, (100, 7 * 2048))
# fig, ax = plt.subplots(nrows=1, ncols=10, sharex=True, sharey=False)
# for i, sp in enumerate(fig.axes):
#     sp.imshow(x2)
#
# plt.imshow(x2[:, :400])
# plt.figure()
# plt.plot(y)
#
# plt.figure()
# plt.plot(np.mean(x2, axis=1))
#
# x2 = (x + 1) / 2
# x2 = np.moveaxis(x2, (0, 1, 2), (1, 2, 0))
# fig, ax = plt.subplots(nrows=1, ncols=30, sharex=True, sharey=False)
# for i, sp in enumerate(fig.axes):
#     sp.imshow(x2[i])
#
# #####
#
#
# from whacc import utils, image_tools
# import matplotlib.pyplot as plt
# import numpy as np
# import os
#
# h5 = '/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/AH0000x000000_using_pole_tracker.h5'
# utils.print_h5_keys(h5)
#
# base_dir_all = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing_features_data/feature_data/DATA_FULL/'
# alt_labels_h5s = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED/'
#
# # base_dir_all = r'G:\My Drive\colab_data2\model_testing_features_data\feature_data\DATA_FULL'
#
# model_save_name_full 'FULL MODEL NAME TO SAVE IN H5 HERE'
# all_h5s = utils.get_h5s(base_dir_all)
# alt_labels_h5s = utils.get_h5s(alt_labels_h5s)
# for h5, alt_labels_h5 in zip(all_h5s, alt_labels_h5s):
#     bn1 = os.path.basename(h5)
#     bn2 = os.path.basename(alt_labels_h5)
#     assert bn1[:16] == bn2[:16], 'files dont match'
#     base_dir = os.path.dirname(h5) + os.sep + 'transformed_data'
#     df_tmp_x, tmp_y = make_data_chunks(h5, base_dir)
#     df_tmp_x = concat_numpy_memmory_save(df_tmp_x, base_dir)
#     pred = bst.predict(df_tmp_x)
#
#     with h5py.File(alt_labels_h5, 'r+') as hf:
#         try:
#             hf.create_dataset(model_save_name_full, data=pred)
#         except:
#             del hf[model_save_name_full]
#             time.sleep(.1)
#             hf.create_dataset(model_save_name_full, data=pred)
#
#
#
# h5 = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing/all_data/final_predictions/ALT_LABELS_FINAL_PRED/AH0407_160613_JC1003_AAAC_ALT_LABELS.h5'
# a = utils.print_h5_keys(h5, return_list = True)
# utils.lister_it(a, 'model_5____')
#
# pred = image_tools.get_h5_key_and_concatenate(h5, 'model_5____3lag__regular_labels__MODEL_FULL_TRAIN_rollSTD_3_7_rollMEAN_3_7_shift_neg2_to_2_val_with_real_val_PM_JC_LIGHT_GBM')
# # trial_nums_and_frame_nums = image_tools.get_h5_key_and_concatenate(h5, 'trial_nums_and_frame_nums')
# # max_val_stack = image_tools.get_h5_key_and_concatenate(h5, 'max_val_stack')
# # trial_nums_and_frame_nums.astype(int)
# # # in_range
# # labels = image_tools.get_h5_key_and_concatenate(h5, 'labels')
# # labels
# #
# # max_val_stack = max_val_stack - np.min(max_val_stack)
# # max_val_stack = max_val_stack/np.max(max_val_stack)
# #
# # plt.figure(figsize = (10, 5))
# # for i1, i2 in utils.loop_segments(trial_nums_and_frame_nums[1, :].astype(int)):
# #     x = max_val_stack[i1:i2] - np.min(max_val_stack[i1:i2])
# #     x = x/np.max(x)
# #     plt.plot(x)
# #     # plt.plot(labels[i1:i2])
# #     if i1>10*4000:
# #         asdf
#
# """##################################################################################################################"""
# """##################################################################################################################"""
# """##################################################################################################################"""
# """##################################################################################################################"""
# """##################################################################################################################"""
# """##################################################################################################################"""
#
#
#
# # import matplotlib.pyplot as plt
# # from whacc.feature_maker import feature_maker
# # h5_in = '/Users/phil/Desktop/AH0407_160613_JC1003_AAAC_3lag.h5'
# #
# # FM = feature_maker(h5_in, frame_num_ind=2, delete_if_exists = True) #set frame_num_ind to an int so that we can quickly look at the transformation
# # data, data_name = FM.shift(5, save_it=False) # shift it 5 to the right, fills in with nan as needed, look at just the 5th frame num set (5th video)
# # # to see how it looks
# # print('the below name will be used to save the data in the the h5 when calling functions with save_it=True')
# # print(data_name)
# # FM.shift(5, save_it=True) # now lets save it
# #
# # data, data_name_rolling_mean_100 = FM.rolling(100, 'mean', save_it=True)  #lets do the rolling mean and save it
# #
# # data, data_name = FM.operate('diff', kwargs={'periods': 1}, save_it=False) # lets perform a diff operation
# # print(data_name)
# # print(data)
# # FM.operate('diff', kwargs={'periods': -1}, save_it=True) # now lets save it and save it
# #
# # # now lets change the operational key so we can transform some data we just transformed, specifically the rolling mean 10
# # FM.set_operation_key(data_name_rolling_mean_100)
# #
# # data, data_name_diff_100_mean = FM.operate('diff', kwargs={'periods': -50}, save_it=False) # lets check how the name will change
# # print(data_name_diff_100_mean)
# # print("notice the FD__ twice, this means the data has been transformed twice")
# # print('also notice that how the data was transformed can be seen because it is split up by ____ (4 underscores)')
# #
# # data, data_name_diff_100_mean = FM.operate('diff', kwargs={'periods': -50}, save_it=True) # save it
# #
# # a = utils.print_h5_keys(h5_in, 1, 1)
# # key_name_list = ['FD__original', 'FD__FD__FD__original_rolling_mean_W_100_SFC_0_MP_100_____diff_periods_-50____', 'FD__FD__original_rolling_mean_W_100_SFC_0_MP_100____']
# # with h5py.File(h5_in, 'r+') as h:
# #     for k in key_name_list:
# #         plt.plot(h[k][:8000, 0])
# #
# #
#
# h5_in = '/Users/phil/Desktop/AH0407_160613_JC1003_AAAC_3lag.h5'
# a = utils.print_h5_keys(h5_in, 1, 1)
#
# def rename_h5_images_to_feature_data(h5, len_shape = 2, shape_ind_1 = 2048):
#     if not utils.h5_key_exists(h5, 'images'):
#         print('images key does not exist, returning')
#         return
#     with h5py.File(h5, 'r+') as h:
#         s = h['images'].shape
#         assert len(s) == len_shape, 'length of images key does not match length check, might be actual images and not feature data'
#         assert s[1] == shape_ind_1, 'shape of ind 1 does not match, might be actual images and not feature data'
#         h['feature_data'] = h['images']
#         del h['images']
#
# temp_dir = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing_features_data/feature_data/'
# for h5_in in utils.get_h5s(temp_dir, 0):
#     rename_h5_images_to_feature_data(h5_in, len_shape = 2, shape_ind_1 = 2048)
#
#
#
#
# def rename_h5_images_to_feature_data(h5, len_shape = 2, shape_ind_1 = 2048):
#     if not utils.h5_key_exists(h5, 'feature_data'):
#         print('images key does not exist, returning')
#         return
#     with h5py.File(h5, 'r+') as h:
#         s = h['feature_data'].shape
#         assert len(s) == len_shape, 'length of images key does not match length check, might be actual images and not feature data'
#         assert s[1] == shape_ind_1, 'shape of ind 1 does not match, might be actual images and not feature data'
#         h['FD__original'] = h['feature_data']
#         del h['feature_data']
#
# temp_dir = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing_features_data/feature_data/'
# for h5_in in utils.get_h5s(temp_dir, 0):
#     rename_h5_images_to_feature_data(h5_in, len_shape = 2, shape_ind_1 = 2048)
#
#
#
# "FD__FD__XXXXX_rolling_mean_W_100_SFC_0_MP_100_____diff_periods_"
# "FD__XXXXX_rolling_mean_W_100_SFC_0_MP_100____"
# "XXXXX"
#
# "FD__FD__FD__original_rolling_mean_W_100_SFC_0_MP_100_____diff_periods_"
# "FD__FD__original_rolling_mean_W_100_SFC_0_MP_100____"
# "FD__original"
#
#
# from whacc.feature_maker import feature_maker
# h5_in = '/Users/phil/Desktop/holy_test_set_10_percent_3lag.h5'
#
# FM = feature_maker(h5_in, frame_num_ind=2, delete_if_exists = True, operational_key='FD__original')
#
# data, data_name_rolling_mean_100 = FM.rolling(11, 'mean', save_it=True)  #lets do the rolling mean and save it
# """
# so this works fine just update whacc and try again
# """

from whacc import PoleTracking, utils
import whacc
import os
import glob


# this will use the pole tracker on all subdirectories, if you want to do it on one directory
# just set ... video_directory = ['your/full.directory/here/']
mp4_path = '/content/gdrive/My Drive/WhACC_PROCESSING_FOLDER/processing_1'
search_term = '*.mp4'
folders_with_MP4s = whacc.utils.recursive_dir_finder(mp4_path, search_term)
print(folders_with_MP4s, sep='\n')


for video_directory in folders_with_MP4s:
    utils.make_mp4_list_dict(video_directory)




CROP_SIZE = [71, 71]
PT = dict()
for i, video_directory in enumerate(folders_with_MP4s):
    template_img_full_name = glob.glob(video_directory + os.path.sep + '*template_img.png')
    if len(template_img_full_name) == 1:  # if the template already exists
        PT[i] = PoleTracking(video_directory=video_directory, template_png_full_name=template_img_full_name[0])
    else:
        PT[i] = PoleTracking(video_directory=video_directory)  # create the 'PoleTracking' class
        PT[i].cut_out_pole_template(crop_size=CROP_SIZE, frame_num=2000, file_ind=2)  # user cut out template image
        PT[i].save_template_img(cust_save_dir=PT[i].video_directory)  # save the template image
