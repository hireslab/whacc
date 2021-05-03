import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

h5_file = "/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/AH0667x170317.h5"


trial_num = 1
frame_num = 0
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


max_match_val_all_2 = ['cv2.TM_CCOEFF']
for method in methods:
    with h5py.File(h5_file, 'r+') as h:
        cum_frames = (h['trial_nums_and_frame_nums'][1, :]).astype('int').copy()
        cum_frames = np.cumsum(cum_frames)
        frame_ind = (np.sum(h['trial_nums_and_frame_nums'][1, :trial_num]) + frame_num).astype('int')
        print(frame_ind)
        template_img = h['images'][frame_ind].copy()
        max_match_val = []
        max_match_val_all = []
    # plt.figure()
    # plt.imshow(template_img)
    with h5py.File(h5_file, 'r+') as h:
        for i, k in enumerate(tqdm(h['images'])):
            if np.any(cum_frames == i):
                max_match_val_all.append(max_match_val)
                max_match_val = []
            res = cv2.matchTemplate(template_img,k,  eval(method))
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            max_match_val.append(max_val)
            # max_match_val.append(res[0][0])
    max_match_val_all_2.append(max_match_val_all)
    plt.figure()
    plt.title(method)
    for k in max_match_val_all:
        plt.plot(k)



#
# with h5py.File(h5_file, 'r+') as h:
#     max_match_val_all = []
#     for i, k in enumerate(tqdm(h['images'])):
#         if np.any(cum_frames == i):
#             max_match_val_all.append(max_match_val)
#             max_match_val = []
#         a = np.sum(np.abs(template_img - k))
#         # res = cv2.matchTemplate(k, template_img, cv2.TM_CCOEFF)
#         # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#         max_match_val.append(a)
# plt.figure()
# plt.plot(max_match_val_all[0])
