from whacc import PoleTracking
import whacc
import glob
import os
import matplotlib.pyplot as plt

mp4_path = "/Users/phil/Dropbox/Autocurator/testing_data/MP4s"
mp4_path = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/AH0000x000000/"
mp4_path = "/Users/phil/Downloads/untitled folder 2/"
# mp4_path = "/Users/phil/Dropbox/HIRES_LAB/GitHub/Phillip_AC/autoCuratorDiverseDataset/delete_after_oct_2021/"
search_term = '*.mp4'
##
folders_with_MP4s = whacc.utils.recursive_dir_finder(mp4_path, search_term)
print(folders_with_MP4s, sep='\n')
# folders_with_MP4s = folders_with_MP4s[:2]

PT = dict()
for i, video_directory in enumerate(folders_with_MP4s):
    template_img_full_name = glob.glob(video_directory + os.path.sep + '*template_img.png')
    if len(template_img_full_name) == 1:  # if the template already exists
        PT[i] = PoleTracking(video_directory=video_directory, template_png_full_name=template_img_full_name[0])
    else:
        PT[i] = PoleTracking(video_directory=video_directory)  # create the class
        PT[i].cut_out_pole_template(crop_size=[61, 61], frame_num=1200,
                                    file_ind=2)  # cut out template image## 270 , 120# 270, 235# 187 , 249# 168 , 276
        PT[i].save_template_img(cust_save_dir=PT[i].video_directory)  # save the template image

for i in PT:
    PT_tmp = PT[i]
    PT_tmp.get_trial_and_file_names(print_them=True, num_to_print=5)
    PT_tmp.save_template_img(cust_save_dir=PT_tmp.video_directory)
    plt.figure()
    plt.title(PT_tmp.video_directory)
    plt.imshow(PT_tmp.template_image)

for i in PT:
    PT[i].use_narrow_search_to_speed_up = False
    output_h5 = PT[i].track_all_and_save()


#
# #
# import h5py
# import copy
# with h5py.File('/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/AH0667x170317.h5', 'r') as hf:
#     # print(hf.keys())
#     L = copy.deepcopy(hf['locations'][:])
#
#
# #
# import matplotlib.pyplot as plt
# a = L[0]
# plt.plot(a[:, 0], a[:, 1], 'ko')

