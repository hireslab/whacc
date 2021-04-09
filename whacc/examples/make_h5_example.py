# from PSM_testbed import WhACC
from PSM_testbed.WhACC import PoleTracking
# from PSM_testbed.WhACC import utils
# from Users/phil/Dropbox/Autocurator/code/WhACC_testing/PSM_testbed/ import PSM_testbed

import PSM_testbed
import glob
import os
import matplotlib.pyplot as plt

mp4_path = "/Users/phil/Dropbox/Autocurator/testing_data/MP4s"
search_term = '*.mp4'
##
folders_with_MP4s = PSM_testbed.WhACC.utils.recursive_dir_finder(mp4_path, search_term)
print(folders_with_MP4s, sep='\n')
# folders_with_MP4s = folders_with_MP4s[:2]

PT = dict()
for i, video_directory in enumerate(folders_with_MP4s):
    template_img_full_name = glob.glob(video_directory + os.path.sep + '*template_img.png')
    if len(template_img_full_name) == 1:  # if the template already exists
        PT[i] = PoleTracking(video_directory=video_directory, template_png_full_name=template_img_full_name[0])
    else:
        PT[i] = PoleTracking(video_directory=video_directory)  # create the class
        PT[i].cut_out_pole_template(video_directory, crop_size=[61, 61], frame_num=1200,
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
    output_h5 = PT[i].track_all_and_save()
