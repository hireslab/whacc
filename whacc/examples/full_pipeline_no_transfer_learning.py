from whacc import PoleTracking, utils, image_tools, model_maker, feature_maker
import whacc
import glob
import os
import matplotlib.pyplot as plt
from whacc.touch_curation_GUI import touch_gui

"""
NOTES TO DO 
ensure each has an overwrite option just delete file if exists dont want to 
retrack things on accident that would waste time

"""
"""######## MAKE H5s FROM MP4s ########"""

# this will use the pole tracker on all subdirectories, if you want to do it on one directory
# just set ... video_directory = ['your/full.directory/here/']
mp4_path = '/Users/phil/Desktop/pipeline_test/'
search_term = '*.mp4'
folders_with_MP4s = whacc.utils.recursive_dir_finder(mp4_path, search_term)
print(folders_with_MP4s, sep='\n')

PT = dict()
for i, video_directory in enumerate(folders_with_MP4s):
    template_img_full_name = glob.glob(video_directory + os.path.sep + '*template_img.png')
    if len(template_img_full_name) == 1:  # if the template already exists
        PT[i] = PoleTracking(video_directory=video_directory, template_png_full_name=template_img_full_name[0])
    else:
        PT[i] = PoleTracking(video_directory=video_directory)  # create the 'PoleTracking' class
        PT[i].cut_out_pole_template(crop_size=[61, 61], frame_num=2000, file_ind=2)  # user cut out template image
        PT[i].save_template_img(cust_save_dir=PT[i].video_directory)  # save the template image

for i in PT: #plot image and display trial numbers extract to ensure they match
    PT_tmp = PT[i]
    PT_tmp.get_trial_and_file_names(print_them=True, num_to_print=5)
    PT_tmp.save_template_img(cust_save_dir=PT_tmp.video_directory)
    plt.figure()
    plt.title(PT_tmp.video_directory)
    plt.imshow(PT_tmp.template_image)

for i in PT:
    output_h5 = PT[i].track_all_and_save()


"""######## MAKE 3lag HF FILES - do this for each h5 generated above, only one is illustrated here ########"""
f = '/Users/phil/Desktop/pipeline_test/AH0407x160609.h5'
h5_3lag = f.replace('.h5', '_3lag.h5')
image_tools.convert_to_3lag(f, h5_3lag)
# utils.print_h5_keys(h5_3lag)
# utils.print_h5_keys(f)

# touch_gui(h5_3lag, 'labels', 'labels') # if you want to view your h5 data

h5_feature_data = h5_3lag.replace('.h5', '_feature_data.h5')
mod = model_maker.load_final_model()
in_gen = image_tools.ImageBatchGenerator(500, h5_3lag, label_key='labels')
feature_maker.convert_h5_to_feature_h5(mod, in_gen, h5_feature_data)

utils.print_h5_keys(h5_feature_data)



from whacc.feature_maker import feature_maker, total_rolling_operation_h5_wrapper
from tqdm.auto import tqdm
import numpy as np


##########################################################################################
##################### NOW ENGINEER LAG SMOOTH SD FEATURES ETC#############################
##########################################################################################
##########################################################################################

FM = feature_maker(h5_feature_data, operational_key='FD__original', delete_if_exists=True)

for periods in tqdm([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]):
    data, key_name = FM.shift(periods, save_it=True)

for smooth_it_by in tqdm([3, 7, 11, 15, 21, 41, 61]):
    data, key_name = FM.rolling(smooth_it_by, 'mean', save_it=True)

for periods in tqdm([-50, -20, -10, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 10, 20, 50]):
    data, key_name = FM.operate('diff', kwargs={'periods': periods}, save_it=True)

for smooth_it_by in tqdm([3, 7, 11, 15, 21, 41, 61]):
    data, key_name = FM.rolling(smooth_it_by, 'std', save_it=True)

win = 1
# key_to_operate_on = 'FD__original'
op = np.std
mod_key_name = 'DF_TOTAL_std_' + str(win) + '_of_'
all_keys = utils.lister_it(utils.print_h5_keys(FM.h5_in, 1, 0), 'FD__', 'DF_TOTAL')
for key_to_operate_on in tqdm(all_keys):
    data_out = total_rolling_operation_h5_wrapper(FM, win, op, key_to_operate_on, mod_key_name = mod_key_name, save_it = True)
