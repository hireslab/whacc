from whacc import PoleTracking, utils
import whacc
import glob
import os
import matplotlib.pyplot as plt
from whacc.touch_curation_GUI import touch_gui

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

f = '/Users/phil/Desktop/pipeline_test/AH0407x160609.h5'
f2 = f.replace('.h5', '_3lag.h5')
utils.convert_to_3lag(f, f2)
utils.print_h5_keys(f2)
utils.print_h5_keys(f)


f = '/Users/phil/Desktop/pipeline_test/AH0407x160609.h5'
f2 = f.replace('.h5', '_3lag.h5')
utils.stack_lag_h5_maker(f, f2, buffer=2, shift_to_the_right_by=0) # dont change these setting these are for making 3lag, make simple function for this

# if you want to hand curate you can use the GUI, you can also scan through your images.

touch_gui(f2, 'labels', 'labels')
