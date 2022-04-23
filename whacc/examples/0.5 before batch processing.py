from whacc import PoleTracking, utils
import whacc
import glob
import os

mp4_path = "/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_batch_processing_test/add_files_here_FINISHED/"
mp4_path = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_batch_processing_test/add_files_here/'
folders_with_MP4s = whacc.utils.recursive_dir_finder(mp4_path, '*.mp4')
_ = [print(str(i) + ' ' + k) for i, k in enumerate(folders_with_MP4s)]

for video_directory in folders_with_MP4s:
    utils.make_mp4_list_dict(video_directory)

you_are_samson_king = True
if you_are_samson_king:
  CROP_SIZE = [71, 71]
else:
  CROP_SIZE = [61, 61]

init_template_dir = None
PT = dict()
for i, video_directory in enumerate(folders_with_MP4s):
    template_img_full_name = glob.glob(video_directory + os.path.sep + '*template_img.png')
    if len(template_img_full_name) == 1:
        init_template_dir = template_img_full_name
    if len(template_img_full_name) == 0:  # if the template already exists
        PT[i] = PoleTracking(video_directory=video_directory)  # create the 'PoleTracking' class
        PT[i].custom_init_pole_template_dir = init_template_dir # auto find a close match when init-ing
        PT[i].cut_out_pole_template(crop_size=CROP_SIZE, frame_num=2000, file_ind=2)  # user cut out template image
        PT[i].save_template_img(cust_save_dir=PT[i].video_directory)  # save the template image



PT[i] = PoleTracking(video_directory=video_directory)  # create the 'PoleTracking' class
init_template_dir = '/Users/phil/Desktop/template_img_set_test.png'
PT[i].custom_init_pole_template_dir = init_template_dir # auto find a close match when init-ing
PT[i].cut_out_pole_template(crop_size=CROP_SIZE, frame_num=2000, file_ind=2)  # user cut out template image
PT[i].save_template_img(cust_save_dir=PT[i].video_directory)  # save the template image



# PT = PoleTracking(video_directory=video_directory)  # create the class
# PT.cut_out_pole_template(crop_size=CROP_SIZE, frame_num=2000, file_ind=0)
# PT.save_template_img(cust_save_dir=PT.video_directory)
#
#
#
#
# you_are_samson_king = True
# if you_are_samson_king:
#   CROP_SIZE = [71, 71]
# else:
#   CROP_SIZE = [61, 61]
#
# init_template_dir = None
# PT = dict()
# for i, video_directory in enumerate(folders_with_MP4s):
#     template_img_full_name = glob.glob(video_directory + os.path.sep + '*template_img.png')
#     init_template_dir = template_img_full_name
#     if len(template_img_full_name) == 0:  # if the template already exists
#         PT[i].custom_init_pole_template_dir = init_template_dir # auto find a close match when init-ing
#         PT[i] = PoleTracking(video_directory=video_directory)  # create the 'PoleTracking' class
#         PT[i].cut_out_pole_template(crop_size=CROP_SIZE, frame_num=2000, file_ind=2)  # user cut out template image
#         PT[i].save_template_img(cust_save_dir=PT[i].video_directory)  # save the template image
#
#
