from whacc import PoleTracking, utils, image_tools, model_maker, feature_maker
import os
from whacc.feature_maker import feature_maker, total_rolling_operation_h5_wrapper, convert_h5_to_feature_h5, standard_feature_generation, load_selected_features
import h5py
import numpy as np
import time
import shutil
from pathlib import Path
from natsort import os_sorted


def batch_process_videos_on_colab(bd, local_temp_dir, video_batch_size=30):
    bd_base_name = os.path.basename(os.path.normpath(bd))
    # load model once in the beginning
    RESNET_MODEL = model_maker.load_final_model()  # NOTE: there will be a warning since this likely isn't optimized for GPU, this is fine
    fd = utils.load_feature_data()  # load feature data info
    while True:  # once a   ll files are
        time_list = []
        start = time.time()
        grab_file_list = True
        while grab_file_list:  # continuously look for files to run
            # get files that tell us which mp4s to process
            list_of_file_dicts = np.asarray(utils.get_files(bd, '*file_list_for_batch_processing.pkl'))
            # sort it by the newest first since we we edit it each time (becoming the newest file)
            # this ensures we finished one set completely first
            inds = np.flip(np.argsort([os.path.getctime(k) for k in list_of_file_dicts]))
            list_of_file_dicts = list_of_file_dicts[inds]
            if len(list_of_file_dicts) == 0:
                print('FINISHED PROCESSING')
                assert False, "FINISHED PROCESSING no more files to process"
            # load file dictionary
            file_dict = utils.load_obj(list_of_file_dicts[0])
            # get base directory for current videos we are processing
            mp4_bd = os.path.dirname(list_of_file_dicts[0])
            # copy folder structure for the finished mp4s and predictions to go to
            utils.copy_folder_structure(bd, os.path.normpath(bd) + '_FINISHED')
            # check if all the files have already been processes
            if np.all(file_dict['is_processed'] == True):
                x = list_of_file_dicts[
                    0]  # copy the instruction file with list of mp4s to final directory we are finished
                shutil.move(x, x.replace(bd_base_name, bd_base_name + '_FINISHED'))
                x = os.path.dirname(x) + os.sep + 'template_img.png'
                shutil.move(x, x.replace(bd_base_name, bd_base_name + '_FINISHED'))
            else:
                grab_file_list = False  # ready to run data

        # overwrite local folder to copy files to
        if os.path.exists(local_temp_dir):
            shutil.rmtree(local_temp_dir)
        Path(local_temp_dir).mkdir(parents=True, exist_ok=True)
        # copy over mp4s and template image to local directory
        x = os.sep + 'template_img.png'
        template_dir = local_temp_dir + x
        shutil.copy(mp4_bd + x, template_dir)
        process_these_videos = np.where(file_dict['is_processed'] == False)[0][:video_batch_size]
        for i in process_these_videos:
            x = os.sep + os.path.basename(file_dict['mp4_names'][i])
            shutil.copy(mp4_bd + x, local_temp_dir + x)

        # track the mp4s for the pole images
        PT = PoleTracking(video_directory=local_temp_dir, template_png_full_name=template_dir)
        PT.track_all_and_save()

        # convert the images to '3lag' images
        #  h5_in = '/Users/phil/Desktop/temp_dir/AH0407x160609.h5'
        h5_in = PT.full_h5_name
        h5_3lag = h5_in.replace('.h5', '_3lag.h5')
        image_tools.convert_to_3lag(h5_in, h5_3lag)

        # convert to feature data
        # h5_3lag = '/Users/phil/Desktop/temp_dir/AH0407x160609_3lag.h5'
        h5_feature_data = h5_3lag.replace('.h5', '_feature_data.h5')
        in_gen = image_tools.ImageBatchGenerator(500, h5_3lag, label_key='labels')
        convert_h5_to_feature_h5(RESNET_MODEL, in_gen, h5_feature_data)

        # delete 3lag don't it need anymore
        os.remove(h5_3lag)
        # h5_feature_data = '/Users/phil/Desktop/temp_dir/AH0407x160609_3lag_feature_data.h5'
        # generate all the modified features (41*2048)+41 = 84,009
        standard_feature_generation(h5_feature_data)
        all_x = load_selected_features(h5_feature_data)
        # delete the big o' file
        file_nums = str(process_these_videos[0] + 1) + '_to_' + str(process_these_videos[-1] + 1) + '_of_' + str(
            len(file_dict['is_processed']))
        h5_final = h5_in.replace('.h5', '_final_to_combine_' + file_nums + '.h5')
        print(h5_final)
        with h5py.File(h5_final, 'w') as h:
            h['final_3095_features'] = all_x
        utils.copy_over_all_non_image_keys(h5_in, h5_final)
        # then if you want you can copy the images too maybe just save as some sort of mp4 IDK
        os.remove(h5_feature_data)
        x = os.path.dirname(list_of_file_dicts[0]) + os.sep
        dst = x.replace(bd_base_name, bd_base_name + '_FINISHED') + os.path.basename(h5_final)
        shutil.copy(h5_final, dst)

        for k in process_these_videos:  # save the dict file so that we know the video has been processed
            file_dict['is_processed'][k] = True
        utils.save_obj(file_dict, list_of_file_dicts[0])

        # move the mp4s to the final dir
        for i in process_these_videos:
            x = mp4_bd + os.sep + os.path.basename(file_dict['mp4_names'][i])
            shutil.move(x, x.replace(bd_base_name, bd_base_name + '_FINISHED'))
        time_list.append(time.time() - start)


"""
prior to this you will need a template image in each of your MP4 directories 
and you must have run the following on the complete i.e. -- no missing mp4s due to them still uploading for example, 
so this is best to do on local then upload. bu if all are already uploaded can do this just as easily on the cloud. 

base_directory = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_batch_processing_test/add_files_here/'
folders_with_MP4s = whacc.utils.recursive_dir_finder(base_directory, '*.mp4')
for video_directory in folders_with_MP4s:
    utils.make_mp4_list_dict(video_directory) # this creates all the a file for batch processing to keep track of every 
    # folder with MP4s to process  
"""
video_batch_size = 5
# local colab folder, or other local folder.
local_temp_dir = '/Users/phil/Desktop/temp_dir'
# base_processing_folder
bd = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_batch_processing_test/add_files_here/'

batch_process_videos_on_colab(bd, local_temp_dir, video_batch_size=30)

b2_back_one = os.path.dirname(os.path.normpath(bd))  # will look one directory back
utils.auto_combine_final_h5s(b2_back_one) # if length is one and 0 through end then just rename it



from whacc import PoleTracking
video_directory = '/Users/phil/Dropbox/HIRES_LAB/curation_for_auto_curator/Data/Samson/Session322/'
PT = PoleTracking(video_directory=video_directory)  # create the 'PoleTracking' class
PT.cut_out_pole_template(crop_size=[71]*2, frame_num=2000, file_ind=2)  # user cut out template image
PT.save_template_img(cust_save_dir=PT.video_directory)  # save the template image




#
# from whacc import PoleTracking, utils, image_tools, model_maker, feature_maker
# import os
# from whacc.feature_maker import feature_maker, total_rolling_operation_h5_wrapper, convert_h5_to_feature_h5, standard_feature_generation, load_selected_features
# import h5py
# import numpy as np
# import time
# import shutil
# from pathlib import Path
# from natsort import os_sorted
#
# ################
# ################ USER SETTINGS BELOW
# ################
# video_batch_size = 5
# # local colab folder, or other local folder.
# local_temp_dir = '/Users/phil/Desktop/temp_dir'
# # base_processing_folder
# bd = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_batch_processing_test/add_files_here/'
# bd_base_name = os.path.basename(os.path.normpath(bd))
# ################
# ################ USER SETTINGS ABOVE
# ################
# # load model once in the beginning
# RESNET_MODEL = model_maker.load_final_model()  # NOTE: there will be a warning since this likely isn't optimized for GPU, this is fine
# fd = utils.load_feature_data()  # load feature data info
# while True:  # once all files are
#     time_list = []
#     start = time.time()
#     grab_file_list = True
#     while grab_file_list:  # continuously look for files to run
#         # get files that tell us which mp4s to process
#         list_of_file_dicts = np.asarray(utils.get_files(bd, '*file_list_for_batch_processing.pkl'))
#         # sort it by the newest first since we we edit it each time (becoming the newest file)
#         # this ensures we finished one set completely first
#         inds = np.flip(np.argsort([os.path.getctime(k) for k in list_of_file_dicts]))
#         list_of_file_dicts = list_of_file_dicts[inds]
#         if len(list_of_file_dicts) == 0:
#             print('FINISHED PROCESSING')
#             assert False, "FINISHED PROCESSING no more files to process"
#         # load file dictionary
#         file_dict = utils.load_obj(list_of_file_dicts[0])
#         # get base directory for current videos we are processing
#         mp4_bd = os.path.dirname(list_of_file_dicts[0])
#         # copy folder structure for the finished mp4s and predictions to go to
#         utils.copy_folder_structure(bd, os.path.normpath(bd) + '_FINISHED')
#         # check if all the files have already been processes
#         if np.all(file_dict['is_processed'] == True):
#             x = list_of_file_dicts[0]  # copy the instruction file with list of mp4s to final directory we are finished
#             shutil.move(x, x.replace(bd_base_name, bd_base_name + '_FINISHED'))
#             x = os.path.dirname(x) + os.sep + 'template_img.png'
#             shutil.move(x, x.replace(bd_base_name, bd_base_name + '_FINISHED'))
#         else:
#             grab_file_list = False  # ready to run data
#
#     # overwrite local folder to copy files to
#     if os.path.exists(local_temp_dir):
#         shutil.rmtree(local_temp_dir)
#     Path(local_temp_dir).mkdir(parents=True, exist_ok=True)
#     # copy over mp4s and template image to local directory
#     x = os.sep + 'template_img.png'
#     template_dir = local_temp_dir + x
#     shutil.copy(mp4_bd + x, template_dir)
#     process_these_videos = np.where(file_dict['is_processed'] == False)[0][:video_batch_size]
#     for i in process_these_videos:
#         x = os.sep + os.path.basename(file_dict['mp4_names'][i])
#         shutil.copy(mp4_bd + x, local_temp_dir + x)
#
#     # track the mp4s for the pole images
#     PT = PoleTracking(video_directory=local_temp_dir, template_png_full_name=template_dir)
#     PT.track_all_and_save()
#
#     # convert the images to '3lag' images
#     #  h5_in = '/Users/phil/Desktop/temp_dir/AH0407x160609.h5'
#     h5_in = PT.full_h5_name
#     h5_3lag = h5_in.replace('.h5', '_3lag.h5')
#     image_tools.convert_to_3lag(h5_in, h5_3lag)
#
#     # convert to feature data
#     # h5_3lag = '/Users/phil/Desktop/temp_dir/AH0407x160609_3lag.h5'
#     h5_feature_data = h5_3lag.replace('.h5', '_feature_data.h5')
#     in_gen = image_tools.ImageBatchGenerator(500, h5_3lag, label_key='labels')
#     convert_h5_to_feature_h5(RESNET_MODEL, in_gen, h5_feature_data)
#
#     # delete 3lag don't it need anymore
#     os.remove(h5_3lag)
#     # h5_feature_data = '/Users/phil/Desktop/temp_dir/AH0407x160609_3lag_feature_data.h5'
#     # generate all the modified features (41*2048)+41 = 84,009
#     standard_feature_generation(h5_feature_data)
#     all_x = load_selected_features(h5_feature_data)
#     # delete the big o' file
#     file_nums = str(process_these_videos[0] + 1) + '_to_' + str(process_these_videos[-1] + 1) + '_of_' + str(
#         len(file_dict['is_processed']))
#     h5_final = h5_in.replace('.h5', '_final_to_combine_' + file_nums + '.h5')
#     print(h5_final)
#     with h5py.File(h5_final, 'w') as h:
#         h['final_3095_features'] = all_x
#     utils.copy_over_all_non_image_keys(h5_in, h5_final)
#     # then if you want you can copy the images too maybe just save as some sort of mp4 IDK
#     os.remove(h5_feature_data)
#     x = os.path.dirname(list_of_file_dicts[0]) + os.sep
#     dst = x.replace(bd_base_name, bd_base_name + '_FINISHED') + os.path.basename(h5_final)
#     shutil.copy(h5_final, dst)
#
#     for k in process_these_videos:  # save the dict file so that we know the video has been processed
#         file_dict['is_processed'][k] = True
#     utils.save_obj(file_dict, list_of_file_dicts[0])
#
#     # move the mp4s to the final dir
#     for i in process_these_videos:
#         x = mp4_bd + os.sep + os.path.basename(file_dict['mp4_names'][i])
#         shutil.move(x, x.replace(bd_base_name, bd_base_name + '_FINISHED'))
#     end = time.time()
#     time_list.append(end - start)
#
#
#
