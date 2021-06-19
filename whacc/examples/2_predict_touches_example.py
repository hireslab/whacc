from whacc import utils
from whacc import image_tools

m_list = utils.get_model_list('/Users/phil/Dropbox/Colab data/model_iterations')
model_2_load = m_list[12]
# model_2_load = m_list[6]


H5_list = utils.get_h5s('/Users/phil/Dropbox/Autocurator/testing_data/MP4s/')
H5_file = image_tools.predict_multiple_H5_files(H5_list,
                                                model_2_load,
                                                append_model_and_labels_to_name_string = False,
                                                batch_size = 1000,
                                                save_on = True,
                                                label_save_name = 'labels')

# import h5py
# import matplotlib.pyplot as plt
# for k in H5_list:
#     with h5py.File(k, 'r') as hf:
#             plt.plot(hf['labels'][:])



# import h5py
# import matplotlib.pyplot as plt
# h5_file_name = '/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/AH0667x170317-.h5'
# with h5py.File(h5_file_name, 'r') as hf:
#     # print(hf['labels'])
#     for k in hf.keys():
#         print(k)
#     plt.plot(hf['MODEL__resnet50 regular images full unfrozen20201028FULL__labels'][:])
# #
