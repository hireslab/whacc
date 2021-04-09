# import tensorflow as tf
from PSM_testbed.WhACC import utils
from PSM_testbed.WhACC import image_tools


#
# m_list = utils.get_model_list('/Users/phil/Dropbox/Colab data/model_iterations')
#
# H5_list = ['/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/AH0667x170317-.h5']
#
# model_2_load = m_list[15]
# # MOD_BASE_NAME = model_2_load.split('/')[-1].split('.')[0]
# model = tf.keras.models.load_model(model_2_load)
# gen = image_tools.ImageBatchGenerator(1000, H5_list)
# predictions = model.predict(gen, verbose = 1)
# ###

# h5_list = utils.get_h5s('/Users/phil/Dropbox/Autocurator/testing_data/MP4s/')
# h5_2_load = h5_list[2]

m_list = utils.get_model_list('/Users/phil/Dropbox/Colab data/model_iterations')
model_2_load = m_list[12]

# H5_list = [h5_2_load]
H5_list = utils.get_h5s('/Users/phil/Dropbox/Autocurator/testing_data/MP4s/')
H5_file = image_tools.run_multiple_H5_files(H5_list,
                                            model_2_load,
                                            append_model_and_labels_to_name_string = False,
                                            batch_size = 1000,
                                            save_on = True,
                                            label_save_name = 'labels')

import h5py
import matplotlib.pyplot as plt
for k in H5_list:
    with h5py.File(k, 'r') as hf:
            plt.plot(hf['labels'][:])



# import h5py
# import matplotlib.pyplot as plt
# h5_file_name = '/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/AH0667x170317-.h5'
# with h5py.File(h5_file_name, 'r') as hf:
#     # print(hf['labels'])
#     for k in hf.keys():
#         print(k)
#     plt.plot(hf['MODEL__resnet50 regular images full unfrozen20201028FULL__labels'][:])
# #
