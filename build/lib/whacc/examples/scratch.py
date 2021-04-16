# # from whacc import image_tools
# #
# # image_tools.h5_iterative_creator()
#
#
# from whacc import utils
# from whacc import image_tools
# import time
# from keras.preprocessing.image import ImageDataGenerator
# import h5py
# import matplotlib.pyplot as plt
# from tqdm import tqdm
#
#
# # # $$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # plt.figure(figsize=[5, 10])
# # with h5py.File('/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/AH0667x170317_subset_AUG.h5', 'r') as hf:
# #     for k in range(20):
# #         print(k)
# #         plt.imshow(image_tools.img_unstacker(hf['images'][30*k: 30*(k+1)], 5))
# #         plt.pause(.5)
# #         plt.show()
#
#
#
# # $$$$$$$$$$$$$$$$$$$$$$$$$$$$
# image_tools.split_h5('/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/AH0667x170317.h5', [9, 1],
#                        '/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/AH0667x170317_temp_')
# # $$$$$$$$$$$$$$$$$$$$$$$$$$$$
from whacc import image_tools
help(image_tools.predict_multiple_H5_files)

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$

