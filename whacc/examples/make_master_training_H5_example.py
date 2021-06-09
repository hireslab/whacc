

from whacc import utils
from whacc import image_tools

"""
from google.colab import drive 
drive.mount('/content/gdrive')
"""

# This is all that is needed to make a master H5 of near touches only, the iamges do not have color chanels to save space.
all_h5s = utils.get_h5s('/content/gdrive/My Drive/Colab data/curation_for_auto_curator/finished_contacts/')
all_h5s_imgs = utils.get_h5s('/content/gdrive/My Drive/Colab data/curation_for_auto_curator/H5_data/')
h_cont = utils._get_human_contacts_(all_h5s)
h5c = image_tools.h5_iterative_creator('/content/gdrive/My Drive/Colab data/curation_for_auto_curator/test_____.h5',
                                       overwrite_if_file_exists = True,
                                       color_channel = False)
utils.create_master_dataset(h5c, all_h5s_imgs, h_cont, borders = 80, max_pack_val = 100)
