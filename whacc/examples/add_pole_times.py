# import numpy as np
# # import scipy.io
# import scipy.io as spio
# from google.colab import drive
# import glob
# import matplotlib.pyplot as plt
# import h5py
# import os
# # from functools import partial
# from tqdm import tqdm
# # tqdm = partial(tqdm, position=0, leave=True)
# import difflib
# import copy
# import re
# from matplotlib.colors import ListedColormap
# import itertools
# from pathlib import Path

from whacc import utils

pole_file_list = utils.get_files("/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH1131X26032020ses338_SAMSON/", '*.mat')


pt = utils.loadmat(pole_file_list[0])
