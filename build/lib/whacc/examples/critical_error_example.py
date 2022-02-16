from whacc import utils, image_tools, transfer_learning, analysis
from IPython.utils import io
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow import keras
from sklearn.utils import class_weight
import time
from pathlib import Path
import os
import copy
import numpy as np
from tensorflow.keras import applications
from pathlib import Path
import shutil
import zipfile
from datetime import datetime
import pytz
import json
from whacc import model_maker

from whacc.model_maker import *
import itertools

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import h5py
from whacc import image_tools
from whacc import utils
import copy
import time
import os
import pdb
import glob
from tqdm.contrib import tzip
import scipy.io as spio
import h5py
from tqdm.notebook import tqdm
from matplotlib import cm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.patches as mpatches
from scipy.signal import medfilt, medfilt2d
import cv2


def foo_critical_error(tmp1_in):
    num_critical_error = []
    for kk in tmp1_in:
        num_critical_error.append(len(kk[0]) - len([1 for k in kk[0] if k == 'append' or k == 'deduct']))
    return num_critical_error


pred = [0,0,0,1,1,1,1,0,0,0,0]
real = [0,0,0,1,1,1,1,0,0,0,0]
