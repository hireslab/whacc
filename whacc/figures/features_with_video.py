from whacc.feature_maker import feature_maker, total_rolling_operation_h5_wrapper
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import h5py
import seaborn as sns
# run quoted code below to make the original mini file
import scipy
import scipy.cluster.hierarchy as sch
import pandas as pd
import os
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

def cluster_corr(corr_array, max_div_by = 2):
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/max_div_by
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx], idx

"""NOTE you will have to have both the features and the images in one H5 file"""

h5_in = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/example_feature.h5'

# utils.print_h5_keys(h5_in)
with h5py.File(h5_in, 'r') as h:
    images = h['images'][:]
    labels = h['labels'][:]
    frame_nums = h['frame_nums'][:]
    base_name = h['base_name']
    FD__original = h['FD__original'][:]


# for div_by in [4]:#[2, 3, 4, 30, 50, 100]: I think 4 is the best for visualizing the groups of neurons
div_by = 4
FD_img = FD__original-np.min(FD__original)
FD_img = ((FD_img/np.max(FD_img))*255).astype(np.uint8)
cc = np.corrcoef(FD_img, rowvar=False)
plt.figure()
sns.heatmap(cc)
cc, sorted_inds = cluster_corr(cc, div_by) # number here sets threshold for clustering, higher results in more cluster
plt.figure()
sns.heatmap(cc)

# FD_img = FD__original-np.min(FD__original)
# FD_img = ((FD_img/np.max(FD_img))*255).astype(np.uint8)
# # tmp1 = np.concatenate(([0], np.diff(labels)==1))
# # tmp1 = labels==1
# sorted_inds = np.argsort(np.mean(FD_img[tmp1, :], axis=0))
# FD_img = FD_img[:, sorted_inds]
# cc = np.corrcoef(FD_img, rowvar=False)
# plt.figure()
# sns.heatmap(cc)

FD_img = FD_img[:, sorted_inds] # sort features by clustering

expand_by = 1000 # size of whisker and feature frames, so total will (be expand_by*2 by expand_by)
bd = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/final_video/' + os.sep
Path(bd).mkdir(parents=True, exist_ok=True)
width = 32
height = 64

for i, (img, pole_img) in enumerate(tqdm(zip(FD_img, images), total=len(images))):
    img = img.reshape([height, width])
    img = cv2.resize(img, dsize=(expand_by, expand_by), interpolation=cv2.INTER_AREA)
    img = np.repeat(img[:, :, None], 3, axis = 2)
    pole_img = cv2.resize(pole_img, dsize=(expand_by, expand_by), interpolation=cv2.INTER_AREA)
    img = np.hstack((pole_img, img))
    im = Image.fromarray(img)
    im.save(bd+'frame'+"{:012d}".format(i)+'.png')
    if i>999999999:
        break



# bd = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/final_video/final_small_100_frames/'
cd_to = """cd """ + '''"'''+bd+'''"'''
mp4_file_name = 'out12.mp4'
mp4_file_name = 'final_example_vid_divby_'+str(div_by)+'.mp4'
os.system(cd_to + """ && ffmpeg -y -framerate 15 -i frame%012d.png -c:v libx264 -crf 0 """ + mp4_file_name)


#below for quick time compatible
mp4_file_name = 'quicktime_compatible_final_example_vid_divby_'+str(div_by)+'.mp4'
os.system(cd_to + """ && ffmpeg -y -framerate 20 -i frame%012d.png -vcodec libx264 -pix_fmt yuv420p -lossless 1  """ + mp4_file_name)

