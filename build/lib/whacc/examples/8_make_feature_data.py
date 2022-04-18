from whacc import utils, image_tools
from whacc.feature_maker import feature_maker, total_rolling_operation_h5_wrapper
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import h5py
import seaborn as sns
import os
import numpy as np
import cv2

# run quoted code below to make the original mini file
"""
## a) use this range [14537, 14607]
base_name = 'AH0407_160613_JC1003_AAAC_3lag.h5'
# /Users/phil/Dropbox/HIRES_LAB/curation_for_auto_curator/Data/Jon/AH0407/160613/AH0407x160609-144.mp4
img_h5 = utils.lister_it(utils.get_h5s('/Volumes/GoogleDrive-114825029448473821206/My Drive/Colab data/curation_for_auto_curator/DATA_FULL_in_range_only/'), base_name)
feature_h5 = utils.lister_it(utils.get_h5s('/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/FEATURE_DATA/colab data/curation_for_auto_curator/DATA_FULL_in_range_only/data_AH0407_160613_JC1003_AAAC/3lag/'), base_name)
inds = [14419, 16417]
with h5py.File(img_h5[0], 'r') as h:
    images = h['images'][inds[0]:inds[1]]
    labels = h['labels'][inds[0]:inds[1]]
    frame_nums = np.asarray([len(labels)])
with h5py.File(feature_h5[0], 'r') as h:
    FD__original = h['FD__original'][inds[0]:inds[1]]

with h5py.File('/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/example_feature.h5', 'w') as h:
    h['images'] = images
    h['labels'] = labels
    h['frame_nums'] = frame_nums
    h['FD__original'] = FD__original
    h['base_name'] = base_name.encode("ascii", "ignore")
    h['img_h5'] = img_h5[0].encode("ascii", "ignore")
    h['feature_h5'] = feature_h5[0].encode("ascii", "ignore")
"""
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

h5_in = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/example_feature.h5'

FM = feature_maker(h5_in, operational_key='FD__original', delete_if_exists=True)

for periods in tqdm([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]):
    data, key_name = FM.shift(periods, save_it=True)

for smooth_it_by in tqdm([3, 7, 11, 15, 21, 41, 61]):
    data, key_name = FM.rolling(smooth_it_by, 'mean', save_it=True)

for periods in tqdm([-50, -20, -10, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 10, 20, 50]):
    data, key_name = FM.operate('diff', kwargs={'periods': periods}, save_it=True)

for smooth_it_by in tqdm([3, 7, 11, 15, 21, 41, 61]):
    data, key_name = FM.rolling(smooth_it_by, 'std', save_it=True)

win = 1
# key_to_operate_on = 'FD__original'
op = np.std
mod_key_name = 'FD_TOTAL_std_' + str(win) +  '_of_'
all_keys = utils.lister_it(utils.print_h5_keys(FM.h5_in, 1, 0), 'FD__', 'FD_TOTAL')
for key_to_operate_on in tqdm(all_keys):
    data_out = total_rolling_operation_h5_wrapper(FM, win, op, key_to_operate_on, mod_key_name = mod_key_name, save_it = True)

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

# utils.print_h5_keys(h5_in)
with h5py.File(h5_in, 'r') as h:
    images = h['images'][:]
    labels = h['labels'][:]
    frame_nums = h['frame_nums'][:]
    base_name = h['base_name']
    FD__original = h['FD__original'][:]




utils.open_folder('/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM')

fps = 20
fourcc = cv2.VideoWriter_fourcc(*'XVID')
mp4_name = h5_in.replace('.h5', '.mp4')
width = 61
hieght = 61
channel = 3
video = cv2.VideoWriter(mp4_name, fourcc, float(fps), (width, hieght))
for img in images:
    img = np.stack(([img[:, :, 2], img[:, :, 1], img[:, :, 0]]), axis = 2) # resort for because order is reverse of numpy.imshow
    video.write(img)
video.release()


fps = 20
fourcc = cv2.VideoWriter_fourcc(*'XVID')
mp4_name = h5_in.replace('.h5', '_features.mp4')
width = 32
hieght = 64
channel = 1




FD_img = FD__original-np.min(FD__original)
FD_img = ((FD_img/np.max(FD_img))*255).astype(np.uint8)
cc = np.corrcoef(FD_img, rowvar=False)
cc, sorted_inds = cluster_corr(cc)
plt.figure()
sns.heatmap(cc)


FD_img = FD_img[:, sorted_inds]
# sorted_inds = np.argsort(np.mean(FD_img[labels==1, :], axis=0))
# FD_img = FD_img[:, sorted_inds]

# FD_img = np.vstack([FD_img[:, k] for k in sorted_inds]).T

for img in FD_img:
    # img = FD_img[k, :]
    img = img.reshape([hieght, width])
    img = np.repeat(img[:, :, None], 3, axis = 2)
    video.write(img)
video.release()
####################################
from pathlib import Path
from PIL import Image
expand_by = 1000
bd = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/example_feature_pngs_div_by_3/'
Path(bd).mkdir(parents=True, exist_ok=True)
# bd = '/Users/phil/Desktop/example_feature_pngs/'
for i, (img, pole_img) in enumerate(tqdm(zip(FD_img, images), total = len(images))):
    img = img.reshape([hieght, width])
    img = cv2.resize(img, dsize=(expand_by, expand_by), interpolation=cv2.INTER_AREA)
    img = np.repeat(img[:, :, None], 3, axis = 2)

    # pole_img = np.stack(([pole_img[:, :, 2], pole_img[:, :, 1], pole_img[:, :, 0]]), axis = 2) # resort for because order is reverse of numpy.imshow
    pole_img = cv2.resize(pole_img, dsize=(expand_by, expand_by), interpolation=cv2.INTER_AREA)
    img = np.hstack((pole_img, img))
    im = Image.fromarray(img)
    im.save(bd+'frame'+"{:08d}".format(i)+'.png')

cd_to = """cd """ + '''"'''+bd+'''"'''
mp4_file_name = 'out10.mp4'
os.system(cd_to + """ && ffmpeg -framerate 20 -i frame%08d.png -c:v libx264 -crf 0 """ + mp4_file_name)

mp4_file_name = 'out10_5fps.mp4'
os.system(cd_to + """ && ffmpeg -framerate 5 -i frame%08d.png -c:v libx264 -crf 0 """ + mp4_file_name)

#below for quick time compatible
mp4_file_name = 'movie2.mp4'
os.system(cd_to + """ && ffmpeg -framerate 20 -i frame%08d.png -vcodec libx264 -pix_fmt yuv420p -lossless 1  """ + mp4_file_name)

# /Users/phil/Dropbox/HIRES_LAB/curation_for_auto_curator/Data/Jon/AH0407/160613/AH0407x160609-144.mp4

####################################

from PIL import Image
expand_by = 1000
bd = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/example_feature_pngs/'
# bd = '/Users/phil/Desktop/example_feature_pngs/'
for i, img in enumerate(tqdm(FD_img)):
    img = img.reshape([hieght, width])
    img = cv2.resize(img, dsize=(expand_by, expand_by), interpolation=cv2.INTER_AREA)
    img = np.repeat(img[:, :, None], 3, axis = 2)
    im = Image.fromarray(img)
    im.save(bd+'frame'+"{:08d}".format(i)+'.png')

cd_to = """cd """ + '''"'''+bd+'''"'''
mp4_file_name
os.system(cd_to + """ && ffmpeg -framerate 20 -i frame%08d.png -c:v libx264 -crf 0 """ + mp4_file_name)


cd "/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/example_feature_pngs3"

ffmpeg -framerate 20 -i frame%08d.png -vcodec libx264 -pix_fmt yuv420p -lossless 1 movie.mp4
# plt.figure()
# sns.heatmap(tmp1)
# plt.figure()
# sns.heatmap(np.tile(tmp1, 10))
#
# # cd "/Users/phil/Desktop/example_feature_pngs/"
# # "/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/example_feature_pngs/"

ffmpeg -framerate 20 -i frame%08d.png -c:v libx264 -crf 0 output3.mp4

ffmpeg -framerate 20 -i frame%08d.png -c:v copy output.mkv

ffmpeg -framerate 20 -i frame%08d.png -c:v libvpx-vp9 -pix_fmt yuva420p -lossless 1 out.webm

####################################
#
# from PIL import Image
# from subprocess import Popen, PIPE
#
# fps, duration = 24, 100
# p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', '24', '-i', '-', '-vcodec', 'mpeg4', '-qscale', '1', '-r', '24', 'video.avi'], stdin=PIPE)
# tmp1 = ['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', '24', '-i', '-', '-vcodec', 'mpeg4', '-qscale', '1', '-r', '24', 'video.avi']
# # p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-c', 'v r10k', 'video.avi'], stdin=PIPE)
#
# for img in FD_img:
#     img = img.reshape([hieght, width])
#     img = np.repeat(img[:, :, None], 3, axis = 2)
#     im = Image.fromarray(img)
#     im.save(p.stdin, 'JPEG')
# p.stdin.close()
# p.wait()



with h5py.File('/Users/phil/Desktop/0006_cp.hdf5', 'r') as h:
    for k in h:
        try:
            print(type(h[k][:]))
        except:
            pass
