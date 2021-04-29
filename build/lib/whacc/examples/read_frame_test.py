import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

cap = cv2.VideoCapture("/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH1131X26032020ses338_SAMSON/AH1131X26032020ses338-52.mp4")
img_to_load = "/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH1131X26032020ses338_SAMSON/template_img.png"
template_image = np.asarray(Image.open(img_to_load))
amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

frame_number = 2000

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
res, frame = cap.read()

gpu_frame = cv2.cuda_GpuMat()

res = cv2.matchTemplate(frame, template_image, 'cv2.TM_CCOEFF')












#
# plt.figure()
#
# plt.imshow(frame)
#
#
#
# frame_number = 1
#
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
# res, frame = cap.read()
# plt.figure()
# plt.imshow(frame)
#
#
#
# cap.set(cv2.CAP_PROP_POS_FRAMES, -9999)
#
#
#
#
# start_frame = 2000
#
# tmp1 = np.arange(start_frame, amount_of_frames)
# tmp2 = np.flip(np.arange(0, start_frame))
#
# tmp1
