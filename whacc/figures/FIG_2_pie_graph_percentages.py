from whacc import utils
import h5py
import matplotlib.pyplot as plt
import numpy as np
def foo_rename(instr):#keep
    if isinstance(instr, list):
        for i, k in enumerate(instr):
            instr[i] = foo_rename(k)
        return instr
    else:
        return '/Volumes/GoogleDrive-114825029448473821206/My Drive' + instr.split('My Drive')[-1]


all_data = utils.load_obj(foo_rename('/content/gdrive/My Drive/colab_data2/all_data'))
main_mod = all_data[45]['info']

for k2 in ['train', 'val', 'test']:
    k = foo_rename(main_mod['h5_'+k2])
    exec(k2 + ' = "' + k + '"')
    print(k)

# training
383810 # aug - 3 border X 10
292217 # reg - 80 border (676027 - 383810)
# validation
163120 # aug - 3 border X 10
118734 # regular - 80 border (281854 - 163120)
# test
0      # aug
38884  #regular


inds = [383810, 163120, 0]
d = {'Training': [],'Validation': [],'Test': []}
keys = list(d.keys())
count = []
for i, k2 in enumerate(['train', 'val', 'test']):
    k = foo_rename(main_mod['h5_'+k2])
    with h5py.File(k, 'r') as h:
        L = len(h['labels'][:])
        count.append(L)
        d[keys[i]].append(np.sum(h['labels'][inds[i]:] != 0) / L) # reg touch
        d[keys[i]].append(np.sum(h['labels'][:inds[i]] != 0) / L) # aug touch
        d[keys[i]].append(np.sum(h['labels'][:inds[i]] == 0) / L) # aug no-touch
        d[keys[i]].append(np.sum(h['labels'][inds[i]:] == 0) / L) # reg no-touch

print(np.round(count, -3)) #array([676000, 282000,  39000])

import seaborn as sns
colors = np.asarray(sns.color_palette("Paired")[:4])
colors = colors[[1, 0, 2, 3,], :]
#Using matplotlib
labels = ['regular\ntouch', 'augmented\ntouch', 'augmented\nno-touch', 'regular\nno-touch']
pie, ax = plt.subplots(figsize=[10,6])
plt.pie(x=d['Training'], autopct="%.1f%%", explode=[0.02]*4, labels=labels, pctdistance=0.5, colors= colors)
plt.title("Training Data\n676k frames", fontsize=14)

pie, ax = plt.subplots(figsize=[10,6])
plt.pie(x=d['Validation'], autopct="%.1f%%", explode=[0.02]*4, labels=labels, pctdistance=0.5, colors= colors)
plt.title("Validation Data\n282k frames", fontsize=14)

pie, ax = plt.subplots(figsize=[10,6])
plt.pie(x=d['Test'][::3], autopct="%.1f%%", explode=[0.02]*2, labels=labels[::3], pctdistance=0.5, colors= colors[::3, :])
plt.title("Testing Data\n39k frames", fontsize=14)

#
# k = test
# utils.print_h5_keys(k)
#
#
# a1 = 0
# with h5py.File(k, 'r') as h:
#     print(h['labels'].shape)
#     imgs = h['images'][a1:a1+20]
# fig, ax = plt.subplots(5, 4)
# for i, a in enumerate(ax.flatten()):
#     a.imshow(imgs[i])


# # ind = int(281854//1.1)
# # a1 = 234878
# # a2 = 281854
# # a3 = int((a2-a1)/20)
# # with h5py.File(k, 'r') as h:
# #     print(h['labels'].shape)
# #     imgs = h['images'][a1:a2:a3]
# # fig, ax = plt.subplots(5, 4)
# # for i, a in enumerate(ax.flatten()):
# #     a.imshow(imgs[i])
# #
#
# #
# #
# # with h5py.File(k, 'r') as h:
# #     print(h['labels'].shape)
# #     imgs = h['images'][163120:163120+20]
# #     print(len(h['labels'][163120:]))
# # fig, ax = plt.subplots(5, 4)
# # for i, a in enumerate(ax.flatten()):
# #     a.imshow(imgs[i])
# #
# #
# #
# #
# #
# #
# #
# # utils.get_dict_info(main_mod)
# #
# #
# # utils.open_folder(foo_rename('/content/gdrive/My Drive/colab_data2/model_testing/all_data'))
