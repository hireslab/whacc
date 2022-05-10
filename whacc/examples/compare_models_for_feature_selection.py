from natsort import natsorted, ns
import matplotlib.pyplot as plt
import h5py
import numpy as np
from whacc import utils, image_tools
import os
test_set_to = 10
from natsort import natsorted, ns

mod_dir = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/models/10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V1/'
mod_dir = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/models/10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V2/'
# mod_dir = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/models/10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3/'
mod_dir = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/models/10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3_num2/'
# mod_dir = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/models/10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3_num3/'
mod_dir = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/models/10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3_num2_v4_num1/'
mod_dir = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/models/10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3_num2_v4_num2/'
mod_dir = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/models/10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3_num2_v4_num3/'
mod_dir = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/models/10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3_num2_v4_num4/'


def foo1(mod_dir):
    mods = utils.get_files(mod_dir, '*')
    mods = natsorted(mods, alg=ns.REAL)[:test_set_to]
    split_impotance = []
    gain_importance = []
    for k in mods:
        lgbm = utils.load_obj(k)
        split_impotance.append(lgbm.feature_importance(importance_type='split'))
        gain_importance.append(lgbm.feature_importance(importance_type='gain'))
    return np.asarray(split_impotance), np.asarray(gain_importance)


split_impotance, gain_importance = foo1(mod_dir)

features_out_of_10 = np.sum(np.asarray(split_impotance>0), axis = 0)
total_times_used = np.sum(np.asarray(split_impotance), axis = 0)
mean_gain = np.mean(gain_importance, axis = 0)
max_gain = np.max(gain_importance, axis = 0)
min_gain = np.min(gain_importance, axis = 0)

for k in range(10):
    print(np.round(100 * np.mean(features_out_of_10 > k), 2), '% have ', k+1, ' or more')
    print('count ', np.sum(features_out_of_10 > k))


####
next_set_of_features = np.any(gain_importance>2, axis = 0)
np.mean(next_set_of_features), np.sum(next_set_of_features), len(next_set_of_features)

###
#  remove these features
condition_1 = features_out_of_10<=444444# must be used by at least 2 models out of 10 (choosing the inverse)
np.mean(condition_1)

condition_2 = mean_gain<=2
np.mean(condition_2)

all_conditions = (condition_1*1 + condition_2*1) == 2
np.mean(all_conditions)


next_set_of_features = np.invert(all_conditions)
np.sum(next_set_of_features), len(next_set_of_features)

bd = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/feature_index_for_feature_selection'
# set1 = np.load(bd+os.sep+'non_zero_features_bool_29913_features.npy')
# set1 = np.load(bd+os.sep+'used_more_than_once_mean_gain_mre_than_0dot1_features_bool_18445_features.npy')
# set1 = np.load(bd+os.sep+'used_more_than_NONE_times_mean_gain_mre_than_1_features_bool_6390_features.npy')
# set1 = np.load(bd+os.sep+'only_gain_more_than_2_features_bool_4353_features.npy')
# set1 = np.load(bd+os.sep+'only_gain_more_than_2_features_bool_3624_features.npy')
set1 = np.load(bd+os.sep+'only_gain_more_than_2_features_bool_2105_features.npy')

set1_inds = np.where(set1)[0]
assert len(set1_inds)==len(all_conditions), 'wrong stuff is loaded'


next_set_of_features_inds = set1_inds[next_set_of_features]

save_feature_inds_bool = [k in next_set_of_features_inds for k in range(len(set1))]

# np.save(bd+os.sep+'used_more_than_once_mean_gain_mre_than_0dot1_features_bool_18445_features.npy', save_feature_inds_bool)
np.save(bd+os.sep+'only_gain_more_than_2_features_bool_2073_features.npy', save_feature_inds_bool)


"""
##################################################################################################
##################################################################################################
"""


d = '/Users/phil/Dropbox/HIRES_LAB/GitHub/whacc/whacc/whacc_data/feature_data/feature_data_dict.pkl'
d = utils.load_obj(d)
utils.get_dict_info(d)
x = np.asarray(d['full_feature_names'])
features_used_name  = x[d['final_selected_features_bool']]
x = np.unique(features_used_name)
x = natsorted(x, alg=ns.REAL)
for k in x:
    print(k)
    print(np.sum(k==features_used_name))


tmp1, inds = utils.lister_it(x, keep_strings='TOTAL', return_bool_index=True)
for k in np.where(inds)[0]:
    print(tmp1[k])
    print(mean_gain[k])


tmp2 = utils.lister_it(x, remove_string='TOTAL')

for k in tmp1:
    kk = k.replace('FD_TOTAL_std_1_of_', '')
    kk in tmp2

"""
find the max I need to make
"""
# I need to make at least 10 total feature

tmp3 = []
for k in tmp1:
    tmp3.append(k.replace('FD_TOTAL_std_1_of_', ''))


for k in tmp2:
    if k not in tmp3:
        print(k)
        print(np.sum(k==features_used_name))
    else:
        print('++++++++++++++++++++')
        print(k)
        print(np.sum(k==features_used_name))
        print('____________________')



# for k in tmp1['feature_list_short_type']:
#     print(k)
"""
##################################################################################################
##################################################################################################
"""

plt.hist(mean_gain, bins = np.linspace(0, 40, 100))
# plt.yscale('log')
"""
##################################################################################################
##################################################################################################
"""

mod_dir = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/models/'
all_mods = []
# os.listdir(mod_dir)
folder_names = ['top_features_greter_than_3_out_of_10_3095_features',
                '10_sets_frame_nums_split_then_random_split_method_numpy_ORIGINAL_2048_just_grabbed_the_last_2048',
                 '10_sets_frame_nums_split_then_random_split_method',
                 '10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V1',
                 '10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V2',
                 '10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3',
                '10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3_num2',
                '10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3_num3',
                '10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3_num4',
                '10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3_num5',
                '10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3_num2_v4_num1',
                '10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3_num2_v4_num2',
                '10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3_num2_v4_num3',
                '10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3_num2_v4_num4',
                '10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3_num2_v4_num5']

for d in folder_names:
    x = utils.get_files(os.path.join(mod_dir, d), '*.pkl')
    x = natsorted(x, alg=ns.REAL)
    all_mods.append(np.asarray(x))

L = len(folder_names)+2
ind = np.arange(L)  # the x locations for the groups
width = 1/L

folder_names = ['OLD METHOD 3095 features OVERFIT',
                'Original 2,048',
                'full 84,009 features (gain plus count filter)_1',
                '29,913 features (gain plus count filter)_2',
                '18,445 features (gain plus count filter)_3',
                '6,730 features (gain plus count filter)_4.1',
                '6,390 features (only gain filter)_4.2',
                '10,126 features (only gain filter)_4.3',
                '12,481 features (gain ANY greater than filter)_4.4',
                '15,019 features (only gain filter 0.1)_4.5',
                '4,353 feature (only gain >2)',
                '3,624 feature (only gain >2)',
                '3,029 feature (only gain >2)',
                '2,105 feature (only gain >1)',
                '2,073 feature (only gain >2)',]

fig, ax = plt.subplots()
for i, d in enumerate(all_mods):
    y = []
    for k in d:
        m = utils.load_obj(k)
        y.append(m.best_score['valid_0']['auc'])
    y.append(np.mean(y))
    ax.bar(np.arange(len(y))+width*i-.35, 1-np.asarray(y), width, label = folder_names[i])
    # plt.plot(1-np.asarray(y), '.', label = tmp1[i])

plt.legend(loc=1)
plt.ylabel('1 - AUC')
plt.title('feature selection models final AUC for each subset ')
# plt.ylim([0, 0.002])


x = np.arange(11).astype(int)
ax.set_xticks(x)
new_labels = ['mod '+str(k+1) for k in range(11)]
new_labels[-1] = 'mean models'
ax.set_xticklabels(new_labels)



"""
##################################################################################################
##################################################################################################
################## below I tried using a clustering algo to selec features #######################
####################### it didnt work very well so I ditched it ##################################
"""

to_cluster = np.vstack([features_out_of_10, total_times_used, mean_gain, max_gain, min_gain])



from sklearn import cluster
import copy
import seaborn as sns
X = copy.deepcopy(to_cluster.T)
model = cluster.KMeans(n_clusters=2, random_state=0)
model.fit(X)
utils.get_class_info(model)

tmp1 = cluster.AgglomerativeClustering().fit(X)
utils.get_class_info(tmp1)

plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')


from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
X = copy.deepcopy(to_cluster.T)
X = StandardScaler().fit_transform(X)
tsne = TSNE(n_components = 2, perplexity = 400, learning_rate = 100, early_exaggeration = 40, n_iter=250)
X_embedded = tsne.fit_transform(X)
utils.get_class_info(X_embedded)
sns.scatterplot(X_embedded[:,0], X_embedded[:,1])


agg_mod = cluster.AgglomerativeClustering(n_clusters = 5).fit(X_embedded)
agg_mod.labels_

X = copy.deepcopy(to_cluster.T)
agg_mod = cluster.AgglomerativeClustering(n_clusters = 50).fit(X)
labels = agg_mod.labels_
for k in np.unique(labels):
    if np.sum(k == labels)>10:
        print(np.sum(k == labels))
        print(np.mean(features_out_of_10[k == labels]))
        print(np.mean(total_times_used[k == labels]))
        print(np.mean(mean_gain[k == labels]))
        print(np.mean(max_gain[k == labels]))
        print(np.mean(min_gain[k == labels]))
        print('___________')
features_out_of_10 = np.sum(np.asarray(split_impotance>0), axis = 0)
total_times_used = np.sum(np.asarray(split_impotance), axis = 0)
mean_gain = np.mean(gain_importance, axis = 0)
max_gain = np.max(gain_importance, axis = 0)
min_gain = np.min(gain_importance, axis = 0)


model = cluster.KMeans(n_clusters=5)
# fit the model
model.fit(X_embedded)
# assign a cluster to each example
yhat = model.predict(X_embedded)


sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=yhat)


import sklearn



gm = sklearn.mixture.GaussianMixture(n_components=10, random_state=0).fit(X_embedded)
yhat = gm.predict(X_embedded)




import pandas as pd
fig, ax = plt.subplots()
groups = pd.DataFrame(X_embedded, columns=['x', 'y']).assign(category=y).groupby('category')
for name, points in groups:
    ax.scatter(points.x, points.y, label=name)

ax.legend()




yhat = model.predict(X)
# retrieve unique clusters
clusters = np.unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = np.where(yhat == cluster)
	# create scatter of these samples
	plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
plt.show()


# utils.get_class_info(tmp1)
# k-means clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = KMeans(n_clusters=2)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()




# gaussian mixture clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = GaussianMixture(n_components=2)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

x = StandardScaler().fit_transform(X)

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(principalComponents[:, 0], principalComponents[:, 1], principalComponents[:, 2])


finalDf = pd.concat([principalDf, pd.df[['target']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()





fn = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_batch_processing_test/add_files_here_FINISHED/Jon/AH0407/160613/AH0407x160609_final_to_combine_1_to_5_of_5.h5'
fn = '/Users/phil/Desktop/pipeline_test/data_FINISHED/AH0407x160609_final_to_combine_1_to_10_of_10.h5'
fn = '/Users/phil/Desktop/feature_example2_feature_data.h5'
fn = '/Users/phil/Desktop/pipeline_test/data_FINISHED/AH0407x160609_final_to_combine_1_to_2_of_10.h5'
from whacc import utils, image_tools
import matplotlib.pyplot as plt
utils.print_h5_keys(fn)

d = utils.load_feature_data()

all_x = []
for k in d['feature_list_unaltered']:
    with h5py.File(fn, 'r') as h:
        x = h[k][:]
        if len(x.shape) == 1:
            x = x[:, None]
        all_x.append(x)
all_x = np.hstack(all_x)
X = all_x[:, d['final_selected_features_bool']]



X = image_tools.get_h5_key_and_concatenate(fn, 'final_features_2105')
labels = image_tools.get_h5_key_and_concatenate(fn, 'labels')

mod_dir = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/models/10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V1/'
mod_dir = '/Volumes/GoogleDrive-114825029448473821206/My Drive/LIGHT_GBM/models/10_sets_frame_nums_split_then_random_split_method_numpy_feature_selection_V3_num2_v4_num4/'

mods = utils.get_files(mod_dir, '*')
# mods = natsorted(mods, alg=ns.REAL)[:test_set_to]
lgbm = utils.load_obj(mods[1])
# lgbm_shap = lgbm.predict(X, pred_contrib=True)


y_hat = lgbm.predict(X)
plt.plot(y_hat)

plt.plot(labels)


"""
clearly the files being loaded are not the same as the files being trained on
"""



arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
np.delete(arr, 1, 0), arr
