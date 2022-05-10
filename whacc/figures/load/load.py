from whacc import utils
from whacc.utils import info
import h5py
from whacc import model_maker, image_tools
import numpy as np


def make_example_h5():
    h5_in = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing/all_data/all_models/regular_80_border_aug_0_to_9/data/3lag/train_3lag.h5'
    h5_new = '/Users/phil/Desktop/feature_example2.h5'
    utils.print_h5_keys(h5_in)
    with h5py.File(h5_in, 'r') as h:
        with h5py.File(h5_new, 'w') as h2:
            h2['images'] = h['images'][655102:656252]
            h2['labels'] = h['labels'][655102:656252]
            h2['frame_nums'] = [len(h2['labels'][:])]

    # convert to feature data
    # h5_3lag = '/Users/phil/Desktop/temp_dir/AH0407x160609_3lag.h5'
    h5_feature_data = h5_new.replace('.h5', '_feature_data.h5')
    in_gen = image_tools.ImageBatchGenerator(500, h5_new, label_key='labels')
    RESNET_MODEL = model_maker.load_final_model()
    utils.convert_h5_to_feature_h5(RESNET_MODEL, in_gen, h5_feature_data)

    # h5_feature_data = '/Users/phil/Desktop/temp_dir/AH0407x160609_3lag_feature_data.h5'
    # generate all the modified features (41*2048)+41 = 84,009
    utils.standard_feature_generation(h5_feature_data)
    return h5_feature_data


def get_example_h5_selected_2105_features(h5_feature_data=None):
    if h5_feature_data is None:
        h5_feature_data = '/Users/phil/Desktop/feature_example2_feature_data.h5'
    d = utils.load_feature_data()
    feature_index = d['final_selected_features_bool']

    with h5py.File(h5_feature_data, 'r') as h:
        all_x = []
        all_y = np.asarray(h['labels'][:])
        for k in d['feature_list_unaltered']:
            x = h[k][:]
            if len(x.shape) == 1:
                x = x[:, None]
            all_x.append(x)
    all_x = np.hstack(all_x)
    all_x = all_x[:, feature_index]
    return all_x, all_y


all_x, all_y = get_example_h5_selected_2105_features()


