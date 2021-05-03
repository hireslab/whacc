H5_file_name = '/Users/phil/Dropbox/Autocurator/data/samsons_subsets/train_and_validate/AH1131X25032020ses335_subset.h5'
label_key = 'labels'

H5_file_name = '/Users/phil/Dropbox/Autocurator/data/samsons_subsets/testing/sladkfjalksdjflkasdklf.h5'


from whacc.touch_curation_GUI import touch_gui


touch_gui(H5_file_name, label_key, 'labels')

import h5py
import numpy as np
with h5py.File(H5_file_name, 'r') as h:
    print(np.unique(h['labels'][:]))
    print(np.where(h['labels'][:]==-1))
