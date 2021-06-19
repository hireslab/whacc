# H5_file_name = '/Users/phil/Dropbox/Autocurator/data/samsons_subsets/training_and_validation/AH1131X25032020ses335_subset.h5'
H5_file_name = '/Users/phil/Downloads/AH0667x170317_subset.h5'
label_read_key = 'labels'
label_write_key = 'labels'
from whacc.touch_curation_GUI import touch_gui
touch_gui(H5_file_name, label_read_key, label_write_key)



