from whacc import utils
from whacc import image_tools

H5_list_to_train = utils.get_h5s('/Users/phil/Downloads/untitled folder 2/')
H5_list_to_train = utils.lister_it(H5_list_to_train,
                                   keep_strings=['subset'])  # get only the H5 files with the word 'subset'
print(H5_list_to_train)
bd = '/Users/phil/Downloads/untitled folder 2/'  # base directory to put files
split_h5_files = image_tools.split_h5(H5_list_to_train, [4, 1], temp_base_name=[bd + 'training', bd + 'validation'],
                                      add_numbers_to_name=False)



