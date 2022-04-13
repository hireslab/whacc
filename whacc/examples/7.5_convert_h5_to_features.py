from whacc import image_tools,model_maker, feature_maker

h5_in = '/Users/phil/Desktop/pipeline_test/AH0407x160609_3lag.h5'
h5_FD = h5_in.replace('.h5', '_feature_data.h5')

mod = model_maker.load_final_model()
in_gen = image_tools.ImageBatchGenerator(500, h5_in, label_key='labels')
feature_maker.convert_h5_to_feature_h5(mod, in_gen, h5_FD)
