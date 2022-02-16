from whacc import image_tools, analysis, utils
import numpy as np
import matplotlib.pyplot as plt


f = '/Users/phil/Dropbox/Colab data/H5_data/ALT_LABELS_FINAL_PRED/AH0698_170601_PM0121_AAAA_ALT_LABELS.h5'
a = utils.print_h5_keys(f, return_list=True, do_print=True)

a2 = a[88]
print(a2)
pred = image_tools.get_h5_key_and_concatenate([f], a2)
pred = ((pred > .5) * 1).flatten()
real = image_tools.get_h5_key_and_concatenate([f], a[105])

frame_num_array = (np.ones(60) * 4000).astype(int)

a = analysis.error_analysis(real, pred, frame_num_array=frame_num_array)


print(set(a.all_error_type))
h5_img = '/Users/phil/Dropbox/Colab data/H5_data/regular/AH0698_170601_PM0121_AAAA_regular.h5'
pp = analysis.pole_plot(h5_img, pred_val=a.pred, true_val=a.real, figsize = [10, 5])


def display_error_types(pp, a, show_type, cnt = 0, edge_size = 2):
    certain_type = np.where(np.asarray(a.all_error_type) == show_type)[0]

    for k in range(len(a.all_error_type)):
        cnt+=1
        if cnt in certain_type:
            break
        else:
            print('ALL DONE')
            return None

    pp.current_frame = a.all_errors.copy()[cnt][0]-edge_size
    pp.len_plot = len(a.all_errors.copy()[cnt])+(edge_size*2)
    pp.plot_it()

    plt.plot([edge_size, edge_size], [-1, 10], 'g')
    plt.plot([pp.len_plot-edge_size-1, pp.len_plot-edge_size-1], [-1, 10], 'g')
    plt.title(a.all_error_type[cnt] + '   ' + str(a.all_error_nums[cnt]))
    print(cnt)
    return cnt

h5_img = '/Users/phil/Dropbox/Colab data/H5_data/regular/AH0698_170601_PM0121_AAAA_regular.h5'
pp = analysis.pole_plot(h5_img, pred_val=a.pred, true_val=a.real, figsize = [10, 5])
show_type = 'join'
cnt = -1

cnt = display_error_types(pp, a, show_type, cnt = cnt, edge_size = 2)
