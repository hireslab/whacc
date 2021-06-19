import h5py
import numpy as numpy
import matplotlib.pyplot as plt
from whacc.utils import *


class subset_h5_generator:
    """ """
    def __init__(self, h5_img_file, label_key):
        self.h5_img_file = h5_img_file
        self.label_key = label_key
        with h5py.File(self.h5_img_file, 'r') as F:
            self.labels = F[label_key][:]

    def save_subset_h5_file(self, file_save_dir = None, save_name = None):
        """

        Parameters
        ----------
        file_save_name :
             (Default value = None)

        Returns
        -------

        """
        if save_name is None:
            save_name = os.path.basename(self.h5_img_file).split('.h5')[0] + '_subset.h5'
        if file_save_dir is None:
            file_save_dir = os.path.dirname(self.h5_img_file)
        file_save_name = file_save_dir + os.path.sep + save_name
        all_labels = self.labels
        all_inds = self.all_inds
        try:
            os.remove(file_save_name)
            print('Deleting file with same name')
        except:
            pass
        with h5py.File(self.h5_img_file, 'r') as F:
            images = numpy.asarray([F['images'][k] for k in all_inds])
            labels = numpy.asarray([all_labels[k] for k in all_inds.astype('int')])
            in_range = numpy.asarray([F['in_range'][k] for k in all_inds])
            with h5py.File(file_save_name, 'w') as hf:  # auto close in case of failure using 'with'
                hf.create_dataset('images', data=images)
                hf.create_dataset('labels', data=labels)
                hf.create_dataset('in_range', data=in_range)
                hf.create_dataset('all_inds', data=all_inds)
                hf.create_dataset('retrain_H5_info', data=[str(self.retrain_H5_info).encode("ascii", "ignore")])
                hf.close()
                print('finished with ' + file_save_name)

    def plot_pole_grab(self, im_stack, fig_size=None):
        """

        Parameters
        ----------
        im_stack :
            
        fig_size :
             (Default value = None)

        Returns
        -------

        """
        if fig_size is None:
            width_plt = self.retrain_H5_info['num_high_prob_past_max_y'] + self.retrain_H5_info['seg_len_look_dist'] * 2
            plt.figure(figsize=[width_plt, 5])
        else:
            plt.figure(figsize=fig_size)
        _ = plt.imshow(im_stack)

    def keep_only_pole_up_times(self, start_pole, stop_pole):
        """

        Parameters
        ----------
        start_pole :
            
        stop_pole :
            

        Returns
        -------

        """
        with h5py.File(self.h5_img_file, 'r') as hf:
            cumsum_frames = np.concatenate((np.asarray([0]), np.cumsum(hf['trial_nums_and_frame_nums'][1, :])))
            tot_frames = np.sum(hf['trial_nums_and_frame_nums'][1, :])

        b = np.vstack((start_pole + cumsum_frames[:-1], cumsum_frames[1:] - 1)).astype('int')
        b = np.min(b, axis=0)
        a = np.vstack((stop_pole + cumsum_frames[:-1], cumsum_frames[1:])).astype('int')
        a = np.min(a, axis=0)

        keep_mask = np.zeros(tot_frames.astype('int'))
        for k1, k2 in zip(b, a):
            keep_mask[k1:k2] = 1
        return keep_mask

    def get_example_segments(self,
                             seg_len_before_touch=10,
                             seg_len_after_touch=10,
                             min_y=.2,
                             max_y=.8,
                             num_to_sample=40,
                             min_seg_size=6,
                             start_and_stop_pole_times=None):
        """

        Parameters
        ----------
        seg_len_before_touch :
             (Default value = 10)
        seg_len_after_touch :
             (Default value = 10)
        min_y :
             (Default value = .2)
        max_y :
             (Default value = .8)
        num_to_sample :
             (Default value = 40)
        min_seg_size :
             (Default value = 6)
        start_and_stop_pole_times :
             (Default value = None)

        Returns
        -------

        """
        labels = self.labels
        if start_and_stop_pole_times is not None:
            pole_mask = self.keep_only_pole_up_times(start_pole=start_and_stop_pole_times[0],
                                                     stop_pole=start_and_stop_pole_times[1])
            labels = labels * pole_mask
        segs = numpy.where(labels > min_y)[0]
        chunks_tmp, chunk_inds = group_consecutives(segs, step=1)
        # print(len(chunks))
        chunks = []
        for i, k in enumerate(chunks_tmp):
            if len(k) >= min_seg_size:
                chunks.append(chunks_tmp[i])
                # print('popped it')
        # print(len(chunks))
        self.chunks = chunks  ###remove
        good_up_segs = []
        good_down_segs = []
        for k in chunks:
            try:
                up = labels[k[0] - seg_len_before_touch:k[0]]
                down = labels[k[-1] - 1:k[-1] + seg_len_before_touch - 1]
                assert (len(up) == seg_len_before_touch)
                assert (len(down) == seg_len_before_touch)
                good_up_segs.append(numpy.min(up) <= min_y)
                good_down_segs.append(numpy.min(down) <= min_y)
            except:  # if we are on the edges just toss these
                good_up_segs.append(False)
                good_down_segs.append(False)
        good_up_segs = numpy.where(good_up_segs)[0]
        good_down_segs = numpy.where(good_down_segs)[0]
        a = numpy.random.choice(good_up_segs, size=num_to_sample, replace=False)
        up_start = [chunks[k][0] + 1 for k in a]

        a = numpy.random.choice(good_down_segs, size=num_to_sample, replace=False)
        down_start = [chunks[k][-1] + 1 for k in a]

        all_inds = numpy.asarray([])
        onset_list = []
        offset_list = []
        for k1, k2 in zip(up_start, down_start):
            onset_list.append(numpy.asarray(range(k1 - seg_len_before_touch, k1 + seg_len_after_touch)).astype('int'))
            all_inds = numpy.concatenate((all_inds, onset_list[-1]))
            offset_list.append(numpy.asarray(range(k2 - 1 - seg_len_after_touch, k2 - 1 + seg_len_before_touch)).astype('int'))
            all_inds = numpy.concatenate((all_inds, offset_list[-1]))
        retrain_H5_info = {'seg_len_look_dist': seg_len_before_touch,
                           'min_y': min_y,
                           'max_y': max_y,
                           'num_to_sample': num_to_sample,
                           'num_high_prob_past_max_y': seg_len_after_touch}
        # inds_2_add = numpy.linspace(0, 50 - 1, ).astype(int)
        # all_inds = numpy.concatenate((numpy.float64(inds_2_add), all_inds))
        self.all_inds = all_inds.astype('int')
        self.onset_list = onset_list
        self.offset_list = offset_list
        self.retrain_H5_info = retrain_H5_info
        # return all_inds, onset_list, offset_list, retrain_H5_info

    def get_img_stack(self, frame_inds, h5_img_key='images'):
        """

        Parameters
        ----------
        frame_inds :
            
        h5_img_key :
             (Default value = 'images')

        Returns
        -------

        """

        im_stack = None
        for k in frame_inds:
            with h5py.File(self.h5_img_file, 'r') as h:
                tmp_img = numpy.asarray(h[h5_img_key][k])
                if im_stack is None:
                    im_stack = tmp_img
                else:
                    im_stack = numpy.hstack((im_stack, tmp_img))
        return im_stack

    def plot_all_onset_or_offset(self, up_or_down_list, fig_size=None, h5_img_file=None):
        """

        Parameters
        ----------
        up_or_down_list :
            
        fig_size :
             (Default value = None)
        h5_img_file :
             (Default value = None)

        Returns
        -------

        """
        if h5_img_file is None:
            h5_img_file = self.h5_img_file

        for i, k in enumerate(up_or_down_list):
            if i == 0:
                im_stack = self.get_img_stack(k)
            else:
                im_stack = numpy.vstack((im_stack, self.get_img_stack(k)))
        self.plot_pole_grab(im_stack, fig_size)
