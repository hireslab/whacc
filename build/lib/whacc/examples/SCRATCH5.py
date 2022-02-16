from whacc.image_tools import *
import copy
import pdb


class ImageBatchGenerator_feature_array(keras.utils.Sequence):

    def __init__(self, time_length, batch_size, h5_file_list, label_key='labels', feature_len=None,
                 label_index_to_lstm_len=None, edge_value=-1, remove_any_time_points_with_edges = True):
        """
        
        Parameters
        ----------
        time_length : total time points
        batch_size : batch output for generator 
        h5_file_list : list of h5 strings or single h5 string 
        label_key : where y output comes from 
        feature_len : length of the features per time point
        label_index_to_lstm_len : determines look back and look forward index refers to where the 'current' time point is
        within the range of look_back_len; e.g. look_back_len = 7 label_index_to_lstm_len = 3 (middle index of 7) then 
        time point 0 will be at 3 and index 0, 1, 2 will be the past values and index 4, 5, 6 will be the future values.
        look_back_len = 7 label_index_to_lstm_len = 0 (first index) then current time point will be at index 0 and all
        other time point (1, 2, 3, 4, 5, 6) will be future values. Default is middle time point
        edge_value : what to replace the edge values with, when time shifting you will have edges with no value, this
        will replace those values with this number.
        remove_any_time_points_with_edges : if true then batch size will not be the actual batch size it will be batch
        size - the number of time points with edges in them, x and y will still match and this method is preferred for
        training due to it not including unknown values.
        """
        assert time_length % 2 == 1, "number of images must be odd"
        if label_index_to_lstm_len is None:
            label_index_to_lstm_len = time_length // 2  # in the middle
        h5_file_list = utils.make_list(h5_file_list, suppress_warning=True)
        num_frames_in_all_H5_files = get_total_frame_count(h5_file_list)
        file_inds_for_H5_extraction = batch_size_file_ind_selector(
            num_frames_in_all_H5_files, batch_size)
        subtract_for_index = reset_to_first_frame_for_each_file_ind(
            file_inds_for_H5_extraction)
        self.remove_any_time_points_with_edges = remove_any_time_points_with_edges
        self.label_key = label_key
        self.batch_size = batch_size
        self.H5_file_list = h5_file_list
        self.num_frames_in_all_H5_files = num_frames_in_all_H5_files
        self.file_inds_for_H5_extraction = file_inds_for_H5_extraction
        self.subtract_for_index = subtract_for_index
        self.label_index_to_lstm_len = label_index_to_lstm_len
        self.lstm_len = time_length
        self.feature_len = feature_len
        self.edge_value = edge_value
        if remove_any_time_points_with_edges:
            self.edge_value = np.nan
            print('remove_any_time_points_with_edges == True : forcing edge_value to np.nan to aid in removing these time points')


        self.get_frame_edges()
        # self.full_edges_mask = self.full_edges_mask - (self.lstm_len // 2 - self.label_index_to_lstm_len)

    def __getitem__(self, num_2_extract):
        h = self.H5_file_list
        i = self.file_inds_for_H5_extraction
        all_edges = self.all_edges_list[np.int(i[num_2_extract])]
        H5_file = h[np.int(i[num_2_extract])]
        num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]

        with h5py.File(H5_file, 'r') as h:
            b = self.lstm_len // 2
            tot_len = h['images'].shape[0]

            # assert tot_len - b > self.batch_size, "reduce batch size to be less than total length of images minus floor(lstm_len) - 1, MAX->" + str(
            #     tot_len - b - 1)

            i1 = num_2_extract_mod * self.batch_size - b
            i2 = num_2_extract_mod * self.batch_size + self.batch_size + b
            edge_left_trigger = abs(min(i1, 0))
            edge_right_trigger = min(abs(min(tot_len - i2, 0)), b)
            x = h['images'][max(i1, 0):min(i2, tot_len)]
            if edge_left_trigger + edge_right_trigger > 0:  # in case of edge cases
                pad_shape = list(x.shape)
                pad_shape[0] = edge_left_trigger + edge_right_trigger
                pad = np.zeros(pad_shape).astype('float32')
                if edge_left_trigger > edge_right_trigger:
                    x = np.concatenate((pad, x), axis=0)
                else:
                    x = np.concatenate((x, pad), axis=0)

            s = list(x.shape)
            s.insert(1, self.lstm_len)
            out = np.zeros(s).astype('float32')  # before was uint8
            Z = self.label_index_to_lstm_len - self.lstm_len // 2
            for i in range(self.lstm_len):
                i_temp = i
                i = i - Z
                i1 = max(0, b - i)
                i2 = min(s[0], s[0] + b - i)
                i3 = max(0, i - b)
                i4 = min(s[0], s[0] + i - b)
                # print('take ', i3, ' to ', i4, ' and place in ', i1, ' to ', i2)
                out[i1:i2, i_temp, ...] = x[i3:i4, ...]

            out = out[b:s[0] - b, ...]
            i1 = num_2_extract_mod * self.batch_size
            i2 = num_2_extract_mod * self.batch_size + self.batch_size
            raw_Y = h[self.label_key][i1:i2]

            adjust_these_edge_frames = np.intersect1d(all_edges.flatten(), np.arange(i1, i2))
            b2 = b - self.label_index_to_lstm_len  # used to adjust mask postion based on where the center value is
            for atef in adjust_these_edge_frames:
                mask_ind = np.where(atef == all_edges)[1][0]
                mask_ind = mask_ind - b2
                mask_ind = mask_ind % self.full_edges_mask.shape[0]  # wrap around index

                mask_ = self.full_edges_mask[mask_ind]
                mask_ = mask_ == 1
                out_ind = atef + i1 - b2
                out_ind = out_ind % out.shape[0]  # wrap around index
                out[out_ind][mask_] = self.edge_value


            s = out.shape
            out = np.reshape(out, (s[0], s[1] * s[2]))
            if self.remove_any_time_points_with_edges:
                keep_inds = ~np.isnan(np.mean(out, axis = 1))
                out = out[keep_inds]
                raw_Y = raw_Y[keep_inds]

            return out, raw_Y

    def __len__(self):
        return len(self.file_inds_for_H5_extraction)

    def getXandY(self, num_2_extract):
        """

        Parameters
        ----------
        num_2_extract :


        Returns
        -------

        """
        rgb_tensor, raw_Y = self.__getitem__(num_2_extract)
        return rgb_tensor, raw_Y

    def image_transform(self, raw_X):
        """input num_of_images x H x W, image input must be grayscale
        MobileNetV2 requires certain image dimensions
        We use N x 61 x 61 formated images
        self.IMG_SIZE is a single number to change the images into, images must be square

        Parameters
        ----------
        raw_X :


        Returns
        -------


        """
        # kept this cause this is the format of the image generators I know this is redundant
        rgb_batch = copy.deepcopy(raw_X)
        rgb_tensor = rgb_batch
        self.IMG_SHAPE = (self.feature_len)
        return rgb_tensor

    def get_frame_edges(self):
        self.all_edges_list = []
        b = self.lstm_len // 2

        s = [b * 2, self.lstm_len, self.feature_len]
        for H5_file in self.H5_file_list:
            with h5py.File(H5_file, 'r') as h:
                full_edges_mask = np.ones(s)
                tmp1 = np.arange(1, self.lstm_len)
                front_edge = tmp1[:self.label_index_to_lstm_len]
                back_edge = tmp1[:self.lstm_len - self.label_index_to_lstm_len - 1]

                edge_ind = np.flip(front_edge)
                for i in front_edge:
                    # print(i - 1, ':', edge_ind[i - 1])
                    # print(full_edges_mask[i - 1, :edge_ind[i - 1], ...].shape)
                    # print('\n')
                    full_edges_mask[i - 1, :edge_ind[i - 1], ...] = np.zeros_like(
                        full_edges_mask[i - 1, :edge_ind[i - 1], ...])

                edge_ind = np.flip(back_edge)
                for i in back_edge:
                    # print(-i, -edge_ind[i - 1], ':')
                    # print(full_edges_mask[-i, -edge_ind[i - 1]:, ...].shape)
                    # print('\n')
                    full_edges_mask[-i, -edge_ind[i - 1]:, ...] = np.zeros_like(
                        full_edges_mask[-i, -edge_ind[i - 1]:, ...])

                all_edges = []
                for i1, i2 in utils.loop_segments(h['frame_nums']):  # 0, 1, 3998, 3999 ; 4000, 4001, 7998, 7999; ...
                    edges = (np.asarray([[i1], [i2 - b]]) + np.arange(0, b).T).flatten()
                    all_edges.append(edges)

                all_edges = np.asarray(all_edges)
            self.all_edges_list.append(all_edges)
            # pdb.set_trace()
            full_edges_mask = full_edges_mask.astype(int)
            self.full_edges_mask = full_edges_mask == 0


h5 = '/Volumes/GoogleDrive-114825029448473821206/My Drive/colab_data2/model_testing_features_data/feature_data/regular_80_border/DATA_FULL/3lag/holy_test_set_10_percent_3lag.h5'
# h5 = '/Users/phil/Desktop/temp.h5'
a = ImageBatchGenerator_feature_array(7, 4000, h5, label_key='labels', feature_len=2048,
                                      label_index_to_lstm_len=3, edge_value=-1, remove_any_time_points_with_edges = False)
x, y = a.__getitem__(0)
# # utils.np_stats(x)
# # s = x.shape;
# # x2 = np.reshape(x, (s[0], s[1] * s[2]))


plt.figure()
plt.plot(y[2200:2300])

plt.figure()
plt.imshow(x[2200:2300, ::512].T)


