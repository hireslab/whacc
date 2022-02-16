



def rolling_mean(self, window, center = None, min_periods = None):
    window, center, min_periods, add_name_list, data_frame = self._rolling_(self, window, center = center, min_periods = min_periods)
    add_name_str = 'mean'.join(add_name_list)
    for i1, i2 in utils.loop_segments(self.frame_nums):
        df_rolling = self.data[i1:i2].rolling(window=window, min_periods=min_periods)
        data_frame[i1:i2] = df_rolling.mean().shift(-(center)).astype(np.float32)
    return data_frame, add_name_str



skew
sum
validate
sem             type->   method                       ...dow=3,min_periods=3,center=True,axis=0]>
quantile        type->   method                       ...dow=3,min_periods=3,center=True,axis=0]>
var             type->   method                       ...dow=3,min_periods=3,center=True,axis=0]>
std             type->   method                       ...dow=3,min_periods=3,center=True,axis=0]>
agg             type->   method                       ...dow=3,min_periods=3,center=True,axis=0]>
mean            type->   method                       ...dow=3,min_periods=3,center=True,axis=0]>
max             type->   method                       ...dow=3,min_periods=3,center=True,axis=0]>
kurt            type->   method                       ...dow=3,min_periods=3,center=True,axis=0]>
cov             type->   method                       ...dow=3,min_periods=3,center=True,axis=0]>
count           type->   method                       ...dow=3,min_periods=3,center=True,axis=0]>
corr            type->   method                       ...dow=3,min_periods=3,center=True,axis=0]>
apply           type->   method                       ...dow=3,min_periods=3,center=True,axis=0]>
aggregate       type->   method                       ...dow=3,min_periods=3,center=True,axis=0]>
min             type->   method                       ...dow=3,min_periods=3,center=True,axis=0]>
median          type->   method                       ...dow=3,min_periods=3,center=True,axis=0]>

tmp1 = pd.DataFrame(np.random.rand(10, 100))

df_rolling = tmp1.rolling(window=3, min_periods=3, )
utils.get_class_info(df_rolling)
