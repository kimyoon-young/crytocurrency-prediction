import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import h5py

class PastSampler:
    '''
    Forms training samples for predicting future values from past value
    '''

    def __init__(self, N, K, sliding_window=True):
        '''
        Predict K future sample using N previous samples
        '''
        self.K = K
        self.N = N
        self.sliding_window = sliding_window

    def transform(self, A):
        M = self.N + self.K  # Number of samples per row (sample + target)
        # indexes
        if self.sliding_window:
            I = np.arange(M) + np.arange(A.shape[0] - M + 1).reshape(-1, 1)
        else:
            if A.shape[0] % M == 0:
                I = np.arange(M) + np.arange(0, A.shape[0], M).reshape(-1, 1)

            else:
                a = np.arange(M)
                Md = np.arange(0, A.shape[0] - M, M).reshape(-1, 1)
                I = np.arange(M) + np.arange(0, A.shape[0] - M, M).reshape(-1, 1)

        B = A[I].reshape(-1, M * A.shape[1], A.shape[2])
        ci = self.N * A.shape[1]  # Number of features per sample
        return B[:, :ci], B[:, ci:]  # Sample matrix, Target matrix


# data file path
dfp = 'data/bitcoin2015to2017.csv'

# Columns of price data to use
columns = ['Close']
df = pd.read_csv(dfp)
time_stamps = df['Timestamp']
df = df.loc[:, columns]

print(df.head())
original_df = pd.read_csv(dfp).loc[:, columns]
print(original_df.head())


file_name = 'bitcoin2015to2017_close_16_2.h5'


# 최소/최대값이 각각 1,0 이 되도록 스케일링
scaler = MinMaxScaler()
# feature scaling (normalization)
for c in columns:
    #
    df[c] = scaler.fit_transform(df[c].values.reshape(-1, 1))

print(df.head())

# Features are input sample dimensions(channels)
A = np.array(df)[:, None, :]
original_A = np.array(original_df)[:, None, :]
time_stamps = np.array(time_stamps)[:, None, None]

# Make samples of temporal sequences of pricing data (channel)
# 256 x 2 hours = 512 / 24 hours -> 약 21일
NPS, NFS = 16, 2  # Number of past and future samples
ps = PastSampler(NPS, NFS, sliding_window=False)
B, Y = ps.transform(A)
input_times, output_times = ps.transform(time_stamps)
original_B, original_Y = ps.transform(original_A)


with h5py.File(file_name, 'w') as f:
    f.create_dataset("inputs", data=B)
    f.create_dataset('outputs', data=Y)
    f.create_dataset("input_times", data=input_times)
    f.create_dataset('output_times', data=output_times)
    f.create_dataset("original_datas", data=np.array(original_df))
    f.create_dataset('original_inputs', data=original_B)
    f.create_dataset('original_outputs', data=original_Y)