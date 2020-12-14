import random
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, data_path, train_size=0.7):
        self.data_path = data_path
        self.train_size = train_size
        self.test_size = 1 - train_size

    def load_data(self):
        data = pd.read_csv(self.data_path).values
        n_rows, n_cols = data.shape
        shuffled = np.random.permutation(n_rows)
        train_data = data[shuffled[:int(n_rows * self.train_size)], :]
        test_data = data[shuffled[int(n_rows * self.train_size):], :]

        # standardize
        scaler = MinMaxScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        
        return n_rows, n_cols, train_data, test_data

    # t is missingness threshold, set to 0.2 by the paper
    def get_missed_data(self, data, t=0.5, missing_mechanism="MCAR", randomness="uniform"):
        missed_data = data.copy()
        n_rows, n_cols = missed_data.shape
        mask = None
        v = np.random.uniform(size=(n_rows, n_cols))

        if missing_mechanism == "MCAR" and randomness == "uniform":
            mask = (v <= t)
        elif missing_mechanism == "MCAR" and randomness == "random":
            missed_cols = np.random.choice(n_cols, int(n_cols / 2), replace=False)
            c = np.zeros(n_cols, dtype=bool)
            c[missed_cols] = True
            mask = (v <= t) * c
        else:
            missed_cols = np.random.choice(n_cols, int(n_cols / 2), replace=False)
            c = np.zeros(n_cols, dtype=bool)
            c[missed_cols] = True
            # calculate the medians
            sample_cols = np.random.choice(n_cols, 2)
            m1, m2 = np.median(missed_data[:, sample_cols], axis=0)

            m1 = missed_data[:, sample_cols[0]] <= m1
            m2 = missed_data[:, sample_cols[1]] >= m2
            m = (m1 * m2)[:, np.newaxis]
            
            if missing_mechanism == "MNAR" and randomness == "uniform":
                mask = m * (v <= t)
            elif missing_mechanism == "MNAR" and randomness == "random":
                mask = m * (v <= t) * c
        missed_data[mask] = 0
        return missed_data, mask


        
