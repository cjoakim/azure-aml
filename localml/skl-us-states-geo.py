import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import neighbors

# https://www.dataquest.io/blog/k-nearest-neighbors-in-python/


def print_info(obj, name):
    print('{} -> type: {}, shape: {}'.format(name, type(obj), obj.shape))

def print_df_info(df, msg=''):
    print('')
    print('print_df_info: {}'.format(msg))
    print(df.head(3))
    print(df.tail(3))
    print(df.dtypes)
    print(df.shape)
    print(df.columns)

def keyword_arg(keyword, default_value):
    for idx, arg in enumerate(sys.argv):
        if arg == keyword:
            next_idx = idx + 1
            if next_idx < len(sys.argv):
                return sys.argv[next_idx]
    return default_value

def boolean_flag_arg(flag):
    for arg in sys.argv:
        if arg == flag:
            return True
    return False

# python skl-us-states-geo.py datasets/postal_codes/postal_codes_us.csv
# python skl-us-states-geo.py datasets/postal_codes/postal_codes_us.csv --train --testfile


if __name__ == "__main__":
    start_time = time.time()
    infile = sys.argv[1].lower()
    print('infile is: {}'.format(infile))

    df = pd.read_csv(infile, delimiter=',')
    print('df type and shape: {} {}'.format(type(df), df.shape))

    print('df columns: {} '.format(df.columns.values))

    if boolean_flag_arg('--train'):
        data = df[['lat', 'lng']]  # select just these two features
        target = df['state']
        print_info(data, 'data')
        print_info(target, 'target')

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.15)  # random_state=0, test_size=0.33
        print_info(X_train, 'X_train')
        print_info(X_test, 'X_test')
        print_info(y_train, 'y_train')
        print_info(y_test, 'y_test')

        knn = neighbors.KNeighborsClassifier(n_neighbors=1)

        print('fitting...')
        knn.fit(X_train, y_train)

        if boolean_flag_arg('--testfile'):
            testfile = keyword_arg('--testfile', 'datasets/postal_codes/test_locations.csv')
            with open(testfile, 'rt', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    tokens = line.strip().split(',')
                    if idx > 0:
                        if len(tokens) > 3:
                            lat = float(tokens[0])
                            lng = float(tokens[1])
                            sample = [lat, lng]
                            reshaped = np.array(sample).reshape((1, -1))
                            print('sample: {}  reshaped: {}'.format(sample, reshaped))
                            p = knn.predict(reshaped)
                            print('sample: {} -> prediction: {}'.format(line.strip(), p))

        print('scoring...')
        score = knn.score(X_test, y_test)
        print('score: {} {}'.format(type(score), str(score)))  # <class 'numpy.float64'> 0.9143029080736452

        end_time = time.time()
        print('elapsed time: {}'.format(end_time - start_time))
