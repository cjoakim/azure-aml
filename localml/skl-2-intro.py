
# Chris Joakim, 2021/01/11
# See https://scikit-learn.org/stable/tutorial/statistical_inference/settings.html

# https://scikit-learn.org/stable/modules/classes.html  (API)

import os
import pickle
import sys

from sklearn import datasets
from sklearn import svm

from joblib import dump, load

def digits():
    digits = datasets.load_digits()
    print(digits.data)    # The data is always a 2D array, shape (n_samples, n_features)
    print(digits.target)

    clf = svm.SVC(gamma=0.001, C=100.)  # clf = classifier
    print('clf: {} {}'.format(clf, str(type(clf))))

    # C is the Regularization parameter. The strength of the regularization is inversely proportional to C. 
    # Must be strictly positive. The penalty is a squared l2 penalty.

    result = clf.fit(digits.data[:-1], digits.target[:-1])  # fit == "train the model"
    print('fit result: {} {}'.format(str(type(result)), result))
    
    result = clf.predict(digits.data[-1:])
    print('clf predict result -1: {} {}'.format(str(type(result)), result))

    result = clf.predict(digits.data[-2:])
    print('clf predict result -2: {} {}'.format(str(type(result)), result))

    s = pickle.dumps(clf)
    print('s - type: {} len: {}'.format(str(type(s)), len(s)))

    clf2 = pickle.loads(s)
    print('clf2: {} {}'.format(clf2, str(type(clf2))))

    result = clf2.predict(digits.data[-1:])
    print('clf2 predict result -1: {} {}'.format(str(type(result)), result))

    result = clf2.predict(digits.data[-2:])
    print('clf2 predict result -2: {} {}'.format(str(type(result)), result))

    # use joblib’s replacement for pickle which is more efficient on big data but it can only
    # pickle to the disk and not to a string
    dump(clf, 'tmp/clf.joblib') 
    clf3 = load('tmp/clf.joblib')
    result = clf3.predict(digits.data[-1:])
    print('clf3 predict result -1: {} {}'.format(str(type(result)), result))

    result = clf3.predict(digits.data[-2:])
    print('clf3 predict result -2: {} {}'.format(str(type(result)), result))

def iris():
    iris = datasets.load_iris()
    print(iris.data)
    print(iris.target)

if __name__ == "__main__":

    if len(sys.argv) > 1:
        func = sys.argv[1].lower()
        if func == 'digits':
            digits()
        elif func == 'iris':
            iris()
