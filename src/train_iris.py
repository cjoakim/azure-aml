# Modified from https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/

import argparse
import joblib
import os

import numpy as np

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Import the AML SDK library
import azureml.core
from azureml.core import Dataset
from azureml.core import Datastore
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import Workspace
from azureml.core.run import Run

def main():
    print('AML SDK library loaded; version:', azureml.core.VERSION)

    # parse the command-line args:
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=str, default='linear',
                        help='Kernel type to be used in the algorithm')
    parser.add_argument('--penalty', type=float, default=1.0,
                        help='Penalty parameter of the error term')
    args = parser.parse_args()

    run = Run.get_context()
    print('run: {} {}'.format(run, str(type(run))))

    ws = run.experiment.workspace
    print('ws: {} {}'.format(ws, str(type(ws))))

    run.log('Kernel type', np.str(args.kernel))
    run.log('Penalty', np.float(args.penalty))

    # load my iris dataset (not the sklearn one)
    # https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.dataset(class)?view=azure-ml-py#get-by-name-workspace--name--version--latest--
    # dataset = Dataset.get_by_id(ws, id='61efe025-64ff-403d-9923-4f313b3ce048')
    # dataset = Dataset.get_by_name(ws, name='iris.csv')
    # print('dataset: {} {}'.format(dataset, str(type(dataset))))

    iris = Dataset.get_by_name(ws, name='iris.csv').to_pandas_dataframe()
    iris.head()
    iris.describe()

    # INTO THE REALM OF MACHINE LEARNING ...

    # Seperating the data into dependent and independent variables
    X = iris.iloc[:, :-1].values
    y = iris.iloc[:, -1].values
    print('X:')
    print(X)
    print('y:')
    print(y)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    print('split')
    print('X_train:')
    print(X_train)
    print('X_test:')
    print(X_test)
    print('y_train:')
    print(y_train)
    print('y_test:')
    print(y_test)

    # LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Summary of the predictions made by the classifier
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Accuracy score
    print('accuracy is ',accuracy_score(y_pred,y_test))

    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(classifier, 'outputs/iris_logistic_regression_model.joblib')


    # --- original below 
    
    # X -> features, y -> label
    # X = iris.data
    # y = iris.target

    # dividing X, y into train and test data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # training a linear SVM classifier
    # svm_model_linear = SVC(kernel=args.kernel, C=args.penalty).fit(X_train, y_train)
    # svm_predictions = svm_model_linear.predict(X_test)

    # model accuracy for X_test
    # accuracy = svm_model_linear.score(X_test, y_test)
    # print('Accuracy of SVM classifier on test set: {:.2f}'.format(accuracy))
    # run.log('Accuracy', np.float(accuracy))

    # creating a confusion matrix
    # cm = confusion_matrix(y_test, svm_predictions)
    # print(cm)

    # os.makedirs('outputs', exist_ok=True)
    # # files saved in the "outputs" folder are automatically uploaded into run history
    # joblib.dump(svm_model_linear, 'outputs/iris_model.joblib')


if __name__ == '__main__':
    main()
