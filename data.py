import math

import numpy as np
import pandas as pd

import time

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import models
import graphing
import preprocessing


def init_data():
    data_filename = 'raw data/vari_summary.csv'
    classes_filename = 'raw data/vari_classifier_result.csv'

    data = pd.read_csv(filepath_or_buffer=data_filename, delimiter=',', comment="#", skipinitialspace=True,
                       low_memory=True)
    data = data.drop('classifier_name', axis=1)
    data = data.drop('best_class_name', axis=1)
    data = data.drop('best_class_score', axis=1)

    print('data loaded')
    class_data = pd.read_csv(filepath_or_buffer=classes_filename, delimiter=',', comment='#', skipinitialspace=True,
                             low_memory=True)
    print('class data loaded')

    return data, class_data


def prepare_data(data):
    print('preparing data')
    X, y = get_features(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1 / 9, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1 / 9, random_state=42)
    print('data successfully prepared')
    return X_train, y_train, X_test, y_test


def get_features(data):
    X = np.vstack([data['num_selected_g_fov'], data['mean_obs_time_g_fov'], data['time_duration_g_fov'],
                   data['min_mag_g_fov'], data['max_mag_g_fov'], data['mean_mag_g_fov'], data['median_mag_g_fov'],
                   data['range_mag_g_fov'], data['trimmed_range_mag_g_fov'], data['std_dev_mag_g_fov'],
                   data['skewness_mag_g_fov'], data['kurtosis_mag_g_fov'], data['mad_mag_g_fov'],
                   data['abbe_mag_g_fov'], data['iqr_mag_g_fov'], data['stetson_mag_g_fov'],
                   data['std_dev_over_rms_err_mag_g_fov'], data['outlier_median_g_fov'], data['num_selected_bp'],
                   data['mean_obs_time_bp'], data['time_duration_bp'], data['min_mag_bp'], data['max_mag_bp'],
                   data['mean_mag_bp'], data['median_mag_bp'], data['range_mag_bp'], data['trimmed_range_mag_bp'],
                   data['std_dev_mag_bp'], data['skewness_mag_bp'], data['kurtosis_mag_bp'], data['mad_mag_bp'],
                   data['abbe_mag_bp'], data['iqr_mag_bp'], data['stetson_mag_bp'], data['std_dev_over_rms_err_mag_bp'],
                   data['outlier_median_bp'], data['num_selected_rp'], data['mean_obs_time_rp'],
                   data['time_duration_rp'], data['min_mag_rp'], data['max_mag_rp'], data['mean_mag_rp'],
                   data['median_mag_rp'], data['range_mag_rp'], data['trimmed_range_mag_rp'], data['std_dev_mag_rp'],
                   data['skewness_mag_rp'], data['kurtosis_mag_rp'], data['mad_mag_rp'], data['abbe_mag_rp'],
                   data['iqr_mag_rp'], data['stetson_mag_rp'], data['std_dev_over_rms_err_mag_rp'],
                   data['outlier_median_rp']], dtype=np.float32).T
    y = (data['best_class_name'])
    return X, y


def random_forest(data, classes):
    # Random Forest
    X_train, y_train, X_test, y_test = prepare_data(data)
    # X_train, y_train = get_features(filtered_data)
    # X_test, y_test = get_features(full_data)
    rfm, rfm_y_pred, rfm_y_score, rfm_score_classes = models.run_random_forest(X_train, y_train, X_test, y_test, quick_train=True)

    graphing.plot_roc_curve(rfm_y_score, y_train, y_test, class_names=classes)
    graphing.plot_precision_recall_curve(rfm_y_score, y_test, classes)
    graphing.graph_features(rfm.feature_importances_)
    graphing.graph_confusion_matrix(classes, rfm_y_pred, y_test)

    graphing.roc_scoring(rfm_score_classes, rfm_y_score, y_test)
    graphing.precision_recall_scoring(rfm_score_classes, rfm_y_pred, y_test)
    print(classification_report(y_test, rfm_y_pred, labels=classes))


def neural_network(data, classes):
    X = data.iloc[:, 5:59]
    X_features = pd.DataFrame(X, dtype=np.float32)
    y = data.iloc[:, 73:74]
    model, y_pred, y_test, y_train, y_score = models.run_neural_network(X_features, y)
    graphing.plot_roc_curve(y_score, y_train, y_test, classes)
    graphing.plot_precision_recall_curve(y_score, y_test, classes)
    graphing.graph_confusion_matrix(classes, y_pred, y_test)


def naive_bayes(data, classes):
    X, y = get_features(data)

    from sklearn import naive_bayes
    # X_train, y_train, X_test, y_test = prepare_data(data)

    Y = label_binarize(y, classes=classes)
    random_state = np.random.RandomState(0)
    # Split into training and test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.5, random_state=random_state
    )
    print(Y_train)

    classifier = OneVsRestClassifier(
        make_pipeline(StandardScaler(), naive_bayes.GaussianNB())
    )
    classifier.fit(X_train, Y_train)
    y_score = classifier.predict_proba(X_test)
    # print(y_test)
    # y_pred, _ = models.run_logistic_regression(X_train, y_train, X_test, y_test)
    graphing.plot_precision_recall_curve(y_score, Y_test, classes)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
    classifier = (naive_bayes.GaussianNB())
    classifier.fit(X_train, y_train)
    y_score = classifier.predict_proba(X_test)
    graphing.plot_roc_curve(y_score, y_train, y_test, class_names=classes)

    X_train, y_train, X_test, y_test = prepare_data(data)
    y_pred = models.run_naive_bayes(X_train, y_train, X_test, y_test)
    print(classification_report(y_test, y_pred, labels=classes))


def logistic_regression(data, classes):
    X, y = get_features(data)

    from sklearn.linear_model import LogisticRegression
    # X_train, y_train, X_test, y_test = prepare_data(data)

    Y = label_binarize(y, classes=classes)
    random_state = np.random.RandomState(0)
    # Split into training and test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.5, random_state=random_state
    )
    print(Y_train)

    classifier = OneVsRestClassifier(
        make_pipeline(StandardScaler(), LogisticRegression())
    )
    classifier.fit(X_train, Y_train)
    y_score = classifier.predict_proba(X_test)
    # print(y_test)
    # y_pred, _ = models.run_logistic_regression(X_train, y_train, X_test, y_test)
    graphing.plot_precision_recall_curve(y_score, Y_test, classes)


    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_score = classifier.predict_proba(X_test)
    graphing.plot_roc_curve(y_score, y_train, y_test, class_names=classes)

    # print(classification_report(y_test, y_pred, labels=classes))


def main():
    classes = ['ECL', 'LPV', 'SOLAR_LIKE', 'DSCT|GDOR|SXPHE', 'AGN', 'RS', 'S', 'RR', 'OTHER']

    try:
        # data = pd.read_csv('combined.csv', low_memory=True)
        raw_summary = pd.read_csv('clean full data/combined_clean.csv', low_memory=True)
        # combined = pd.read_csv('raw data/combined.csv', memory_map=True)
        # memory_mapping in numpy
        # total_data = pd.read_csv('preprocessed data/total.csv', low_memory=True)
        # full_data = pd.read_csv('clean full data/0.2_combined_clean.csv', low_memory=True)
        # data_50 = pd.read_csv('preprocessed data/0.5.csv', low_memory=True)
        # data_20 = pd.read_csv('preprocessed data/0.2.csv', low_memory=True)
        # data_30 = pd.read_csv('clean full data/0.3_combined_clean.csv', low_memory=True)
        # data_75 = pd.read_csv('preprocessed data/0.75.csv', low_memory=True)
        print('csv read in successfully')
        data = raw_summary
    except FileNotFoundError:
        print("making new read")
        data_init, class_data = init_data()
        total_data = preprocessing.combine_dataframes(data_init, class_data)
        data = total_data

    print(data.shape)
    graphing.graph_data_types(data)

    # X, y = get_features(data)
    # models.randomize_random_forest(X, y)
    # print(combined.shape)

    # logistic_regression(data, classes)
    # naive_bayes(data, classes)
    # random_forest(data, classes)
    # neural_network(data, classes)

    # preprocessing.filter_classes(total_data, "total")
    # preprocessing.filter_confidence('raw data/combined.csv', 0.9, True)



"""
Notes:
Use the sklearn randomizer to find the optimal hyperparameters, use validation set to do so, then test on test set.
Use area under the precision recall curve to determine how good a model is

TODOS:
1. Compare the classifications of my models to Gaia by id number

Neural networks:
Use softmax, not sigmoid for the activation func because I have more than two classifications.
"""

if __name__ == '__main__':
    main()
