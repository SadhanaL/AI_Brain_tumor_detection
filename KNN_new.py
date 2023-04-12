# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:26:41 2022

@author: sadha
"""

import numpy as np
import logging
import random
import matplotlib.pyplot as plt
from math import sqrt
from csv import reader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(asctime)s %(message)s',)

def read_data(_file_name):
    """
    Reads data from a CSV file and returns a list of data points.

    Parameters
    ----------
    _file_name : string
        Name of the CSV file to read.

    Returns
    -------
    _data : List
        DESCRIPTION.

    """
    assert('.csv' in _file_name)
    
    _data = list()
    with open(_file_name, 'r') as file:
        _read_file = reader(file)
        _header_flag = True
        for _data_pt in _read_file:
            if _header_flag == True:
                _header_flag = False
                continue
            if not _data_pt:
                continue
            _data.append(_data_pt)
    return _data


def fold_split(_data, _k_folds):
    """
    Split a dataset into k folds

    Parameters
    ----------
    _data : TYPE
        DESCRIPTION.
    _k_folds : TYPE
        DESCRIPTION.

    Returns
    -------
    split_data : TYPE
        DESCRIPTION.

    """
    split_data = list()
    data_copy = list(_data)
    fold_length = int(len(_data) / _k_folds)
    for k in range(_k_folds):
        fold = list()
        while len(fold) < fold_length:
            _index = random.randrange(len(data_copy))
            fold.append(data_copy.pop(_index))
        split_data.append(fold)
    return split_data

def evaluate_fold_accuracy(_dataset, _k_folds, k):
    """
    Calculates fold accuracies for all the K folds.

    Parameters
    ----------
    _dataset : TYPE
        DESCRIPTION.
    _k_folds : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.

    Returns
    -------
    _fold_acc : TYPE
        DESCRIPTION.

    """
    
    foldset = fold_split(_dataset, _k_folds)
    _fold_acc = list()
    _fold_num = 0
    for fold in foldset:
        train_set = list(foldset)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for _each_data_pt in fold:
            _data_pt_copy = list(_each_data_pt)
            test_set.append(_data_pt_copy)
            _data_pt_copy[-1] = None
        predicted_class = knn_model(train_set, test_set, k)
        target = [row[-1] for row in fold]
        accuracy = calc_accuracy(target, predicted_class)
        # for i in range(len(target)):
        #     print("Fold: {}, Target: {}, Prediction: {}".format(_fold_num,
                                                                       # target[i], 
                                                                       # predicted_class[i]))
        _fold_acc.append(accuracy)
        cm = confusion_matrix(np.array(target), np.array(predicted_class))
        print('Confusion Matrix: {}'.format(cm))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(['Negative','Positive']))
        disp.plot()
        plt. show()
        _fold_num += 1
    return _fold_acc

def calc_euclidean_distance(_data_pt_1, _data_pt_2):
    """
    Calculates Euclidean distance between two data points.

    Parameters
    ----------
    _data_pt_1 : list
        DESCRIPTION.
    _data_pt_2 : list
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    _dist = 0.0 
    for i in range(len(_data_pt_1)-1):
        _dist += (_data_pt_1[i] - _data_pt_2[i])**2
    return sqrt(_dist)

    
def find_K_nearest_neighbors(_train_set, _test_data_pt, _no_of_neighbors):
    """
    Find the K - nearest neighbors to a test data point 

    Parameters
    ----------
    _train_set : TYPE
        DESCRIPTION.
    _test_data_pt : TYPE
        DESCRIPTION.
    _no_of_neighbors : TYPE
        DESCRIPTION.

    Returns
    -------
    _neearest_neighbors : TYPE
        DESCRIPTION.

    """
    _distances = list()
    _nearest_neighbors = list()
    for _each_train_data_pt in _train_set:
        _dist = calc_euclidean_distance(_test_data_pt, _each_train_data_pt)
        _distances.append((_each_train_data_pt, _dist))
    _distances.sort(key=lambda tup: tup[1])
    for i in range(_no_of_neighbors):
        _nearest_neighbors.append(_distances[i][0])
    return _nearest_neighbors


def knn_model(_train_set, _test_set, _no_of_neighbors):
    """
    Runs KNN algorithm and generates a prediction.

    Parameters
    ----------
    _train_set : TYPE
        DESCRIPTION.
    _test_set : TYPE
        DESCRIPTION.
    _no_of_neighbors : TYPE
        DESCRIPTION.

    Returns
    -------
    _predicted_class_list : List
        DESCRIPTION.

    """
    _predicted_class_list = list()
    for _each_data_pt in _test_set:
        _neighbors = find_K_nearest_neighbors(_train_set, _each_data_pt, _no_of_neighbors)
        _neighbor_classes = [_data_pt[-1] for _data_pt in _neighbors]
        _predicted_class = max(set(_neighbor_classes), key=_neighbor_classes.count)
        _predicted_class_list.append(_predicted_class)
    return(_predicted_class_list)

def calc_accuracy(_target, _predicted_class):
    """
    Calculates accuracy based on target and predictions.

    Parameters
    ----------
    _target : TYPE
        DESCRIPTION.
    _predicted_class : TYPE
        DESCRIPTION.

    Returns
    -------
    _accuracy : TYPE
        DESCRIPTION.

    """
    _is_correct = 0
    for i in range(len(_target)):
        if _target[i] == _predicted_class[i]:
            _is_correct += 1
    _accuracy = _is_correct / float(len(_target)) * 100.0
    return _accuracy

if __name__ == '__main__':
    # evaluate algorithm
    no_of_folds = 5
    no_of_neighbors = 44
    accuracy = list()
    
    csv = 'new_feature.csv'
    data = read_data(csv)
    for i in range(len(data[0])-1):
        for _data_pt in data:
            _data_pt[i] = float(_data_pt[i].strip())
    # convert class column to integers
    _lookup_index = len(data[0])-1
    _labels = [data_pt[_lookup_index] for data_pt in data]
    _set_of_labels = set(_labels)
    _lookup = dict()
    for i, _label in enumerate(_set_of_labels):
        _lookup[_label] = i
    for data_pt in data:
        data_pt[_lookup_index] = _lookup[data_pt[_lookup_index]]
    
    
    # Uncomment for single no_of_neighbors
    _fold_acc = evaluate_fold_accuracy(data, no_of_folds, no_of_neighbors)
    accuracy.append(sum(_fold_acc)/float(len(_fold_acc)))
    print('Mean Accuracy: {}'.format(sum(_fold_acc)/float(len(_fold_acc))))
    
    ## Uncomment for graph
    #neighbors = list()
    # for no_of_neighbors in range (10,30):
    #     neighbors.append(no_of_neighbors)
    #     fold_acc = evaluate_fold_accuracy(data, folds ,no_of_neighbors)
    #     accuracy.append(sum(fold_acc)/float(len(fold_acc)))
    #     # print('Scores: %s' % scores)
    #     print('Neighbors: {} Mean of fold accuracy: {}'.format(no_of_neighbors,(sum(fold_acc)/float(len(fold_acc)))))
    # plt.Figure()
    # plt.plot(neighbors,accuracy)


   