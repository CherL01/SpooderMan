#!/usr/bin/env python3
import cv2
import numpy as np
from data_loader import load_data
import utils

def train_classifier(train_data, train_labels, k=7):
    """Initializes and trains the K-Nearest Neighbors classifier."""
    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    return knn

def evaluate_classifier(knn, test_data, test_labels, k=7):
    """Evaluates classifier accuracy and generates a confusion matrix."""
    correct = 0
    confusion_matrix = np.zeros((6, 6), dtype=np.int32)

    for i in range(len(test_data)):
        _, result, _, _ = knn.findNearest(test_data[i].reshape(1, -1), k)
        predicted_label = int(result[0][0])
        actual_label = test_labels[i]

        if predicted_label == actual_label:
            correct += 1
        confusion_matrix = utils.update_confusion_matrix(confusion_matrix, actual_label, predicted_label)

    accuracy = utils.calculate_accuracy(correct, len(test_data))
    return accuracy, confusion_matrix