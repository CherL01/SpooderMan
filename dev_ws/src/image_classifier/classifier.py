#!/usr/bin/env python3
import cv2
import numpy as np
import logging
import utils

def train_classifier(train_data, train_labels):
    """
    Initializes and trains the K-Nearest Neighbors (KNN) classifier.

    Args:
        train_data (np.ndarray): Training data, where each row represents a feature vector.
        train_labels (np.ndarray): Corresponding labels for the training data.

    Returns:
        knn: Trained KNN classifier.
    """
    logging.info("Initializing and training the K-Nearest Neighbors classifier.")
    try:
        knn = cv2.ml.KNearest_create()
        knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
        logging.info(f"Successfully trained KNN classifier on {len(train_data)} samples.")
    except Exception as e:
        logging.error(f"Error during KNN training: {e}")
        raise
    return knn

def evaluate_classifier(knn, test_data, test_labels, k=7):
    """
    Evaluates the KNN classifier on the test dataset and generates a confusion matrix.

    Args:
        knn: Trained KNN classifier.
        test_data (np.ndarray): Test data, where each row represents a feature vector.
        test_labels (np.ndarray): Corresponding labels for the test data.
        k (int): Number of neighbors to consider in KNN (default is 7).

    Returns:
        tuple: Accuracy as a float, and the confusion matrix as a 2D numpy array.
    """
    logging.info("Evaluating the KNN classifier.")
    correct = 0
    num_classes = len(np.unique(test_labels))  # Dynamically determine the number of classes
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    for i in range(len(test_data)):
        # Reshape test sample and predict using KNN
        sample = test_data[i].reshape(1, -1)
        try:
            _, result, _, _ = knn.findNearest(sample, k)
            predicted_label = int(result[0][0])
            actual_label = test_labels[i]
        except Exception as e:
            logging.error(f"Error during KNN prediction for sample {i}: {e}")
            continue

        # Update accuracy count
        if predicted_label == actual_label:
            correct += 1

        # Update confusion matrix
        confusion_matrix = utils.update_confusion_matrix(confusion_matrix, actual_label, predicted_label)

        # Optional debug logging
        logging.debug(f"Sample {i}: Actual = {actual_label}, Predicted = {predicted_label}")

    # Calculate accuracy
    accuracy = utils.calculate_accuracy(correct, len(test_data))
    logging.info(f"Evaluation completed. Accuracy: {accuracy:.2%}")

    return accuracy, confusion_matrix