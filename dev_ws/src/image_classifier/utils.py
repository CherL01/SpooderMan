#!/usr/bin/env python3
import numpy as np

def calculate_accuracy(correct_predictions, total_samples):
    """Calculates accuracy as a percentage."""
    return correct_predictions / total_samples

def print_confusion_matrix(confusion_matrix):
    """Nicely prints the confusion matrix."""
    print("Confusion Matrix:")
    print(confusion_matrix)

def update_confusion_matrix(confusion_matrix, true_label, predicted_label):
    """Updates confusion matrix given a true and predicted label."""
    confusion_matrix[true_label, predicted_label] += 1
    return confusion_matrix