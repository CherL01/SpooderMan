#!/usr/bin/env python3
import numpy as np
import logging
import matplotlib.pyplot as plt

def calculate_accuracy(correct_predictions, total_samples):
    """
    Calculates accuracy as a percentage.

    Args:
        correct_predictions (int): Number of correctly predicted samples.
        total_samples (int): Total number of samples.

    Returns:
        float: Accuracy as a decimal (e.g., 0.92 for 92%).
    """
    if total_samples == 0:
        logging.warning("Total samples is 0. Returning accuracy as 0.")
        return 0.0
    accuracy = correct_predictions / total_samples
    logging.debug(f"Calculated accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    return accuracy

def print_confusion_matrix(confusion_matrix, class_labels=None, title="Confusion Matrix"):
    """
    Prints the confusion matrix and plots it as a heatmap.

    Args:
        confusion_matrix (np.ndarray): A 2D numpy array representing the confusion matrix.
        class_labels (list, optional): List of class labels for the axes. Defaults to numeric labels if None.
        title (str): Title for the plot.
    """
    logging.info("Printing confusion matrix...")
    if not isinstance(confusion_matrix, np.ndarray):
        logging.error("Confusion matrix is not a numpy array.")
        raise TypeError("Confusion matrix must be a numpy array.")
    
    # Print the confusion matrix as text
    print("Confusion Matrix:")
    print(confusion_matrix)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    # Handle class labels
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(confusion_matrix.shape[0])]

    # Set axis labels and ticks
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Add text annotations for each cell
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(
                j, i, f"{confusion_matrix[i, j]}",
                horizontalalignment="center",
                color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black"
            )

    plt.tight_layout()
    plt.show()

def update_confusion_matrix(confusion_matrix, true_label, predicted_label):
    """
    Updates the confusion matrix given a true label and a predicted label.

    Args:
        confusion_matrix (np.ndarray): A 2D numpy array representing the confusion matrix.
        true_label (int): The actual class label.
        predicted_label (int): The predicted class label.

    Returns:
        np.ndarray: Updated confusion matrix.
    """
    try:
        confusion_matrix[true_label, predicted_label] += 1
        logging.debug(f"Updated confusion matrix at [{true_label}, {predicted_label}].")
    except IndexError as e:
        logging.error(f"IndexError updating confusion matrix: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error updating confusion matrix: {e}")
        raise
    return confusion_matrix