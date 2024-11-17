#!/usr/bin/env python3
import numpy as np
import logging
import matplotlib.pyplot as plt
import cv2

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

def visualize_misclassified_images(image_paths, predictions, true_labels, class_names=None):
    """
    Visualizes one misclassified image for each true class.

    Args:
        image_paths (list): List of paths to the original images.
        predictions (list or np.ndarray): Predicted labels for the images.
        true_labels (list or np.ndarray): True labels for the images.
        class_names (list, optional): List of class names corresponding to the labels. Defaults to None.
    """
    if len(image_paths) != len(predictions) or len(image_paths) != len(true_labels):
        raise ValueError("The lengths of image_paths, predictions, and true_labels must match.")

    # Dictionary to store the first misclassified image for each true class
    misclassified = {}

    for i, (img_path, pred, truth) in enumerate(zip(image_paths, predictions, true_labels)):
        if pred != truth and truth not in misclassified:
            misclassified[truth] = (img_path, pred)  # Store the path and predicted label

    # Visualize misclassified images
    for true_class, (img_path, pred_label) in misclassified.items():
        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"Could not read the image at {img_path}. Skipping.")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get class names if provided
        path_text = img_path
        pred_text = class_names[pred_label] if class_names else f"Pred: {pred_label}"
        true_text = class_names[true_class] if class_names else f"Truth: {true_class}"

        # Display the image with labels
        plt.figure(figsize=(5, 5))
        plt.imshow(img_rgb)
        plt.title(f"{path_text}\n{pred_text}\n{true_text}", color="red")
        plt.axis("off")
        plt.show()