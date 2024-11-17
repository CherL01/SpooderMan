#!/usr/bin/env python3
import cv2
import numpy as np
import logging
import utils
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

def train_classifier(train_data, train_labels, model="KNN"):
    """
    Initializes and trains a classifier based on the specified model type.

    Args:
        train_data (np.ndarray): Training data, where each row represents a feature vector.
        train_labels (np.ndarray): Corresponding labels for the training data.
        model (str): Type of model to train ("KNN" or "CNN").

    Returns:
        model: Trained classifier model.
    """
    if model == "KNN":
        logging.info("Initializing and training the K-Nearest Neighbors classifier.")
        try:
            knn = cv2.ml.KNearest_create()
            knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
            logging.info(f"Successfully trained KNN classifier on {len(train_data)} samples.")
            return knn
        except Exception as e:
            logging.error(f"Error during KNN training: {e}")
            raise

    elif model == "CNN":
        logging.info("Initializing and training the Convolutional Neural Network (CNN).")
        try:
            # Define sizes
            image_flattened_size = 33 * 33 * 3  # Flattened size of the image
            total_feature_size = train_data.shape[1]

            # Ensure the combined_features include img_contour
            if total_feature_size < image_flattened_size:
                raise ValueError(f"train_data does not contain enough dimensions for the image contour. "
                                f"Expected at least {image_flattened_size}, but got {total_feature_size}.")

            # Extract img_contour for CNN input
            image_data = train_data[:, :image_flattened_size].reshape(-1, 33, 33, 3)

            # Extract spatial and edge features if needed
            additional_features = train_data[:, image_flattened_size:]

            # Convert labels to one-hot encoding
            num_classes = len(np.unique(train_labels))
            train_labels = to_categorical(train_labels, num_classes)

            # Define CNN architecture
            image_input = tf.keras.Input(shape=(33, 33, 3), name="image_input")
            x = Conv2D(32, (3, 3), activation="relu")(image_input)
            x = MaxPooling2D((2, 2))(x)
            x = Conv2D(64, (3, 3), activation="relu")(x)
            x = MaxPooling2D((2, 2))(x)
            x = Flatten()(x)

            # Add fully connected layers
            x = Dense(128, activation="relu")(x)
            output = Dense(num_classes, activation="softmax")(x)

            # Compile the model
            cnn = tf.keras.Model(inputs=image_input, outputs=output)
            cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

            # Train the CNN
            cnn.fit(image_data, train_labels, epochs=20, batch_size=32, verbose=1)
            logging.info("Successfully trained CNN.")
            return cnn
        except Exception as e:
            logging.error(f"Error during CNN training: {e}")
            raise
    else:
        raise ValueError(f"Unsupported model type: {model}")

def evaluate_classifier(model, test_data, test_labels, k=7, model_type="KNN"):
    """
    Evaluates the classifier on the test dataset and generates a confusion matrix.

    Args:
        model: Trained classifier model.
        test_data (np.ndarray): Test data, where each row represents a feature vector.
        test_labels (np.ndarray): Corresponding labels for the test data.
        k (int): Number of neighbors to consider in KNN (default is 7, used for KNN only).
        model_type (str): Type of model ("KNN" or "CNN").

    Returns:
        tuple: Accuracy as a float, the confusion matrix as a 2D numpy array, and predictions.
    """
    logging.info(f"Evaluating the {model_type} classifier.")
    correct = 0
    num_classes = len(np.unique(test_labels))  # Dynamically determine the number of classes
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    predictions = []

    if model_type == "KNN":
        for i in range(len(test_data)):
            # Reshape test sample and predict using KNN
            sample = test_data[i].reshape(1, -1)
            try:
                _, result, _, _ = model.findNearest(sample, k)
                predicted_label = int(result[0][0])
                actual_label = test_labels[i]
            except Exception as e:
                logging.error(f"Error during KNN prediction for sample {i}: {e}")
                continue

            predictions.append(predicted_label)

            # Update accuracy count
            if predicted_label == actual_label:
                correct += 1

            # Update confusion matrix
            confusion_matrix = utils.update_confusion_matrix(confusion_matrix, actual_label, predicted_label)

            # Optional debug logging
            logging.debug(f"Sample {i}: Actual = {actual_label}, Predicted = {predicted_label}")

    elif model_type == "CNN":
        try:
            image_flattened_size = 33 * 33 * 3  # Flattened size of the image
            image_data = test_data[:, :image_flattened_size].reshape(-1, 33, 33, 3)
            predictions = np.argmax(model.predict(image_data), axis=1)

            correct = np.sum(predictions == test_labels)
            for actual, predicted in zip(test_labels, predictions):
                confusion_matrix = utils.update_confusion_matrix(confusion_matrix, actual, predicted)
        except Exception as e:
            logging.error(f"Error during CNN evaluation: {e}")
            raise

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Calculate accuracy
    accuracy = utils.calculate_accuracy(correct, len(test_data))
    logging.info(f"Evaluation completed. Accuracy: {accuracy:.2%}")

    return accuracy, confusion_matrix, predictions