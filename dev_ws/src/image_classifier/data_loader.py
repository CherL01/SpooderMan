#!/usr/bin/env python3
import cv2
import csv
import random
import math
import numpy as np
import logging
import os
from features import compute_shape_features, compute_edge_features, compute_spatial_gradient_features

### data labels ###
# 0: other?
# 1: left turn
# 2: right turn
# 3: u-turn
# 4: stop sign --> u-turn
# 5: goal

def extract_contour(image, resize_dim=(25, 33)):
    """
    Extracts the largest contour in an image, crops to the bounding box, and resizes the cropped image.

    Args:
        image (np.ndarray): Input image in BGR format.
        resize_dim (tuple): Target size for the resized image (width, height).

    Returns:
        np.ndarray: Resized image containing the largest contour or the original resized image if no contour is found.
    """
    logging.debug("Extracting contour from image...")
    try:
        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If contours are found, take the largest contour
        if contours:
            contour = max(contours, key=cv2.contourArea)

            # Get the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(contour)

            # Crop the image to the bounding box
            cropped_image = image[y:y+h, x:x+w]
        else:
            logging.warning("No contours found in the image. Using the original image.")
            # Show the image for debugging
            cv2.imshow("No Contours Found - Original Image", image)
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()  # Close the OpenCV window
            cropped_image = image  # Use the original image if no contour is found

        # Resize the cropped image to the target size
        resized_image = cv2.resize(cropped_image, resize_dim)
        return resized_image

    except Exception as e:
        logging.error(f"Error while extracting contour: {e}")
        raise

def load_data(image_directory='./2024F_imgs', image_type='.png', image_size=(25, 33), grid_size=(8, 8), random_seed=42, split_ratio=0.8):
    """
    Loads and preprocesses image data, including spatial gradient, edge, and contour features.

    Args:
        image_directory (str): Directory containing the images and the labels file.
        image_type (str): File extension of the images (e.g., '.png', '.jpg').
        image_size (tuple): Target size for resized images (width, height).
        grid_size (tuple): Number of cells for spatial gradient features.

    Returns:
        tuple: Four numpy arrays (train_data, train_labels, test_data, test_labels).
    """
    logging.info(f"Loading data from directory: {image_directory}")
    labels_file = os.path.join(image_directory, 'labels.txt')

    try:
        with open(labels_file, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
        logging.info(f"Loaded {len(lines)} labeled entries from {labels_file}.")
    except FileNotFoundError:
        logging.error(f"Labels file not found at {labels_file}.")
        raise
    except Exception as e:
        logging.error(f"Error reading labels file: {e}")
        raise

    # Shuffle and split lines into training and testing datasets
    random.seed(random_seed)
    random.shuffle(lines)
    split_idx = math.floor(len(lines) * split_ratio)
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]
    logging.info(f"Split data into {len(train_lines)} training samples and {len(test_lines)} testing samples.")

    train_data, train_labels = [], []
    for line in train_lines:
        img_path = os.path.join(image_directory, line[0] + image_type)
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"Failed to load image: {img_path}. Skipping.")
            continue

        # Extract contour features
        img_contour = extract_contour(img, image_size).flatten()

        # Extract spatial gradient features
        spatial_features = compute_spatial_gradient_features(img, grid_size)

        # Extract edge features
        edge_features = compute_edge_features(img)

        # Combine all features
        combined_features = np.hstack([img_contour, spatial_features, edge_features])

        train_data.append(combined_features)
        train_labels.append(int(line[1]))

    test_data, test_labels = [], []
    for line in test_lines:
        img_path = os.path.join(image_directory, line[0] + image_type)
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"Failed to load image: {img_path}. Skipping.")
            continue

        # Extract contour features
        img_contour = extract_contour(img, image_size).flatten()

        # Extract spatial gradient features
        spatial_features = compute_spatial_gradient_features(img, grid_size)

        # Extract edge features
        edge_features = compute_edge_features(img)

        # Combine all features
        combined_features = np.hstack([img_contour, spatial_features, edge_features])

        test_data.append(combined_features)
        test_labels.append(int(line[1]))

    # Convert lists to numpy arrays
    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)
    test_data = np.array(test_data, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.int32)

    logging.info(f"Data loading completed: {len(train_data)} training samples, {len(test_data)} testing samples.")
    return train_data, train_labels, test_data, test_labels