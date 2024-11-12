#!/usr/bin/env python3
import cv2
import csv
import random
import math
import numpy as np

def extract_contour(image, resize_dim=(25, 33)):
    """Extracts the largest contour in an image, crops to the bounding box, and returns the resized image."""
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
        cropped_image = image  # If no contour is found, use the original image

    # Resize the cropped image to the target size
    resized_image = cv2.resize(cropped_image, resize_dim)
    return resized_image

def load_data(image_directory='./2024F_imgs/', image_type='.png', image_size=(25, 33)):
    # Read labels file
    with open(image_directory + 'labels.txt', 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    # Shuffle and split lines
    random.shuffle(lines)
    split_idx = math.floor(len(lines) / 2)
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]

    # Load and preprocess training images
    train_data = []
    train_labels = []
    for line in train_lines:
        img = cv2.imread(image_directory + line[0] + image_type)
        img_contour = extract_contour(img, image_size)  # Apply contour extraction with cropping
        train_data.append(img_contour.flatten())
        train_labels.append(int(line[1]))

    # Load and preprocess test images
    test_data = []
    test_labels = []
    for line in test_lines:
        img = cv2.imread(image_directory + line[0] + image_type)
        img_contour = extract_contour(img, image_size)  # Apply contour extraction with cropping
        test_data.append(img_contour.flatten())
        test_labels.append(int(line[1]))

    # Convert to numpy arrays and return
    return (np.array(train_data, dtype=np.float32), np.array(train_labels, dtype=np.int32),
            np.array(test_data, dtype=np.float32), np.array(test_labels, dtype=np.int32))