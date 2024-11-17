#!/usr/bin/env python3
import cv2
import csv
import random
import math
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from features import compute_shape_features, compute_edge_features, compute_spatial_gradient_features

### data labels ###
# 0: other?
# 1: left turn
# 2: right turn
# 3: u-turn
# 4: stop sign --> u-turn
# 5: goal

def extract_sign_from_cardboard(image, resize_dim=(25, 33), delta_h=10, delta_s=10, delta_v=10, dynamic_color_detection=False, plotting=False):
    """
    Detects the brown cardboard area or falls back to detecting red, green, and blue regions (sign colors),
    then detects the directional sign and crops the image.

    Args:
        image (np.ndarray): Input image in BGR format.
        resize_dim (tuple): Target size for the resized image (width, height).
        delta_h (int): Range for hue adjustment around the dominant color.
        delta_s (int): Range for saturation adjustment around the dominant color.
        delta_v (int): Range for value adjustment around the dominant color.

    Returns:
        np.ndarray: Resized grayscale image of the directional sign or the original resized image if no sign is found.
    """
    logging.debug("Detecting brown cardboard and directional sign...")
    try:
        # Convert image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if dynamic_color_detection:
            # Sample a central region to compute the dominant color (focus on the cardboard)
            h, w, _ = image.shape
            sample_region = hsv[h // 4:h - h // 2, w // 4:w - w // 2]  # Use the center of the image for sampling
            dominant_hue = int(np.mean(sample_region[:, :, 0]))  # Dominant Hue
            dominant_saturation = int(np.mean(sample_region[:, :, 1]))  # Dominant Saturation
            dominant_value = int(np.mean(sample_region[:, :, 2]))  # Dominant Value

            # Dynamically compute the HSV bounds for brown
            lower_hsv = np.array([max(0, dominant_hue - delta_h),
                                max(0, dominant_saturation - delta_s),
                                max(0, dominant_value - delta_v)])
            upper_hsv = np.array([min(180, dominant_hue + delta_h),
                                min(255, dominant_saturation + delta_s),
                                min(255, dominant_value + delta_v)])
            
        else:
            lower_hsv = np.array([10, 100, 20])  # Adjust values as needed for brown
            upper_hsv = np.array([20, 255, 200])

        # Create a mask for detecting the brown cardboard
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

        # Visualize the mask overlaid on the original image
        mask_overlay = cv2.bitwise_and(image, image, mask=cleaned_mask)

        # Find contours in the mask
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect the largest cardboard contour
        cardboard_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            # aspect_ratio = w / h

            if area > 7000: # and 0.8 < aspect_ratio < 1.0:  # Keep large, rectangular contours
                cardboard_contour = contour
                break

        # Fallback to red, green, or blue detection if no brown cardboard is found
        if cardboard_contour is None:
            # logging.warning("No suitable brown cardboard detected. Falling back to red, green, and blue contours.")
            
            # Define HSV ranges for red, green, and blue
            color_ranges = {
                "red": [
                    # First red range (0-10)
                    [(0, 100, 100), (10, 255, 255)],
                    # Second red range (160-180)
                    [(160, 100, 100), (180, 255, 255)]
                ],
                "green": [[(35, 40, 40), (85, 255, 255)]],
                "blue": [[(90, 50, 70), (128, 255, 255)]]
            }

            # Create a combined mask for all colors
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            # Process each color
            for color, ranges_list in color_ranges.items():
                color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                
                # Apply each range for the current color
                for ranges in ranges_list:
                    lower, upper = ranges
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    color_mask = cv2.bitwise_or(color_mask, mask)
                
                # Apply morphological operations to clean up the mask
                kernel = np.ones((5,5), np.uint8)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
                
                # Combine with the main mask
                combined_mask = cv2.bitwise_or(combined_mask, color_mask)

            # Find contours in the combined mask
            color_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # # Detect the largest color contour
            # cardboard_contour = max(color_contours, key=cv2.contourArea) if color_contours else None
            # if cardboard_contour is not None:
            #     # logging.info("Color contour detected.")
            #     x, y, w, h = cv2.boundingRect(cardboard_contour)
            #     cardboard_area = image[y:y+h, x:x+w]
            #     cardboard_offset = (x, y)
            # else:
            #     # logging.info("No color contour detected.")
            #     cardboard_area = image
            #     cardboard_offset = (0, 0)
            #     x, y, w, h = 0, 0, image.shape[1], image.shape[0]
            
            # Define a minimum area threshold for a valid directional sign contour
            min_area_threshold = 100  # Adjust this value based on your use case

            # Detect the largest color contour that exceeds the area threshold
            valid_contours = [contour for contour in color_contours if cv2.contourArea(contour) > min_area_threshold]

            if valid_contours:
                # Find the largest valid contour
                cardboard_contour = max(valid_contours, key=cv2.contourArea)
                # logging.info(f"Valid color contour detected with area: {cv2.contourArea(cardboard_contour)}")

                # Get the bounding box for the largest contour
                x, y, w, h = cv2.boundingRect(cardboard_contour)
                cardboard_area = image[y:y+h, x:x+w]
                cardboard_offset = (x, y)
            else:
                # No valid contour detected
                # logging.warning("No valid color contour detected.")
                cardboard_area = image
                cardboard_offset = (0, 0)
                x, y, w, h = 0, 0, image.shape[1], image.shape[0]
        
        else:
            # logging.info("Brown cardboard detected.")
            x, y, w, h = cv2.boundingRect(cardboard_contour)
            cardboard_area = image[y:y+h, x:x+w]
            cardboard_offset = (x, y)

        # Step 2: Detect the directional sign
        gray_cardboard = cv2.cvtColor(cardboard_area, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_cardboard, (1, 1), 0)

        # Use adaptive thresholding for better contrast-based sign detection
        sign_mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours in the sign mask
        sign_contours, _ = cv2.findContours(sign_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect the largest sign contour
        sign_area = cardboard_area
        sign_bbox = None
        for contour in sign_contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Filter out very small contours
                sx, sy, sw, sh = cv2.boundingRect(contour)
                sign_area = cardboard_area[sy:sy+sh, sx:sx+sw]
                sign_bbox = (sx + cardboard_offset[0], sy + cardboard_offset[1],
                             sw, sh)  # Adjust bounding box to the original image
                cropped_original = image[y+sy:y+sy+sh, x+sx:x+sx+sw]
                break
            
        else:
            cropped_original = image  # Fallback to the entire original image

        # Resize the detected sign area
        resized_image = cv2.resize(sign_area, resize_dim)

        if plotting:
            # Plot all debug images in a single matplotlib figure (2 rows)
            fig, ax = plt.subplots(2, 3, figsize=(12, 8))  # Reduced figure size (smaller window)
            debug_image = image.copy()

            # First row
            if cardboard_contour is not None:
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue bounding box for cardboard
            if sign_bbox is not None:
                sx, sy, sw, sh = sign_bbox
                cv2.rectangle(debug_image, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)  # Green bounding box for the sign

            ax[0, 0].imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
            ax[0, 0].set_title("Detected Cardboard or Color Contour")
            ax[0, 0].axis("off")

            ax[0, 1].imshow(cleaned_mask, cmap="gray")
            ax[0, 1].set_title("Refined HSV Mask for Brown")
            ax[0, 1].axis("off")

            ax[0, 2].imshow(cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB))
            ax[0, 2].set_title("Mask Overlay on Original Image")
            ax[0, 2].axis("off")

            # Second row
            if cardboard_contour is not None:
                ax[1, 0].imshow(cv2.cvtColor(cardboard_area, cv2.COLOR_BGR2RGB))
                ax[1, 0].set_title("Cardboard or Color Contour Area")
                ax[1, 0].axis("off")
            else:
                ax[1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                ax[1, 0].set_title("Fallback Entire Image")
                ax[1, 0].axis("off")

            ax[1, 1].imshow(sign_mask, cmap="gray")
            ax[1, 1].set_title("Sign Mask")
            ax[1, 1].axis("off")

            ax[1, 2].imshow(resized_image, cmap="gray")
            ax[1, 2].set_title("Cropped and Resized Sign")
            ax[1, 2].axis("off")

            plt.tight_layout()
            plt.show()

        return resized_image, cropped_original

    except Exception as e:
        logging.error(f"Error while detecting cardboard and sign: {e}")
        raise

def load_data(image_directory='./2024F_imgs', image_type='.png', image_size=(33, 33), grid_size=(8, 8), random_seed=42, split_ratio=0.8, k_fold=True):
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
    test_image_paths = []  # List to store paths of test images
    for line in train_lines:
        img_path = os.path.join(image_directory, line[0] + image_type)
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"Failed to load image: {img_path}. Skipping.")
            continue

        # Extract contour features
        # if line[1] == '0':
        #     img_contour, cropped_original = cv2.resize(img, image_size), cv2.resize(img, image_size)

        # else:
        img_contour, cropped_original = extract_sign_from_cardboard(img, image_size)
        img_contour = img_contour.flatten()

        # Extract spatial gradient features
        spatial_features = compute_spatial_gradient_features(cropped_original, grid_size)

        # Extract edge features
        edge_features = compute_edge_features(cropped_original)

        # Combine all features
        combined_features = np.hstack([img_contour, spatial_features, edge_features])

        train_data.append(combined_features)
        train_labels.append(int(line[1]))
        
    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)
        
    logging.info(f"Done splitting training data: {len(train_data)} training samples.")
    
    if not k_fold:

        test_data, test_labels = [], []
        for line in test_lines:
            img_path = os.path.join(image_directory, line[0] + image_type)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Failed to load image: {img_path}. Skipping.")
                continue

            # Extract contour features
            img_contour, cropped_original = extract_sign_from_cardboard(img, image_size)
            img_contour = img_contour.flatten()

            # Extract spatial gradient features
            spatial_features = compute_spatial_gradient_features(cropped_original, grid_size)

            # Extract edge features
            edge_features = compute_edge_features(cropped_original)

            # Combine all features
            combined_features = np.hstack([img_contour, spatial_features, edge_features])

            test_data.append(combined_features)
            test_labels.append(int(line[1]))
            test_image_paths.append(img_path)  # Store the image path

        # Convert lists to numpy arrays
        # train_data = np.array(train_data, dtype=np.float32)
        # train_labels = np.array(train_labels, dtype=np.int32)
        test_data = np.array(test_data, dtype=np.float32)
        test_labels = np.array(test_labels, dtype=np.int32)
    
    # Normalize data using StandardScaler
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    if not k_fold:
        test_data = scaler.transform(test_data)

        logging.info(f"Data loading completed: {len(train_data)} training samples, {len(test_data)} testing samples.")
    
        return train_data, train_labels, test_data, test_labels, test_image_paths
    
    return train_data, train_labels, None, None, None