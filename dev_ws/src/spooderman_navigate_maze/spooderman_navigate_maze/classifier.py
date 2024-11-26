import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def load_data(folder_path, contour_threshold=3000, visualize=False):

    images = []

    # Define HSV color ranges for red, orange, blue, and green
    color_ranges = {
        "red": [
            # First red range (0-10)
            [(120, 100, 100), (130, 255, 255)],
            # Second red range (160-180)
            [(160, 100, 100), (180, 255, 255)]
        ],
        "green": [[(100, 40, 20), (200, 255, 100)]],
        "blue": [[(150, 50, 50), (255, 255, 255)]]
    }
    
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png'))]

    for image_path in image_files:
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        # Process each color range
        for color, ranges_list in color_ranges.items():
            for ranges in ranges_list:
                lower, upper = ranges
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect the largest contour above the threshold
        valid_contours = [contour for contour in contours if cv2.contourArea(contour) > contour_threshold]

        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_image = image[y:y+h, x:x+w]
            
            if visualize:
                # Visualization: Prepare images for display
                debug_image = image.copy()
                cv2.drawContours(debug_image, [largest_contour], -1, (0, 255, 0), 2)  # Green contour
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue bounding box
                
                # Visualization using matplotlib
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                ax[0].set_title("Original Image")
                ax[0].axis("off")

                ax[1].imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
                ax[1].set_title("Detected Contour")
                ax[1].axis("off")

                ax[2].imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                ax[2].set_title("Cropped Image")
                ax[2].axis("off")

                plt.tight_layout()
                plt.show()
            
        else:
            cropped_image = image  # Keep original image if no valid contour

        # Resize the cropped or original image to the target size
        resized_image = cv2.resize(cropped_image, (224, 224))
        resized_image = resized_image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Append to the lists
        images.append(resized_image)

    images = np.array(images)

    return images

# Build the model
def build_model(num_classes=6):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)
    
    # Set a custom learning rate
    # learning_rate = 0.001  # Adjust this value as needed
    # optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def evaluate_model(model, test_folder, visualize_misclassified=False):
    # Load test data
    images = load_data(test_folder, visualize=False)

    # Predict labels for the test data
    predicted_labels = model.predict(images)
    
    return predicted_labels

# if __name__ == "__main__":
#     model_path = os.path.abspath(os.path.join("dev_ws", "src", "spooderman_navigate_maze", "spooderman_navigate_maze", "classifier_model.h5"))
    
#     # Check if model exists
#     if os.path.exists(model_path):
#         print("Loading existing model...")
#         model = load_model(model_path)
        
#     else:
#         print("Model not found. Please train the model first.")
#         exit()
    
#     # Prompt user for test folder
#     test_folder = os.path.abspath(os.path.join("dev_ws", "src", "spooderman_navigate_maze", "maze_images"))
    
#     # Evaluate the model on the test dataset
#     print("\nEvaluating the model...")
#     evaluate_model(model, test_folder)

