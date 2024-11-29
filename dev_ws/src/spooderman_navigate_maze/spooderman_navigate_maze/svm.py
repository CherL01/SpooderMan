import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import os
import joblib  # Import joblib for saving and loading models

def extract_hsv_color_mask(image):
    """
    Extracts a color mask from an image using HSV color space.
    
    Args:
        image (np.ndarray): Input BGR image.
        color_ranges (dict): Dictionary of HSV color ranges.
    
    Returns:
        np.ndarray: Color mask.
    """
    
    # Define HSV color ranges for red, orange, blue, and green
    color_ranges = {
        "red": [
            # Second red range (160-180)
            [(160, 100, 130), (179, 255, 255)]
        ],
        "green": [[(50, 40, 50), (100, 255, 255)]],
        "blue": [[(100, 50, 50), (150, 255, 255)]]
    }
    
    # Convert to HSV for color detection
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    masks = []

    # Process each color range
    for color, ranges_list in color_ranges.items():
        for ranges in ranges_list:
            lower, upper = ranges
            mask = cv2.inRange(image, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
            masks.append(mask)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # return image, combined_mask
    
    if not contours:
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image, combined_mask

    # Detect the largest contour above the threshold
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_HSV2BGR)
    combined_mask = combined_mask[y:y+h, x:x+w]
    
    return cropped_image, combined_mask

def load_data(folder_path, visualize=False):

    images = []
    
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png'))]
    # Read the labels file
    for image_path in image_files:
        print('image_path:', image_path)
            
        # Load the original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Failed to load {image_path}")
            continue
        
        # normalize the image brightness
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        y, x, _ = original_image.shape
        min_brightness = np.min(original_image[:,:,2])
        max_brightness = np.max(original_image[y//2 - 25:y//2 + 25,x//2 - 25:x//2 + 25,2])
        original_image[:,:,2] = np.clip((original_image[:,:,2] - min_brightness) / (max_brightness - min_brightness) * 255, 0, 255)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_HSV2BGR)
        
        cropped_image, combined_mask = extract_hsv_color_mask(original_image)
        
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image_resized = cv2.resize(cropped_image, (16, 16))
        combined_mask_resized = cv2.resize(combined_mask, (16, 16))
        
        if visualize:
            # Plot all stages in a single figure
            fig, ax = plt.subplots(2, 3, figsize=(16, 4))
            ax[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            ax[0, 0].set_title("Original Image")
            ax[0, 0].axis("off")

            ax[0, 1].imshow(cropped_image, cmap="gray")
            ax[0, 1].set_title("Cropped Image")
            ax[0, 1].axis("off")
            
            ax[0, 2].imshow(combined_mask, cmap="gray")
            ax[0, 2].set_title("Combined Mask")
            ax[0, 2].axis("off")
            
            ax[1, 0].imshow(cropped_image_resized, cmap="gray")
            ax[1, 0].set_title("Resized Cropped Image")
            ax[1, 0].axis("off")
            
            ax[1, 1].imshow(combined_mask_resized, cmap="gray")
            ax[1, 1].set_title("Resized Combined Mask")
            ax[1, 1].axis("off")

            plt.tight_layout()
            plt.show()
        
        cropped_image = cv2.resize(cropped_image, (224, 224))
        images.append(cropped_image)
    
    images = np.array(images)
    return images

def evaluate_svm(classifier, test_folder):
    
    # Load testing data
    print("Loading images...")
    test_data = load_data(test_folder)
    
    # Load a pre-trained ResNet model
    resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    
    test_features = resnet.predict(test_data, batch_size=16)
    
    # Test the SVM
    print("Evaluating SVM...")
    predictions = classifier.predict(test_features)
    
    return predictions

def main():
    # Paths to training and testing folders
    test_folder = os.path.abspath(os.path.join("dev_ws", "src", "spooderman_navigate_maze", "maze_images"))
    SVM_SAVE_PATH = os.path.abspath(os.path.join("dev_ws", "src", "image_classifier", "image_classifier_v3", "svm_model.pkl"))  # Path to save the trained SVM model

    # Load the SVM model (for future use)
    loaded_classifier = joblib.load(SVM_SAVE_PATH)
    print("Loaded SVM model.")
    
    # Test the SVM
    print("Evaluating SVM...")
    predictions = evaluate_svm(loaded_classifier, test_folder)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()