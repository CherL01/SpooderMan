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
            # First red range (0-10)
            [(0, 120, 130), (2, 255, 255)],
            # Second red range (160-180)
            [(160, 100, 130), (179, 255, 255)]
        ],
        "green": [[(50, 40, 50), (100, 255, 255)]],
        "blue": [[(100, 100, 50), (150, 255, 255)]]
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
        return image, combined_mask, masks

    # Detect the largest contour above the threshold
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    max_side = max(w, h)
    cropped_image = image[y:y+h, x:x+w]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_HSV2BGR)
    combined_mask = combined_mask[y:y+h, x:x+w]
    
    return cropped_image, combined_mask, masks

def load_data(folder_path, visualize=False):
    labels_path = os.path.join(folder_path, "labels.txt")
    images = []
    labels = []
    
    # Read the labels file
    with open(labels_path, "r") as file:
        for line in file:
            image_name, label = line.strip().split(",")
            image_path = os.path.join(folder_path, f"{image_name}.png")
            
            # Load the original image
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"Failed to load {image_path}")
                continue
            
            # normalize the image brightness
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
            y, x, _ = original_image.shape
            min_brightness = np.min(original_image[:,:,2])
            max_brightness = np.max(original_image[y//2 - 30:y//2 + 30,x//2 - 150:x//2 + 150,2])
            original_image[:,:,2] = np.clip((original_image[:,:,2] - min_brightness) / (max_brightness - min_brightness) * 255, 0, 255)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_HSV2BGR)
            
            cropped_image, combined_mask, masks = extract_hsv_color_mask(original_image)

            # Adjust brightness and contrast
            # adjusted_image = adjust_brightness_contrast(original_image)

            # Resize image
            # resized_image = cv2.resize(adjusted_image, (64, 64))
            # resized_image = adjusted_image
            # cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            # # Extract Hough transform visualization
            # hough_image = extract_hough_features(combined_mask)
            # keypoints, descriptors, sift_image = extract_sift_features(combined_mask)
            
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
            
            # # Normalize resized image for model input
            # resized_normalized = cropped_image / 255.0
            # resized_normalized_flattened = resized_normalized.flatten()
            
            # cropped_image = cv2.resize(cropped_image, (64, 64)).flatten()
            # combined_mask = cv2.resize(combined_mask, (64, 64)).flatten()
            
            # cropped_image_resized = cropped_image_resized.flatten()
            # combined_mask_resized = combined_mask_resized.flatten()
            
            # combined_features = np.concatenate([cropped_image, combined_mask, cropped_image_resized, combined_mask_resized])
            
            # images.append(combined_features)
            # labels.append(int(label))
            
            cropped_image = cv2.resize(cropped_image, (224, 224))
            images.append(cropped_image)
            labels.append(int(label))
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def main():
    # Paths to training and testing folders
    train_folder = os.path.abspath(os.path.join("dev_ws", "src", "image_classifier", "2024F_imgs"))
    train_folder2 = os.path.abspath(os.path.join("dev_ws", "src", "image_classifier", "test_data"))
    test_folder = os.path.abspath(os.path.join("dev_ws", "src", "image_classifier", "test_data"))
    SVM_SAVE_PATH = os.path.abspath(os.path.join("dev_ws", "src", "image_classifier", "image_classifier_v3", "svm_model_all_data_preprocessed.pkl"))  # Path to save the trained SVM model
    
    # Load training data
    print("Loading training data...")
    train_data, train_labels = load_data(train_folder)
    print(f"Training data shape: {train_data.shape}, Labels: {len(train_labels)}")

    print("Loading training data...")
    train_data2, train_labels2 = load_data(train_folder2)
    print(f"Training data shape: {train_data2.shape}, Labels: {len(train_labels2)}")

    train_data = np.concatenate((train_data, train_data2), axis=0)
    train_labels = np.concatenate((train_labels, train_labels2), axis=0)
    
    # Load testing data
    print("Loading testing data...")
    test_data, test_labels = load_data(test_folder)
    print(f"Testing data shape: {test_data.shape}, Labels: {len(test_labels)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)
    
    # Shuffle training data for randomness
    train_data, train_labels_encoded = shuffle(train_data, train_labels_encoded, random_state=42)

    # Load a pre-trained ResNet model
    resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    train_data = preprocess_input(train_data)
    test_data = preprocess_input(test_data)
    
    # Extract features
    features = resnet.predict(train_data, batch_size=16)
    test_features = resnet.predict(test_data, batch_size=16)
    
    # Train an SVM
    print("Training SVM...")
    classifier = svm.SVC(kernel="linear", C=1.0)
    classifier.fit(features, train_labels_encoded)
    print("SVM training completed.")
    
    # Test the SVM on training data
    print("Evaluating SVM on training set...")
    predictions = classifier.predict(features)
    accuracy = accuracy_score(train_labels_encoded, predictions)
    print(f"Accuracy: {accuracy:.2%}")
    
    # Print a classification report
    print("Classification Report:")
    target_names = [str(label) for label in label_encoder.classes_]  # Convert labels to strings
    print(classification_report(train_labels_encoded, predictions, target_names=target_names))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(train_labels_encoded, predictions)
    print(cm)
    
    # Test the SVM
    print("Evaluating SVM...")
    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_labels_encoded, predictions)
    print(f"Accuracy: {accuracy:.2%}")
    
    # Print a classification report
    print("Classification Report:")
    target_names = [str(label) for label in label_encoder.classes_]  # Convert labels to strings
    print(classification_report(test_labels_encoded, predictions, target_names=target_names))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels_encoded, predictions)
    print(cm)
    
    # Visualize misclassified images
    misclassified_indices = np.where(predictions != test_labels_encoded)[0]  # Indices of misclassified samples

    print(f"\nNumber of misclassified samples: {len(misclassified_indices)}")
    
    # Save the trained SVM
    joblib.dump(classifier, SVM_SAVE_PATH)
    print(f"SVM model saved to {SVM_SAVE_PATH}")

if __name__ == "__main__":
    main()