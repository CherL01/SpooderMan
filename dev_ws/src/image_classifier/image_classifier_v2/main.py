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

# # Function to preprocess the images and labels
# def load_data(folder_path):
#     labels_path = os.path.join(folder_path, "labels.txt")
#     images = []
#     labels = []
    
#     # Read the labels file
#     with open(labels_path, "r") as file:
#         for line in file:
#             image_name, label = line.strip().split(",")
#             image_path = os.path.join(folder_path, f"{image_name}.png")
#             image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
#             image = tf.keras.preprocessing.image.img_to_array(image)
#             images.append(image)
#             labels.append(int(label))
    
#     images = np.array(images) / 255.0  # Normalize images to [0, 1]
#     labels = np.array(labels)
#     return images, labels

def load_data(folder_path, contour_threshold=3000, visualize=False):
    """
    Loads image data and labels, detects and processes contours of specific colors,
    and crops the image based on the contour size threshold.

    Args:
        folder_path (str): Path to the folder containing the images and labels file.
        contour_threshold (int): Minimum contour area to crop around. Images with smaller contours are kept original.

    Returns:
        tuple: Normalized image data (numpy array) and corresponding labels (numpy array).
    """
    labels_path = os.path.join(folder_path, "labels.txt")
    images = []
    labels = []

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

    
    # Read the labels file
    with open(labels_path, "r") as file:
        for line in file:
            image_name, label = line.strip().split(",")
            image_path = os.path.join(folder_path, f"{image_name}.png")
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
            labels.append(int(label))

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

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

# # Train the model
# def train_model(train_folder, save_path):
#     images, labels = load_data(train_folder, visualize=False)
    
#     # Split into training and validation sets
#     split_index = int(0.8 * len(images))
#     train_images, val_images = images[:split_index], images[split_index:]
#     train_labels, val_labels = labels[:split_index], labels[split_index:]
    
#     # Data augmentation
#     datagen = ImageDataGenerator(
#         rotation_range=2, 
#         width_shift_range=0.1, 
#         height_shift_range=0.1, 
#         horizontal_flip=False
#     )
#     datagen.fit(train_images)
    
#     model = build_model()
#     model.fit(datagen.flow(train_images, train_labels, batch_size=16),
#               validation_data=(val_images, val_labels),
#               epochs=10)
    
#     model.save(save_path)
#     return model

# Train the model
def train_model(train_folder, save_path, augmented_folder="augmented_images"):
    images, labels = load_data(train_folder, visualize=False)
    
    # # Split into training and validation sets
    # split_index = int(0.8 * len(images))
    # train_images, val_images = images[:split_index], images[split_index:]
    # train_labels, val_labels = labels[:split_index], labels[split_index:]
    
    # Combine images and labels to shuffle them together
    data = list(zip(images, labels))

    # Shuffle the combined data
    np.random.seed(42)  # Set seed for reproducibility
    np.random.shuffle(data)

    # Unzip the shuffled data back into images and labels
    shuffled_images, shuffled_labels = zip(*data)

    # Convert them back to numpy arrays
    shuffled_images = np.array(shuffled_images)
    shuffled_labels = np.array(shuffled_labels)

    # Perform the random split
    split_index = int(0.8 * len(shuffled_images))
    train_images, val_images = shuffled_images[:split_index], shuffled_images[split_index:]
    train_labels, val_labels = shuffled_labels[:split_index], shuffled_labels[split_index:]
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=2, 
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        horizontal_flip=False
    )
    
    # Generate and save augmented images
    if not os.path.exists(augmented_folder):
        os.makedirs(augmented_folder)
    
    augmented_images = []
    augmented_labels = []
    
    for i, (augmented_image_batch, label_batch) in enumerate(datagen.flow(train_images, train_labels, batch_size=1)):
        augmented_image = augmented_image_batch[0]
        augmented_label = label_batch[0]
        
        # Save the augmented image to the folder
        image_path = os.path.join(augmented_folder, f"augmented_{i}.png")
        array_to_img(augmented_image).save(image_path)
        
        # Add to augmented dataset
        augmented_images.append(augmented_image)
        augmented_labels.append(augmented_label)
        
        # Limit the number of augmented images for demonstration
        if len(augmented_images) >= len(train_images) * 2:  # Augment twice the training data
            break
    
    # Convert augmented data to numpy arrays
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    
    # Combine original and augmented data
    combined_train_images = np.concatenate((train_images, augmented_images), axis=0)
    combined_train_labels = np.concatenate((train_labels, augmented_labels), axis=0)
    
    model = build_model()
    model.fit(datagen.flow(combined_train_images, combined_train_labels, batch_size=16),
              validation_data=(val_images, val_labels),
              epochs=30)
    
    model.save(save_path)
    return model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def evaluate_model(model, test_folder, visualize_misclassified=False):
    # Load test data
    images, labels = load_data(test_folder, visualize=False)

    # Predict labels for the test data
    predicted_labels = model.predict(images)

    # Convert predicted labels to class indices (the index with the highest probability)
    predicted_class_indices = np.argmax(predicted_labels, axis=1)

    # Map numeric labels to their class names
    class_labels = {0: 'Nothing', 1: 'Left', 2: 'Right', 3: 'U-Turn', 4: 'Stop', 5: 'Goal'}
    
    # Print accuracy score
    accuracy = accuracy_score(labels, predicted_class_indices)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Compute confusion matrix
    cm = confusion_matrix(labels, predicted_class_indices)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels.values(), yticklabels=class_labels.values())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Find misclassified images
    misclassified_indices = np.where(predicted_class_indices != labels)[0]
    print(f"Number of misclassified samples: {len(misclassified_indices)}")

    if visualize_misclassified:
    # Display the misclassified images with titles showing the true and predicted labels
        for i in misclassified_indices:
            # Get the image, true label, and predicted label
            img = images[i]

            path = f'{test_folder}/{i}.png'
            
            # Use mpimg.imread to read and display the image
            img = mpimg.imread(path)
            
            true_label = class_labels[labels[i]]
            pred_label = class_labels[predicted_class_indices[i]]
            
            # Display the image with misclassification details
            plt.figure()
            plt.imshow(img)  # Matplotlib expects RGB format
            plt.title(f"True: {true_label}, Predicted: {pred_label}")
            plt.axis('off')  # Hide axes
            plt.show()

    



if __name__ == "__main__":
    model_path = os.path.abspath(os.path.join("dev_ws", "src", "image_classifier", "image_classifier_v2", "classifier_model.h5"))
    train_folder = os.path.abspath(os.path.join("dev_ws", "src", "image_classifier", "2024F_imgs"))
    
    # Check if model exists
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)
    else:
        print("No existing model found. Training a new model...")
        model = train_model(train_folder, model_path)
    
    # Prompt user for test folder
    test_folder = input("\nEnter the path of the test folder: ").strip()
    
    # Evaluate the model on the test dataset
    print("\nEvaluating the model...")
    evaluate_model(model, test_folder)

