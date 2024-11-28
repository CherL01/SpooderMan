import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def load_data(folder_path, contour_threshold=100, visualize=False):
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

            # Brighten the image by increasing its brightness (a simple way is by adding a constant)
            image = cv2.convertScaleAbs(image, alpha=1.2, beta=20)  # alpha controls contrast, beta controls brightness

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

                debug_image = image.copy()
                
                # Visualization using matplotlib
                fig, ax = plt.subplots(1, 2, figsize=(15, 5))
                ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                ax[0].set_title("Original Image")
                ax[0].axis("off")

                ax[1].imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                ax[1].set_title("Cropped Image")
                ax[1].axis("off")

                plt.tight_layout()
                plt.show()

            # Resize the cropped or original image to the target size
            resized_image = cv2.resize(cropped_image, (224, 224))
            resized_image = resized_image.astype(np.float32) / 255.0  # Normalize to [0, 1]
            
            # Append to the lists
            images.append(resized_image)
            labels.append(int(label))

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

if __name__ == "__main__":
    train_folder = os.path.abspath(os.path.join("dev_ws", "src", "image_classifier", "2024F_imgs"))
    images, labels = load_data(train_folder, visualize=True)