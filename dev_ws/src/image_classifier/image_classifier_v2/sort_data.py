import os
import shutil

# Paths
dataset_path = os.path.abspath(os.path.join("dev_ws", "src", "image_classifier", "2024F_imgs_sorted"))
labels_file = os.path.join(dataset_path, "labels.txt")

# Map of label indices to class names
class_names = ["Nothing", "Left", "Right", "U-turn", "Stop", "Goal"]

# Create class folders
for class_name in class_names:
    class_dir = os.path.join(dataset_path, class_name)
    os.makedirs(class_dir, exist_ok=True)

# Move images into corresponding class folders
with open(labels_file, "r") as file:
    for line in file:
        image_name, label = line.strip().split(",")
        label = int(label)  # Convert label to an integer
        src_path = os.path.join(dataset_path, f"{image_name}.png")
        dst_path = os.path.join(dataset_path, class_names[label], f"{image_name}.png")
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"Warning: {src_path} does not exist.")