#!/usr/bin/env python3
import logging
from data_loader import load_data
from classifier import train_classifier, evaluate_classifier
import utils
import os
from utils import visualize_misclassified_images
from sklearn.model_selection import KFold
import numpy as np

def main():
    """
    Main function to load data, train the classifier, and evaluate its performance.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Set to DEBUG for more detailed logs
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    logging.info("Starting the program...")
    
    visualize = False  # Set to True to visualize misclassified images

    # # Step 1: Load the data
    # logging.info("Loading data...")
    # try:
    #     train_img_path = os.path.abspath(os.path.join("dev_ws", "src", "image_classifier", "2024F_imgs"))
    #     test_img_path = os.path.abspath(os.path.join("dev_ws", "src", "image_classifier", "test_data"))
    #     random_seed = 42
    #     # Load the entire dataset without splitting
    #     full_data, full_labels, _, _, _ = load_data(train_image_directory=train_img_path, test_image_directory=test_img_path, random_seed=random_seed, split_ratio=1.0)
    #     logging.info(f"Full data size: {len(full_data)}")
    # except Exception as e:
    #     logging.error(f"Failed to load data: {e}")
    #     return

    # # Step 2: K-Fold Cross-Validation
    # k_folds = 5  # Number of folds
    # kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

    # fold_accuracies = []
    # fold_confusion_matrices = []

    # for fold, (train_indices, test_indices) in enumerate(kfold.split(full_data)):
    #     logging.info(f"Starting fold {fold + 1}/{k_folds}...")

    #     # Split data into training and testing sets for this fold
    #     train_data, test_data = full_data[train_indices], full_data[test_indices]
    #     train_labels, test_labels = full_labels[train_indices], full_labels[test_indices]

    #     # Step 3: Train the classifier
    #     logging.info(f"Training the classifier for fold {fold + 1}...")
    #     try:
    #         trained_model = train_classifier(train_data, train_labels, model="KNN")
    #         logging.info(f"Classifier training completed for fold {fold + 1}.")
    #     except Exception as e:
    #         logging.error(f"Failed to train the classifier for fold {fold + 1}: {e}")
    #         continue

    #     # Step 4: Evaluate the classifier
    #     logging.info(f"Evaluating the classifier for fold {fold + 1}...")
    #     try:
    #         accuracy, confusion_matrix, _ = evaluate_classifier(trained_model, test_data, test_labels, k=7, model_type="KNN")
    #         logging.info(f"Evaluation completed for fold {fold + 1}. Accuracy: {accuracy:.2%}")
    #     except Exception as e:
    #         logging.error(f"Failed to evaluate the classifier for fold {fold + 1}: {e}")
    #         continue

    #     # Store fold results
    #     fold_accuracies.append(accuracy)
    #     fold_confusion_matrices.append(confusion_matrix)

    # # Step 5: Compute average metrics across all folds
    # average_accuracy = np.mean(fold_accuracies)
    # total_confusion_matrix = np.sum(fold_confusion_matrices, axis=0)

    # # Print k-fold results
    # logging.info("K-Fold Cross-Validation Results:")
    # print(f"\nAverage Accuracy across {k_folds} folds: {average_accuracy:.2%}")
    # print("Combined Confusion Matrix:")
    # utils.print_confusion_matrix(total_confusion_matrix)
    
    # if visualize:
    #     # Step 5: Visualize misclassified images
    #     logging.info("Visualizing misclassified images...")
    #     visualize_misclassified_images(test_image_paths, predictions, test_labels)
    
    # Step 1: Load the data
    logging.info("Loading data...")
    try:
        train_img_path = os.path.abspath(os.path.join("dev_ws", "src", "image_classifier", "2024F_imgs"))
        test_img_path = os.path.abspath(os.path.join("dev_ws", "src", "image_classifier", "test_data"))
        split_ratio = 0.5
        random_seed = 42
        train_data, train_labels, test_data, test_labels, test_image_paths = load_data(train_image_directory=train_img_path, test_image_directory=test_img_path, random_seed=random_seed, k_fold=False)
        logging.info("Data successfully loaded.")
        logging.info(f"Training data size: {len(train_data)}, Test data size: {len(test_data)}")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return
    
    # Step 2: Train the classifier
    logging.info("Training the classifier...")
    try:
        trained_model = train_classifier(train_data, train_labels, model="KNN")
        logging.info("Classifier training completed.")
    except Exception as e:
        logging.error(f"Failed to train the classifier: {e}")
        return

    # Step 3: Evaluate the classifier
    logging.info("Evaluating the classifier...")
    try:
        accuracy, confusion_matrix, predictions = evaluate_classifier(trained_model, test_data, test_labels, k=3, model_type="KNN")
        logging.info("Evaluation completed.")
    except Exception as e:
        logging.error(f"Failed to evaluate the classifier: {e}")
        return

    # Step 4: Print results
    logging.info("Printing results...")
    print(f"\nTotal accuracy: {accuracy:.2%}")
    print("Confusion Matrix:")
    utils.print_confusion_matrix(confusion_matrix)
    
    # Step 5: Visualize misclassified images
    logging.info("Visualizing misclassified images...")
    visualize_misclassified_images(test_image_paths, predictions, test_labels)

    logging.info("Program completed successfully.")

if __name__ == "__main__":
    main()