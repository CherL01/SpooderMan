#!/usr/bin/env python3
import logging
from data_loader import load_data
from classifier import train_classifier, evaluate_classifier
import utils
import os

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

    # Step 1: Load the data
    logging.info("Loading data...")
    try:
        train_data, train_labels, test_data, test_labels = load_data(image_directory=os.path.abspath(os.path.join("dev_ws", "src", "image_classifier", "2024F_imgs")))
        logging.info("Data successfully loaded.")
        logging.info(f"Training data size: {len(train_data)}, Test data size: {len(test_data)}")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    # Step 2: Train the classifier
    logging.info("Training the classifier...")
    try:
        knn = train_classifier(train_data, train_labels)
        logging.info("Classifier training completed.")
    except Exception as e:
        logging.error(f"Failed to train the classifier: {e}")
        return

    # Step 3: Evaluate the classifier
    logging.info("Evaluating the classifier...")
    try:
        accuracy, confusion_matrix = evaluate_classifier(knn, test_data, test_labels)
        logging.info("Evaluation completed.")
    except Exception as e:
        logging.error(f"Failed to evaluate the classifier: {e}")
        return

    # Step 4: Print results
    logging.info("Printing results...")
    print(f"\nTotal accuracy: {accuracy:.2%}")
    print("Confusion Matrix:")
    utils.print_confusion_matrix(confusion_matrix)

    logging.info("Program completed successfully.")

if __name__ == "__main__":
    main()