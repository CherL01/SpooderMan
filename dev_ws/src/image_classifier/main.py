#!/usr/bin/env python3
from data_loader import load_data
from classifier import train_classifier, evaluate_classifier
import utils

def main():
    # Load the data
    train_data, train_labels, test_data, test_labels = load_data()

    # Train the classifier
    knn = train_classifier(train_data, train_labels)

    # Evaluate the classifier
    accuracy, confusion_matrix = evaluate_classifier(knn, test_data, test_labels)

    # Print results
    print(f"\nTotal accuracy: {accuracy:.2%}")
    print("Confusion Matrix:")
    utils.print_confusion_matrix(confusion_matrix)

if __name__ == "__main__":
    main()