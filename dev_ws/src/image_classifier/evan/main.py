import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image

# Paths
DATASET_PATH = os.path.abspath(os.path.join("dev_ws", "src", "image_classifier", "2024F_imgs_sorted"))
TEST_DATASET_PATH = "test_data"
MODEL_PATH = os.path.abspath(os.path.join("dev_ws", "src", "image_classifier", "evan", "trained_cnn_model.pth"))

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation for preprocessing images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

# Define the CNN Model (Pretrained ResNet18)
def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer
    return model

# Train the model
def train_model(model, dataloader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")
    return model

# Evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, conf_matrix

# Save the trained model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Load a pre-trained model
def load_model(path, num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    return model

# Perform K-Fold Cross-Validation
def k_fold_validation(dataset, k=5):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_conf_matrices = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold + 1}/{k}")
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        model = get_model(num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        model = train_model(model, train_loader, criterion, optimizer, EPOCHS)
        accuracy, conf_matrix = evaluate_model(model, val_loader)
        fold_accuracies.append(accuracy)
        fold_conf_matrices.append(conf_matrix)

        print(f"Fold {fold + 1} Accuracy: {accuracy * 100:.2f}%")

    # Average Accuracy
    avg_accuracy = sum(fold_accuracies) / k
    print(f"\nAverage K-Fold Accuracy: {avg_accuracy * 100:.2f}%")
    return avg_accuracy, fold_conf_matrices

# Test the model with external test data (if exists)
def test_with_external_data(model):
    if os.path.exists(TEST_DATASET_PATH):
        print("Found test_data folder. Testing on external dataset...")
        test_dataset = datasets.ImageFolder(root=TEST_DATASET_PATH, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        accuracy, conf_matrix = evaluate_model(model, test_loader)

        print(f"Test Data Accuracy: {accuracy * 100:.2f}%")
        print("Confusion Matrix:")
        print(conf_matrix)
    else:
        print("No test_data folder found. Skipping external testing.")

# Main function
def main():
    if os.path.exists(MODEL_PATH):
        print("Loading pre-trained model...")
        model = load_model(MODEL_PATH, num_classes)
    else:
        print("Training a new model...")
        model = get_model(num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        model = train_model(model, train_loader, criterion, optimizer, EPOCHS)
        save_model(model, MODEL_PATH)

    # K-Fold Cross-Validation
    print("\nPerforming K-Fold Cross-Validation...")
    avg_accuracy, conf_matrices = k_fold_validation(dataset)

    print("\nConfusion Matrices for each fold:")
    for i, cm in enumerate(conf_matrices):
        print(f"Fold {i + 1}:")
        print(cm)

    # Test with external dataset (if exists)
    print("\nTesting on External Dataset (if available)...")
    test_with_external_data(model)

# Execute main
if __name__ == "__main__":
    main()
