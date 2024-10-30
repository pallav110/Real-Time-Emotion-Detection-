import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from models.fer_model import FERModel  # Import the actual FERModel used during training
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class EmotionModel(nn.Module):
    def __init__(self, num_classes=7):  # Assuming 7 emotion classes
        super(EmotionModel, self).__init__()
        # Match architecture to saved model
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Adjusted to 32 filters
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Added to match saved model
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # Adjusted for 128 hidden nodes
        self.fc2 = nn.Linear(128, num_classes)  # Final layer for class output

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)  # Flatten to match fully connected layer input size
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def evaluate_model(model_path, test_data_path):
    # Check device compatibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and ensure the architecture matches the saved model
    model = FERModel()  # Initialize with any required parameters if needed
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define the transformations for the test set
    transform = transforms.Compose([
        transforms.Grayscale(),  # Ensure images are in grayscale
        transforms.Resize((48, 48)),  # Resize to match model's expected input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalize if needed
    ])

    # Load the test dataset
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize evaluation metrics
    correct = 0
    total = 0

    # Optional: Initialize per-class accuracy tracking if desired
    class_correct = [0 for _ in range(len(test_dataset.classes))]
    class_total = [0 for _ in range(len(test_dataset.classes))]

    # Evaluation loop
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Update overall accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update per-class accuracy if desired
            for i in range(len(labels)):
                label = labels[i]
                class_total[label] += 1
                class_correct[label] += (predicted[i] == label).item()

    # Calculate overall accuracy
    accuracy = 100 * correct / total
    print(f'Overall Accuracy on the test set: {accuracy:.2f}%')

    # Calculate and display per-class accuracy
    for i, class_name in enumerate(test_dataset.classes):
        if class_total[i] > 0:
            class_accuracy = 100 * class_correct[i] / class_total[i]
            print(f'Accuracy for class {class_name}: {class_accuracy:.2f}%')

if __name__ == "__main__":
    # Specify the path to your trained model and test dataset
    model_path = 'models/fer_model.pth'  # Change this to your model path
    test_data_path = 'data/test1'  # Change this to your test data path

    evaluate_model(model_path, test_data_path)
