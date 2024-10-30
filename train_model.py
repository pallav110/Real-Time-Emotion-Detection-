import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Import your FER model here
from models.fer_model import FERModel  # Make sure this is correctly implemented

# Custom Dataset class
class FERDataset(Dataset):
    def __init__(self, root_dir):
        self.data = ImageFolder(root_dir, transform=transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, label

def load_data(train_dir, test_dir):
    train_dataset = FERDataset(train_dir)
    test_dataset = FERDataset(test_dir)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, train_dataset  # Return the dataset for class access

def train_model():
    train_dir = 'data/train1'
    test_dir = 'data/test1'
    train_loader, test_loader, train_dataset = load_data(train_dir, test_dir)

    num_classes = len(train_dataset.data.classes)  # Access classes correctly
    model = FERModel(num_classes=num_classes).to('cuda')  # Ensure you have a proper model definition

    # Initialize loss function and optimizer
    class_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.data.targets), y=train_dataset.data.targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to('cuda')
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Adjusted learning rate

    # Training loop
    num_epochs = 10  # Number of epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')  # Change to 'cpu' if not using GPU
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), 'models/fer_model.pth')
    print('Model trained and saved as fer_model.pth')

if __name__ == '__main__':
    train_model()
