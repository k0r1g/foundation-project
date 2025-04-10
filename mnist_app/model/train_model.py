import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the CNN model
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)  # Increased dropout for better generalization

    def forward(self, x):
        # First conv layer + batch norm + activation + pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv layer + batch norm + activation + pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third conv layer + batch norm + activation + pooling
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 3 * 3)
        
        # First fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc2(x)
        return x

def train_model(epochs=10, batch_size=64, learning_rate=0.001, save_path='mnist_app/model/saved_model/mnist_cnn.pth'):
    # Set up device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define data transformations with augmentation for better generalization
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST dataset mean and std
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = MNISTModel().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler to reduce LR as training progresses
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

    # Training loop
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print statistics
            if batch_idx % 100 == 99:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/100:.4f}')
                train_losses.append(running_loss/100)
                running_loss = 0.0
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total
        test_losses.append(avg_test_loss)
        test_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
        
        # Update learning rate
        scheduler.step(avg_test_loss)
        
        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f'Improved accuracy from {best_accuracy:.2f}% to {accuracy:.2f}%, saving model to {save_path}')

    print(f'Best Test Accuracy: {best_accuracy:.2f}%')
    
    # Plot training and test loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.title('Losses')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.legend()
    plt.title('Test Accuracy')
    
    # Save the plot
    os.makedirs('mnist_app/model/plots', exist_ok=True)
    plt.savefig('mnist_app/model/plots/training_plots.png')
    
    return model

# Function to test the model on some sample images
def test_with_sample_images(model_path='mnist_app/model/saved_model/mnist_cnn.pth'):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load some test samples
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
    
    # Select 10 random samples
    indices = np.random.choice(len(test_dataset), 10, replace=False)
    samples = [test_dataset[i] for i in indices]
    
    # Plot the samples and predictions
    plt.figure(figsize=(15, 8))
    for i, (image, label) in enumerate(samples):
        image = image.to(device).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][pred].item()
        
        plt.subplot(2, 5, i+1)
        plt.imshow(image.cpu().squeeze().numpy(), cmap='gray')
        plt.title(f"True: {label}, Pred: {pred}\nConf: {confidence:.2f}")
        plt.axis('off')
    
    os.makedirs('mnist_app/model/plots', exist_ok=True)
    plt.savefig('mnist_app/model/plots/sample_predictions.png')
    plt.close()
    
    print("Sample predictions saved to mnist_app/model/plots/sample_predictions.png")

if __name__ == "__main__":
    train_model(epochs=10)
    test_with_sample_images() 
    