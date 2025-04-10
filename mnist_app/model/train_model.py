import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the ResNet model (from train_model2.py)
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.do = nn.Dropout(0.1)

    def forward(self, x):
        # Ensure input is properly flattened
        if len(x.shape) == 4:  # Input is [batch_size, channels, height, width]
            x = x.view(x.size(0), -1)  # Flatten to [batch_size, 784]
            
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        do = self.do(h2 + h1)  # Residual connection
        logits = self.l3(do)
        return logits

def train_model(epochs=5, batch_size=32, learning_rate=0.01, save_path='mnist_app/model/saved_model/mnist_cnn.pth'):
    # Set up device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Add Normalisation to Transforms --- 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST normalization
    ])
    
    # Load and prepare data using the defined transform
    train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    # Split training data into train and validation sets
    train_set, val_set = random_split(train_data, [55000, 5000])
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialise the model
    model = MNISTModel().to(device)
    
    # Define loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Best validation accuracy for model saving
    best_accuracy = 0.0
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # Get batch size and reshape input
            b = x.size(0)
            x = x.view(b, -1)
            
            # 1) Clear previous gradients
            optimizer.zero_grad()
            
            # 2) Forward pass
            logits = model(x)
            
            # 3) Compute loss
            loss = criterion(logits, y)
            
            # 4) Compute gradients
            loss.backward()
            
            # 5) Update weights
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = torch.tensor(train_losses).mean().item()
        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                
                # Reshape input
                b = x.size(0)
                x = x.view(b, -1)
                
                # Forward pass
                logits = model(x)
                
                # Compute loss
                loss = criterion(logits, y)
                val_losses.append(loss.item())
                
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        avg_val_loss = torch.tensor(val_losses).mean().item()
        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
        
        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model improved! Saving model with accuracy: {accuracy:.2f}%")
    
    # Final evaluation on test set
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
    
    test_accuracy = 100 * test_correct / test_total
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    
    return model

# Function to test the model on some sample images
def test_with_sample_images(model_path='mnist_app/model/saved_model/mnist_cnn.pth'):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # --- Apply same transform (including normalization) for testing --- 
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
        # Prepare image for model
        image_tensor = image.to(device).view(1, -1)  # Flatten
        
        # Get prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][pred].item()
        
        # Plot
        plt.subplot(2, 5, i+1)
        plt.imshow(image.squeeze().numpy(), cmap='gray')
        plt.title(f"True: {label}, Pred: {pred}\nConf: {confidence:.2f}")
        plt.axis('off')
    
    # Save plot
    os.makedirs('mnist_app/model/plots', exist_ok=True)
    plt.savefig('mnist_app/model/plots/sample_predictions.png')
    plt.close()
    
    print("Sample predictions saved to mnist_app/model/plots/sample_predictions.png")

if __name__ == "__main__":
    train_model(epochs=5)
    test_with_sample_images() 
    