"""
Script to test model performance on MNIST test data.
Run this script to diagnose model performance issues.
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from mnist_app.model.train_model import MNISTModel

def test_model(model_path='mnist_app/model/saved_model/mnist_cnn.pth'):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    model = MNISTModel().to(device)
    print("Model architecture:")
    print(model)
    
    # Load trained model
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.eval()
    
    # Load test data
    test_dataset = datasets.MNIST('data', train=False, download=True, 
                                  transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Evaluate on test set
    correct = 0
    total = 0
    all_predicted = []
    all_true = []
    
    print("Running test evaluation...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Print information about image data
            print(f"Input shape: {images.shape}")
            print(f"Input min: {images.min().item()}, max: {images.max().item()}, mean: {images.mean().item()}")
            
            # Flatten images if required by the model
            if hasattr(model, 'forward') and 'view' in str(model.forward):
                # Model expects flattened input
                flattened_images = images.view(images.size(0), -1)
                outputs = model(flattened_images)
                print(f"Using flattened images: {flattened_images.shape}")
            else:
                # Model handles the reshaping internally
                outputs = model(images)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Update counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Save for confusion matrix
            all_predicted.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")
    
    # Examine per-digit accuracy
    cm = np.zeros((10, 10), dtype=int)
    for i in range(len(all_true)):
        cm[all_true[i]][all_predicted[i]] += 1
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Calculate per-digit accuracy
    print("\nPer-digit accuracy:")
    for i in range(10):
        digit_correct = cm[i][i]
        digit_total = np.sum(cm[i])
        digit_accuracy = 100 * digit_correct / digit_total
        print(f"Digit {i}: {digit_accuracy:.2f}%")
    
    # Test with some sample images
    print("\nTesting with sample images:")
    
    # Get 10 samples (one of each digit if possible)
    samples = []
    sample_indices = []
    digits_found = set()
    
    # Try to get one of each digit
    for i, (image, label) in enumerate(test_dataset):
        if label.item() not in digits_found and len(samples) < 10:
            samples.append((image, label.item()))
            sample_indices.append(i)
            digits_found.add(label.item())
            
            if len(digits_found) == 10:
                break
    
    # Create a figure to show samples and predictions
    plt.figure(figsize=(15, 8))
    
    for i, (image, label) in enumerate(samples):
        # Prepare for model
        image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension
        
        if hasattr(model, 'forward') and 'view' in str(model.forward):
            # Model expects flattened input
            image_tensor = image_tensor.view(1, -1)
        
        # Get prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][pred].item()
        
        # Plot
        plt.subplot(2, 5, i+1)
        plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f"True: {label}, Pred: {pred}\nConf: {confidence:.2f}")
        plt.axis('off')
    
    # Create directory for plots if it doesn't exist
    os.makedirs('debug_output', exist_ok=True)
    plt.savefig('debug_output/sample_test.png')
    print(f"Sample test saved to debug_output/sample_test.png")

if __name__ == "__main__":
    test_model() 