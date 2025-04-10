"""
Script to train the MNIST model.
Run this script first to train and save the model before starting the application.
"""

from mnist_app.model.train_model import train_model

if __name__ == "__main__":
    print("Training MNIST model...")
    model = train_model(epochs=5)
    print("Training complete! Model saved to mnist_app/model/saved_model/mnist_cnn.pth") 