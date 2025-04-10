import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import io
from mnist_app.model.train_model import MNISTModel

class MNISTPredictor:
    def __init__(self, model_path='mnist_app/model/saved_model/mnist_cnn.pth'):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = MNISTModel().to(self.device)
        
        # Load trained parameters
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define image transformations (same as used in training)
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def preprocess_image(self, image_data):
        """
        Preprocess the image data for the model.
        
        Args:
            image_data: Can be a BytesIO object, numpy array, or file path
            
        Returns:
            Tensor ready for model prediction
        """
        if isinstance(image_data, bytes) or isinstance(image_data, io.BytesIO):
            # If image is provided as bytes or BytesIO
            if isinstance(image_data, bytes):
                image_data = io.BytesIO(image_data)
            image = Image.open(image_data).convert('L')  # Convert to grayscale
        elif isinstance(image_data, np.ndarray):
            # If image is provided as numpy array
            if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                # Convert RGB to grayscale
                image_data = np.mean(image_data, axis=2).astype(np.uint8)
            image = Image.fromarray(image_data.astype(np.uint8))
        elif isinstance(image_data, str):
            # If image is provided as a file path
            image = Image.open(image_data).convert('L')
        else:
            raise ValueError("Unsupported image data format")
        
        # Apply transformations
        tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
        
    def predict(self, image_data):
        """
        Predict the digit in the image and return digit and confidence.
        
        Args:
            image_data: Image data in various formats (bytes, BytesIO, numpy array, file path)
            
        Returns:
            tuple: (predicted_digit, confidence_score)
        """
        # Preprocess the image
        tensor = self.preprocess_image(image_data)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1)
            
            # Get predicted class and confidence
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            return predicted_class, confidence

# Example usage
if __name__ == "__main__":
    predictor = MNISTPredictor()
    # Test with a sample image - you would replace this with your own test
    # image_path = "path_to_test_image.png"
    # predicted_digit, confidence = predictor.predict(image_path)
    # print(f"Predicted digit: {predicted_digit}, Confidence: {confidence:.4f}") 