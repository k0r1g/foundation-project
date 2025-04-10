import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import io
import os
from mnist_app.model.train_model import MNISTModel

class MNISTPredictor:
    def __init__(self, model_path='mnist_app/model/saved_model/mnist_cnn.pth'):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEBUG: Initializing MNISTPredictor with device {self.device}")
        
        # Initialize model
        self.model = MNISTModel().to(self.device)
        
        # Load trained parameters
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"DEBUG: Model loaded from {model_path}")
        except Exception as e:
            print(f"DEBUG ERROR: Error loading model: {e}")
            raise
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define image transformations - Normalization added
        self.transform = transforms.Compose([
            # Cropping/Centering is now done *before* this transform pipeline
            # transforms.Resize((28, 28)), # Resizing handled by centering logic
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST normalization
        ])

    def preprocess_image(self, image_data):
        """
        Preprocess the image data for the model, including cropping and centering.
        
        Args:
            image_data: Can be a BytesIO object, numpy array, or file path
            
        Returns:
            Tensor ready for model prediction
        """
        print(f"DEBUG: Preprocessing image of type {type(image_data)}")
        
        pil_image = None # Initialize pil_image
        
        if isinstance(image_data, bytes) or isinstance(image_data, io.BytesIO):
            if isinstance(image_data, bytes):
                image_data = io.BytesIO(image_data)
            pil_image = Image.open(image_data).convert('L')
            print(f"DEBUG: Processed bytes/BytesIO to grayscale image of size {pil_image.size}")
        elif isinstance(image_data, np.ndarray):
            print(f"DEBUG: Processing numpy array of shape {image_data.shape} and dtype {image_data.dtype}")
            print(f"DEBUG: Array min: {image_data.min()}, max: {image_data.max()}, mean: {image_data.mean()}")
            
            # Ensure the array is uint8 and grayscale
            if len(image_data.shape) == 3:
                if image_data.shape[2] == 4: # RGBA
                     # Simple conversion: Use alpha blending on black background
                     # Or just take one channel if alpha indicates opaque white on black
                     # Assuming canvas gives opaque white (255) on black (0)
                     image_data = image_data[:, :, 0] # Take first channel (assuming it's white)
                elif image_data.shape[2] == 3: # RGB
                     # Convert RGB to grayscale using standard luminosity formula
                     image_data = np.dot(image_data[...,:3], [0.2989, 0.5870, 0.1140])
            
            # Ensure data type is uint8
            if image_data.dtype != np.uint8:
                 # Scale if necessary (e.g., if it's float 0-1)
                 if image_data.max() <= 1.0 and image_data.min() >= 0.0:
                     image_data = (image_data * 255).astype(np.uint8)
                 else:
                     image_data = image_data.astype(np.uint8)
            
            pil_image = Image.fromarray(image_data)
            print(f"DEBUG: Converted numpy array to PIL Image of size {pil_image.size}")
            
        elif isinstance(image_data, str):
            pil_image = Image.open(image_data).convert('L')
            print(f"DEBUG: Loaded image from path, size: {pil_image.size}")
        else:
            print(f"DEBUG ERROR: Unsupported image data format: {type(image_data)}")
            raise ValueError("Unsupported image data format")

        # --- Cropping and Centering Logic --- 
        try:
            # Find bounding box of the digit (non-black pixels)
            bbox = pil_image.getbbox()
            
            if bbox:
                # Crop image to bounding box
                cropped_image = pil_image.crop(bbox)
                print(f"DEBUG: Cropped to bounding box: {bbox}")

                # --- Resize cropped image to fit within a 20x20 box (for padding) --- 
                target_size = 20 
                original_width, original_height = cropped_image.size
                
                # Calculate aspect ratio
                ratio = min(target_size / original_width, target_size / original_height)
                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)
                
                # Ensure dimensions are at least 1x1
                new_width = max(1, new_width)
                new_height = max(1, new_height)
                
                # Resize using LANCZOS for better quality
                resized_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"DEBUG: Resized cropped image to: {new_width}x{new_height}")

                # --- Create a new 28x28 black image --- 
                final_image = Image.new("L", (28, 28), 0) # 'L' for grayscale, 0 for black background

                # --- Calculate position to paste the resized image in the center --- 
                paste_x = (28 - new_width) // 2
                paste_y = (28 - new_height) // 2

                # --- Paste the resized image onto the center --- 
                final_image.paste(resized_image, (paste_x, paste_y))
                pil_image = final_image # Use this centered image for further processing
                print(f"DEBUG: Pasted resized image onto center of 28x28 canvas.")
                
            else:
                # If bounding box is None (empty canvas), create a blank 28x28 image
                pil_image = Image.new("L", (28, 28), 0)
                print("DEBUG: Empty canvas detected, using blank 28x28 image.")

        except Exception as e:
             print(f"DEBUG ERROR: Error during cropping/centering: {e}")
             # Optionally, fall back to original image or raise error
             # For now, let's proceed with the original (potentially uncropped) pil_image if error occurs
             pass 
        # --- End Cropping and Centering --- 

        # Save the final preprocessed PIL image before transform for inspection
        try:
            debug_dir = "/app/debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = f"{debug_dir}/centered_preprocessed_image.png"
            pil_image.save(debug_path)
            print(f"DEBUG: Saved final centered image (before tensor conversion) to {debug_path}")
        except Exception as e:
            print(f"DEBUG ERROR: Could not save debug image: {e}")
        
        # Apply transformations (ToTensor and Normalize)
        tensor = self.transform(pil_image) # Pass the final centered PIL image
        
        # Print tensor stats after transforms
        print(f"DEBUG: Tensor shape after transforms: {tensor.shape}")
        print(f"DEBUG: Tensor min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}, mean: {tensor.mean().item():.4f}")
        
        # Reshape to match model input (flattened vector)
        # Model expects [batch_size, 784]
        tensor = tensor.view(1, -1) 
        print(f"DEBUG: Tensor shape after final flatten: {tensor.shape}")
        
        return tensor.to(self.device)
        
    def predict(self, image_data):
        """
        Predict the digit in the image and return digit and confidence.
        
        Args:
            image_data: Image data in various formats (bytes, BytesIO, numpy array, file path)
            
        Returns:
            tuple: (predicted_digit, confidence_score)
        """
        print(f"DEBUG: Starting prediction")
        
        # Preprocess the image
        tensor = self.preprocess_image(image_data)
        
        # Make prediction
        with torch.no_grad():
            print(f"DEBUG: Running model inference")
            # The model expects flattened input (batch_size, 784)
            output = self.model(tensor)
            print(f"DEBUG: Raw output: {output.cpu().numpy()}")
            
            probabilities = F.softmax(output, dim=1)
            print(f"DEBUG: Probabilities: {probabilities.cpu().numpy()}")
            
            # Get predicted class and confidence
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            print(f"DEBUG: Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
            
            return predicted_class, confidence

# Example usage
if __name__ == "__main__":
    predictor = MNISTPredictor()
    # Test with a sample image - you would replace this with your own test
    # image_path = "path_to_test_image.png"
    # predicted_digit, confidence = predictor.predict(image_path)
    # print(f"Predicted digit: {predicted_digit}, Confidence: {confidence:.4f}") 