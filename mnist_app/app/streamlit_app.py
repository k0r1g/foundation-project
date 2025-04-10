import streamlit as st
import numpy as np
from PIL import Image
import io
import time
from mnist_app.model.predict import MNISTPredictor
from mnist_app.database.db import Database, check_db_connection

# Initialize the predictor and database
@st.cache_resource
def load_predictor():
    return MNISTPredictor()

@st.cache_resource
def load_database():
    return Database()

# Function to convert the drawing to the right format
def process_drawing(drawing_data):
    """Process the drawing data from the canvas."""
    # The drawing comes as a base64 encoded string
    # We convert it to a numpy array
    if drawing_data is None:
        return None
    
    try:
        # Convert to PIL Image
        image = Image.open(io.BytesIO(drawing_data.encode())).convert('L')
        
        # Convert to numpy array and invert colors (MNIST has white digits on black background)
        img_array = 255 - np.array(image)
        
        return img_array
    except Exception as e:
        st.error(f"Error processing drawing: {e}")
        return None

def main():
    st.title("MNIST Digit Recognizer")
    
    # Load predictor and database
    try:
        predictor = load_predictor()
        db = load_database()
    except Exception as e:
        st.error(f"Error loading predictor or database: {e}")
        return
    
    # Check database connection
    db_connected = check_db_connection()
    if not db_connected:
        st.warning("⚠️ Database connection failed. Predictions will not be logged.")
    
    # Create a canvas for drawing
    st.write("Draw a digit (0-9) in the canvas below:")
    
    # We'll use the streamlit-drawable-canvas component for drawing
    try:
        from streamlit_drawable_canvas import st_canvas
        
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        # Get the drawing data
        if canvas_result.image_data is not None:
            # Convert the RGBA image to grayscale
            img_array = np.mean(canvas_result.image_data, axis=2).astype(np.uint8)
            
            # Display the processed image
            st.image(img_array, caption="Processed Image", width=150)
        else:
            img_array = None
    except ImportError:
        st.error("The streamlit-drawable-canvas package is not installed. Please install it with: pip install streamlit-drawable-canvas")
        
        # As a fallback, use a file uploader
        st.write("Or upload an image of a digit:")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read the image
            image = Image.open(uploaded_file).convert('L')
            img_array = np.array(image)
            
            # Display the uploaded image
            st.image(img_array, caption="Uploaded Image", width=150)
        else:
            img_array = None
    
    # Prediction section
    if st.button("Predict"):
        if img_array is not None:
            # Show loading spinner
            with st.spinner("Predicting..."):
                # Make prediction
                predicted_digit, confidence = predictor.predict(img_array)
                
                # Store the prediction ID for later use
                prediction_id = None
                
                # Log the prediction if database is connected
                if db_connected:
                    try:
                        # Convert image to bytes for storage
                        img_bytes = io.BytesIO()
                        Image.fromarray(img_array).save(img_bytes, format="PNG")
                        
                        # Log to database
                        prediction_id = db.log_prediction(
                            predicted_digit=predicted_digit,
                            confidence=confidence,
                            image_data=img_bytes.getvalue()
                        )
                        st.session_state["last_prediction_id"] = prediction_id
                    except Exception as e:
                        st.error(f"Error logging prediction: {e}")
            
            # Display the prediction results
            st.success(f"Prediction complete!")
            st.subheader(f"Predicted Digit: {predicted_digit}")
            st.subheader(f"Confidence: {confidence:.2%}")
            
            # Feedback section
            st.write("Was this prediction correct? Please provide the true digit:")
            
            # Create a number input for the true label
            true_label = st.number_input(
                "True digit (0-9):",
                min_value=0,
                max_value=9,
                step=1,
                key="true_label"
            )
            
            if st.button("Submit Feedback"):
                if db_connected:
                    try:
                        # Update the prediction with the true label
                        if "last_prediction_id" in st.session_state:
                            db.update_true_label(
                                st.session_state["last_prediction_id"],
                                true_label
                            )
                            st.success("Thank you for your feedback!")
                    except Exception as e:
                        st.error(f"Error updating true label: {e}")
                else:
                    st.warning("Database is not connected. Feedback not saved.")
        else:
            st.warning("Please draw or upload a digit first!")
    
    # Display statistics if database is connected
    if db_connected:
        st.sidebar.title("Statistics")
        try:
            # Get accuracy statistics
            stats = db.get_accuracy_stats()
            
            # Display statistics
            st.sidebar.metric("Accuracy", f"{stats['accuracy']:.2%}")
            st.sidebar.metric("Total Predictions with Feedback", stats['total_feedback'])
            st.sidebar.metric("Correct Predictions", stats['correct'])
            
            # Recent predictions
            st.sidebar.title("Recent Predictions")
            recent = db.get_recent_predictions(limit=5)
            
            for pred in recent:
                col1, col2, col3 = st.sidebar.columns(3)
                
                # Display the image if available
                if pred.image_data:
                    img = Image.open(io.BytesIO(pred.image_data))
                    col1.image(img, width=50)
                
                # Display prediction info
                col2.write(f"Predicted: {pred.predicted_digit}")
                
                # Display true label if available
                if pred.true_label is not None:
                    col3.write(f"True: {pred.true_label}")
                else:
                    col3.write("True: N/A")
        except Exception as e:
            st.sidebar.error(f"Error getting statistics: {e}")

if __name__ == "__main__":
    main() 