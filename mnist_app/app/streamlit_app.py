import streamlit as st
import numpy as np
from PIL import Image
import io
import time
import pandas as pd
from datetime import datetime
import sys
import os
import logging
from mnist_app.model.predict import MNISTPredictor
from mnist_app.database.db import Database, check_db_connection

# Set up proper file logging
LOG_DIR = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("mnist_app")

logger.info("StreamlitApp starting")

# Initialize history in session state if not present
if "history" not in st.session_state:
    st.session_state.history = []
    logger.info("Initialized empty history")
    
# Initialize prediction state
if "predicted_digit" not in st.session_state:
    st.session_state.predicted_digit = None
    st.session_state.confidence = None
    logger.info("Initialized prediction state to None")
    
# Clear prediction when canvas changes
def reset_prediction():
    st.session_state.predicted_digit = None
    st.session_state.confidence = None
    logger.info("Prediction state reset due to new drawing")

# Load predictor with caching
@st.cache_resource
def load_predictor():
    logger.info("Loading MNIST predictor model")
    try:
        predictor = MNISTPredictor()
        logger.info("MNIST predictor loaded successfully")
        return predictor
    except Exception as e:
        logger.error(f"Error loading predictor: {e}")
        return None

# Load database with caching
@st.cache_resource
def load_database():
    logger.info("Connecting to database")
    try:
        db = Database()
        logger.info("Database connection successful")
        return db
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def main():
    # Title with styled heading
    st.markdown("<h1 style='text-align: center; color: orange;'>Digit Recognizer</h1>", unsafe_allow_html=True)
    
    # Load predictor and database
    try:
        predictor = load_predictor()
        db = load_database()
        db_connected = check_db_connection() if db else False
        logger.info(f"Database connected: {db_connected}")
    except Exception as e:
        logger.error(f"Error loading predictor or database: {e}")
        st.error(f"Error loading predictor or database: {e}")
        db_connected = False
        return
    
    if not db_connected:
        st.warning("⚠️ Database connection failed. Predictions will not be logged.")
    
    # Layout split: Canvas on the left, prediction on the right
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ① Draw number")
        try:
            from streamlit_drawable_canvas import st_canvas
            
            # Create canvas for drawing
            canvas_result = st_canvas(
                fill_color="white",
                stroke_width=15,
                stroke_color="white",
                background_color="black",  # Black background like MNIST
                height=200,
                width=200,
                drawing_mode="freedraw",
                key="canvas",
            )
            
            # Display a "Predict" button
            predict_clicked = st.button("Predict")
            
            # Check if we have a drawing and the button was clicked
            if canvas_result.image_data is not None:
                # --- REMOVED RAW DRAWING DISPLAY ---
                # st.write("Raw drawing (for debugging):")
                # st.image(canvas_result.image_data, caption="Raw drawing")
                
                # Convert image for prediction
                if predict_clicked:
                    logger.info("Predict button clicked, processing image")
                    
                    # Get just the first channel since we're using white on black like MNIST
                    # Using the improved preprocessing, we pass the whole array
                    # img_array = canvas_result.image_data[:, :, 0].astype(np.uint8) 
                    img_array = canvas_result.image_data # Pass the full RGBA array
                    
                    # Create a PIL Image from the array (preprocessing handles conversion)
                    # img = Image.fromarray(img_array) # Let preprocessing handle this
                    
                    # Save raw image for debugging (keep this logging if helpful)
                    try:
                        debug_img = Image.fromarray(img_array)
                        debug_img.save(f"{LOG_DIR}/raw_canvas_drawing.png")
                        logger.info(f"Saved raw canvas drawing to {LOG_DIR}/raw_canvas_drawing.png")
                    except Exception as e:
                        logger.error(f"Could not save raw canvas debug image: {e}")

                    # --- REMOVED PROCESSED DRAWING DISPLAY ---
                    # st.write("Processed drawing (for prediction):")
                    # st.image(img, caption="Processed for prediction")
                    
                    # Make prediction
                    with st.spinner("Predicting..."):
                        try:
                            logger.info("Calling predict with image")
                            predicted_digit, confidence = predictor.predict(np.array(img_array))
                            logger.info(f"Prediction received: digit={predicted_digit}, confidence={confidence:.4f}")
                            
                            # Store in session state
                            st.session_state.predicted_digit = predicted_digit
                            st.session_state.confidence = confidence
                            
                            # Force a rerun to update the display
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Error during prediction: {e}", exc_info=True)
                            st.error(f"Prediction error: {e}")
            else:
                logger.debug("No drawing detected yet")
                
        except ImportError as e:
            logger.error(f"Import error: {e}")
            st.error("The streamlit-drawable-canvas package is not installed. Please install it with: pip install streamlit-drawable-canvas")
    
    with col2:
        st.markdown("### ② Enter true value")
        true_label = st.text_input("True label:", max_chars=1)
        
        # Display prediction if available
        if st.session_state.predicted_digit is not None:
            predicted_digit = st.session_state.predicted_digit
            confidence = st.session_state.confidence
            
            # Display prediction results
            st.markdown(f"**Prediction:** {predicted_digit}")
            st.markdown(f"**Confidence:** {round(confidence * 100)}%")
            logger.info(f"Displaying prediction: {predicted_digit} with confidence {confidence:.4f}")
            
            # Submit button for logging the feedback
            if st.button("Submit"):
                logger.info(f"Submit button clicked with true_label: {true_label}")
                # Validate true label
                if true_label and true_label.isdigit() and 0 <= int(true_label) <= 9:
                    true_label_int = int(true_label)
                    
                    # Log to database if connected
                    if db_connected and canvas_result.image_data is not None:
                        try:
                            # Convert image to bytes for storage
                            img_bytes = io.BytesIO()
                            img_array = canvas_result.image_data[:, :, 0].astype(np.uint8)
                            img = Image.fromarray(img_array)
                            img.save(img_bytes, format="PNG")
                            
                            # Log to database
                            logger.info(f"Saving prediction to database: {predicted_digit} (true: {true_label_int})")
                            prediction_id = db.log_prediction(
                                predicted_digit=predicted_digit,
                                confidence=confidence,
                                true_label=true_label_int,
                                image_data=img_bytes.getvalue()
                            )
                            
                            # Show feedback
                            if predicted_digit == true_label_int:
                                st.success("Correct prediction! ✅")
                            else:
                                st.warning(f"Incorrect prediction! Model predicted {predicted_digit}, but true digit is {true_label_int}")
                        except Exception as e:
                            logger.error(f"Error logging prediction: {e}", exc_info=True)
                            st.error(f"Error logging prediction: {e}")
                    
                    # Add to session history
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Convert true_label_int to int to avoid type issues when displaying history
                    st.session_state.history.append({
                        "timestamp": timestamp,
                        "pred": int(predicted_digit),
                        "label": int(true_label_int),
                        "confidence": f"{round(confidence * 100)}%"
                    })
                    logger.info(f"Added to session history, now has {len(st.session_state.history)} items")
                    
                    # Force a rerun to update the display
                    st.rerun()
                else:
                    logger.error(f"Invalid true label: {true_label}")
                    st.error("Please enter a valid digit (0-9)")
        else:
            st.info("Draw a digit and click 'Predict' to see the prediction")
    
    # Display history section
    st.markdown("### ③ Display history")
    if st.session_state.history:
        logger.info(f"Displaying session history: {len(st.session_state.history)} items")

        # Try to convert the session history to a dataframe
        try:
            # Preprocess history data for robust DataFrame creation
            processed_history = []
            for entry in st.session_state.history:
                processed_entry = entry.copy()
                # Ensure 'label' is numeric, replace non-digits with pd.NA or similar
                try:
                    processed_entry['label'] = int(entry['label'])
                except (ValueError, TypeError):
                    # Use pandas NA for missing/invalid numeric data
                    processed_entry['label'] = pd.NA 
                processed_history.append(processed_entry)

            # Create DataFrame from processed data
            df = pd.DataFrame(processed_history)
            
            # Define column order and display
            df = df[['timestamp', 'pred', 'label', 'confidence']] # Explicitly order columns
            st.dataframe(df)
            logger.info("Successfully displayed history dataframe")
        except Exception as e:
            # Log the error and try a simpler version
            logger.error(f"Error displaying history dataframe: {e}", exc_info=True)
            st.error("Could not display history in table format. See below:")
            # Display in a simpler format
            for entry in st.session_state.history:
                st.write(f"Time: {entry['timestamp']}, Predicted: {entry['pred']}, Actual: {entry.get('label', 'N/A')}, Confidence: {entry['confidence']}") # Use .get for safety
    else:
        # Only show "No prediction history yet" message
        st.info("No prediction history yet. Make predictions and provide feedback to build history.")
    
    # Display statistics in sidebar
    if db_connected:
        try:
            # Get accuracy statistics
            stats = db.get_accuracy_stats()
            
            # Display statistics
            st.sidebar.title("Statistics")
            st.sidebar.metric("Accuracy", f"{stats['accuracy']:.2%}")
            st.sidebar.metric("Total Predictions with Feedback", stats['total_feedback'])
            st.sidebar.metric("Correct Predictions", stats['correct'])
        except Exception as e:
            logger.error(f"Error getting statistics: {e}", exc_info=True)
            st.sidebar.error(f"Error getting statistics: {e}")

if __name__ == "__main__":
    logger.info("Streamlit app main function starting")
    main()
    logger.info("Streamlit app main function completed") 