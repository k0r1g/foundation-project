import os
import streamlit.web.cli as stcli
import sys

# Entry point for the Streamlit application
if __name__ == "__main__":
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set path to the Streamlit app
    streamlit_app_path = os.path.join(current_dir, "mnist_app", "app", "streamlit_app.py")
    
    # Run the Streamlit app
    sys.argv = ["streamlit", "run", streamlit_app_path]
    sys.exit(stcli.main()) 