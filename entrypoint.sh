#!/bin/bash
set -e

# Check if model exists, if not, train it
if [ ! -f /app/mnist_app/model/saved_model/mnist_cnn.pth ]; then
    echo "Model not found. Training model..."
    python train.py
    echo "Model training complete!"
fi

# Run the Streamlit app
exec streamlit run mnist_app/app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 