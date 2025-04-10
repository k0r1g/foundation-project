#!/bin/bash
set -e

echo "Starting MNIST application..."

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
until PGPASSWORD=postgres psql -h postgres -U postgres -d mnist -c '\q' > /dev/null 2>&1; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done
echo "PostgreSQL is up - executing command"

# Initialize database
echo "Initializing database..."
python -c "from mnist_app.database.db import init_db; init_db()"
echo "Database initialized"

# Check if model exists
if [ ! -f /app/mnist_app/model/saved_model/mnist_cnn.pth ]; then
  echo "Model file not found. Training a basic model with just 2 epochs..."
  # Use quick training for fast startup
  python -c "from mnist_app.model.train_model import train_model; train_model(epochs=2)"
  echo "Basic model training complete."
else
  echo "Model already exists, skipping training."
fi

# Run the Streamlit app
echo "Starting Streamlit application at http://0.0.0.0:8501"
exec streamlit run mnist_app/app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 