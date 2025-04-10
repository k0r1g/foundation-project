# MNIST Digit Classifier

A complete end-to-end application that allows users to draw a digit and have it classified using a PyTorch model trained on the MNIST dataset.

# Link

Link to live app: http://95.217.238.188:8501/

## Features

- **PyTorch Model**: A convolutional neural network trained on the MNIST dataset.
- **Interactive Web Interface**: Draw digits on a canvas and get real-time predictions.
- **Feedback System**: Let the application know if the prediction was correct.
- **Database Logging**: All predictions are stored in a PostgreSQL database.
- **Containerized**: The entire application is containerized using Docker.

## Architecture

The application consists of three main components:

1. **PyTorch Model**: A CNN trained on the MNIST dataset.
2. **Streamlit Web App**: An interactive web interface for drawing digits and receiving predictions.
3. **PostgreSQL Database**: Stores all predictions, including the predicted digit, confidence, and user feedback.

## Prerequisites

- Docker and Docker Compose

## Getting Started

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/mnist-digit-classifier.git
   cd mnist-digit-classifier
   ```

2. Start the application with Docker Compose:
   ```
   docker-compose up -d
   ```

3. Access the application in your browser:
   ```
   http://localhost:8501
   ```

## Usage

1. Draw a digit on the canvas provided.
2. Click the "Predict" button to have the model classify your drawing.
3. View the prediction and confidence score.
4. (Optional) Provide feedback on whether the prediction was correct.

## Deployment

To deploy this application to a production server:

1. Set up a server (e.g., Hetzner, AWS, DigitalOcean).
2. Install Docker and Docker Compose on the server.
3. Clone the repository to the server.
4. Start the application with Docker Compose.
5. Configure your server's firewall to allow traffic on port 8501.

## Project Structure

```
mnist-digit-classifier/
├── mnist_app/
│   ├── model/
│   │   ├── train_model.py  # Model definition and training
│   │   ├── predict.py      # Prediction functionality
│   │   └── saved_model/    # Saved model weights
│   ├── app/
│   │   └── streamlit_app.py # Web interface
│   └── database/
│       └── db.py           # Database operations
├── app.py                  # Entry point
├── train.py                # Model training script
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── entrypoint.sh           # Container entrypoint script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Development

If you want to run the application without Docker:

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   pip install streamlit-drawable-canvas
   ```

2. Train the model:
   ```
   python train.py
   ```

3. Run the Streamlit app:
   ```
   python app.py
   ```

## License

[MIT License](LICENSE)

## Acknowledgements

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)
- [PostgreSQL](https://www.postgresql.org/) 