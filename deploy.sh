#!/bin/bash
# Simple deployment script for a server

# Update packages
echo "Updating server packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker if not installed
if ! [ -x "$(command -v docker)" ]; then
  echo "Installing Docker..."
  sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
  sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
  sudo apt-get update
  sudo apt-get install -y docker-ce
  sudo usermod -aG docker $USER
  echo "Docker installed successfully."
else
  echo "Docker is already installed."
fi

# Install Docker Compose if not installed
if ! [ -x "$(command -v docker-compose)" ]; then
  echo "Installing Docker Compose..."
  sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose
  echo "Docker Compose installed successfully."
else
  echo "Docker Compose is already installed."
fi

# Start the application
echo "Starting the MNIST Digit Classifier..."
docker-compose up -d

echo "Deployment complete! The application should be accessible at http://$(hostname -I | awk '{print $1}'):8501" 