FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for canvas
RUN pip install --no-cache-dir streamlit-drawable-canvas

# Copy application code
COPY . .

# Create directory for model
RUN mkdir -p /app/mnist_app/model/saved_model

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV DB_HOST=postgres
ENV DB_PORT=5432
ENV DB_NAME=mnist
ENV DB_USER=postgres
ENV DB_PASSWORD=postgres

# Set up entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"] 