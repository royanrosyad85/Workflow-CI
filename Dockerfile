FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow
RUN pip install mlflow[extras]==2.22.0

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy MLproject files
COPY MLProject/ ./

# Expose MLflow port
EXPOSE 5000

# Default command to run MLflow project
CMD ["mlflow", "run", ".", "--no-conda"]