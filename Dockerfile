FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV DOCKER_ENV=true
ENV PYTHONUNBUFFERED=1

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy MLProject files
COPY MLProject/ ./MLProject/

# Copy scripts
COPY scripts/ ./scripts/

# Create model output directory
RUN mkdir -p model_output

# Set working directory to MLProject
WORKDIR /app/MLProject

# Default command
CMD ["python", "modelling.py"]