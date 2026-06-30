# Use an official lightweight Python runtime
FROM python:3.10-slim

# Set environment variables to prevent Python from writing pyc files and buffering logs
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    ULTRALYTICS_CONFIG_DIR=/cache/ultralytics

# Set the working directory inside the container
WORKDIR /app

# Install basic system dependencies required for compiling code and running CV libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create cache directory with full read/write permissions for downloading local model weights
RUN mkdir -p /cache && chmod -R 777 /cache

# Copy requirements first to speed up future builds via caching layers
COPY requirements.txt .

# Upgrade pip and install all listed Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the remaining application code into the container
COPY . .

# Inform Docker that the container listens on port 7860 at runtime
EXPOSE 7860

# Command to execute your Flask API
CMD ["python", "-u", "app.py"]
