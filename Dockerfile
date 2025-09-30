# Use official Python image as base
FROM python:3.11-slim

# Install git for repository cloning
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements if exists, else skip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy all files to container
COPY . .

# Set entrypoint to run main.py
CMD ["python", "main.py"]
