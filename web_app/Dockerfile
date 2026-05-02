FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV/Pillow/TensorFlow
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
