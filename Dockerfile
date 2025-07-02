# Use Python 3.10 image (mediapipe is not supported in 3.13)
FROM python:3.12-slim

# Set working directory
WORKDIR /app2

# Install system dependencies required for mediapipe + OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 libgtk2.0-dev \
    && apt-get clean

# Copy files to container
COPY . /app2

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Run your Streamlit app
CMD ["streamlit", "run", "app2.py", "--server.port=8501", "--server.address=0.0.0.0"]
