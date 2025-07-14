# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache usage
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make port 8501 available for Streamlit
EXPOSE 8501

# Set the command to run the app
CMD ["streamlit", "run", "src/app.py", "--server.address", "0.0.0.0"]
