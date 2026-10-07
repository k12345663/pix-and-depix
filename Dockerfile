FROM python:3.10-slim

# Create user to run the app (Hugging Face requirement)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Install system dependencies if required by scipy/numpy or image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY webapp/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . /app/

# Give user ownership of the app directory so it can write to 'processed' and 'uploads' folders
RUN chown -R user:user /app

# Switch to the non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Run from the webapp directory
WORKDIR /app/webapp

# Hugging Face exposes port 7860 by default
EXPOSE 7860

# Run with Gunicorn, binding to 0.0.0.0:7860 and setting a 120s timeout
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "2", "app:app"]
