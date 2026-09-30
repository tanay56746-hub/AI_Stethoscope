FROM python:3.11-slim

# Install curl for download fallbacks and libsndfile for librosa/soundfile
RUN apt-get update && apt-get install -y curl libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install them
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Download weights during the docker build phase so they are baked into the image
RUN python backend/download_weights.py

# HuggingFace Spaces passes the PORT env variable (usually 7860)
ENV PORT=7860
EXPOSE 7860

# Start the server
CMD ["python", "manage.py", "runserver"]
