FROM python:3.12

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY requirements.txt .
COPY app.py .
COPY model/machine_translation.keras model/
COPY model/tokenizer_en.pkl model/
COPY model/tokenizer_fr.pkl model/
COPY templates/ templates/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "app.py"]