FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/ app/

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Pre-download lightweight model (full DIPPER is too large for Docker image, download at runtime)
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('google/t5-efficient-large-nl32'); \
    AutoModelForSeq2SeqLM.from_pretrained('SamSJackson/paraphrase-dipper-no-ctx')"

EXPOSE 8000

# Default to lightweight model; set HUMANIZER_MODEL=full for DIPPER XXL
ENV HUMANIZER_MODEL=lightweight

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
