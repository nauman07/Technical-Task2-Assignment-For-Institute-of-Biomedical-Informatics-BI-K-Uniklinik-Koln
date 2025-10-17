FROM python:3.11-slim

WORKDIR /app

# optional: certificates help in some corp networks
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY rag ./rag
COPY ui ./ui
COPY data ./data

ENV API_URL=http://localhost:8000
ENV CHUNK_SIZE=800
ENV CHUNK_OVERLAP=120
ENV TOP_K=5
ENV MAX_NEW_TOKENS=60

EXPOSE 8000 8051
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
