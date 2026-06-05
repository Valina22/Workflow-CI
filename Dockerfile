FROM python:3.10-slim

LABEL maintainer="Valina Puspita Sari"
LABEL description="Heart Disease Prediction - ML Pipeline"
LABEL version="1.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY MLProject/requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p artifacts models logs mlruns

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RUNNING_IN_DOCKER=1

WORKDIR /app/MLProject

ENTRYPOINT ["python", "modelling.py"]