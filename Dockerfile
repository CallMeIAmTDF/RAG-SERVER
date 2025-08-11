FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libffi-dev libssl-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 2003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "2003"]
