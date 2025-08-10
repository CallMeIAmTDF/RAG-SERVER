FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Cài các thư viện hệ thống cần thiết cho PyTorch và pip build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cập nhật pip lên mới nhất
RUN pip install --upgrade pip

# Cài torch + torchvision CPU
RUN pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Cài các thư viện còn lại
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 2003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "2003"]
