FROM python:3.9-slim

# ติดตั้ง lib ที่จำเป็น
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# ให้ shell script มีสิทธิ์รัน
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
