FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "src/train_cnn.py"]
CMD ["--config", "train_config.example.json"]
