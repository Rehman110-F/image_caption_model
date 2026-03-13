FROM python:3.10-slim

WORKDIR /app

COPY requirements-cpu.txt .
RUN pip install --no-cache-dir -r requirements-cpu.txt

COPY . .

RUN mkdir -p data/images/val2017 \
             data/annotations \
             saved_models \
             logs

EXPOSE 8000

CMD ["python", "scripts/run_api.py"]
