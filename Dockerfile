FROM python:3.10-slim
WORKDIR /code
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY app/ app/
COPY src/__init__.py src/__init__.py
COPY src/text_cleaning.py src/text_cleaning.py
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]