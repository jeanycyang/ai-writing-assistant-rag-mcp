FROM python:3.12.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements-agent.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir -r requirements-agent.txt

COPY . .

CMD ["uvicorn", "services.agent_api.app.main:app", "--host", "0.0.0.0", "--port", "8002"]
