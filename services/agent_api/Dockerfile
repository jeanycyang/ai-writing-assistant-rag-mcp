FROM python:3.12.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/opt/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/opt/huggingface

ARG EMBEDDING_MODEL=BAAI/bge-m3

WORKDIR /app

COPY requirements-agent.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir -r requirements-agent.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}', device='cpu')"

COPY . .

CMD ["uvicorn", "services.agent_api.app.main:app", "--host", "0.0.0.0", "--port", "8002"]
