FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY inference_service/requirements.txt /app/inference_service/requirements.txt
RUN pip install --no-cache-dir -r inference_service/requirements.txt && useradd --uid 10001 --create-home inference
COPY inference_service/*.py /app/inference_service/
COPY inference_service/artifacts/ /app/inference_service/artifacts/
COPY ml/prepare_dataset.py ml/prepare_event_driven.py /app/ml/
COPY contracts/openapi.yaml /app/contracts/openapi.yaml
USER inference
CMD ["python", "-m", "uvicorn", "inference_service.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
