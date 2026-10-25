FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY generator_service/requirements.txt /app/generator_service/requirements.txt
RUN pip install --no-cache-dir -r generator_service/requirements.txt \
    && useradd --uid 10001 --create-home generator \
    && mkdir -p /app/runtime && chown generator:generator /app/runtime
COPY generator_service/*.py /app/generator_service/
COPY examples/risk/event-driven/*.py /app/examples/risk/event-driven/
COPY examples/risk/vps-business/catalog.json /app/examples/risk/vps-business/catalog.json
COPY contracts/openapi.yaml /app/contracts/openapi.yaml
USER generator
CMD ["python", "-m", "uvicorn", "generator_service.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
