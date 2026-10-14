# Churn API image (whizdom-churn-serve): the FastAPI app with the project code.
FROM python:3.13-slim

# LightGBM needs the OpenMP library, which the slim image does not include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*


# uv installs the exact versions of uv.lock
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, so this layer is cached when only the code changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Project code needed by the API and the batch scoring
COPY src/ src/
COPY eda/ eda/
COPY serving/ serving/
COPY configs/ configs/
COPY catalog/ catalog/

EXPOSE 8000
CMD [".venv/bin/python", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
