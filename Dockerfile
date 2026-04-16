# Dockerfile for the PhysML / myco REST API microservice (Stage 18).
#
# Build:
#   docker build -t physml-server .
#
# Run:
#   docker run -p 8000:8000 physml-server

FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install server extras
RUN pip install --no-cache-dir fastapi uvicorn[standard]

# Copy source
COPY physml/ physml/
COPY pyproject.toml* setup.py* setup.cfg* ./

# Install the physml package in editable mode if pyproject.toml exists,
# otherwise just ensure the package is on the path
RUN if [ -f pyproject.toml ] || [ -f setup.py ]; then \
        pip install --no-cache-dir -e .; \
    fi

EXPOSE 8000

CMD ["uvicorn", "physml.server:app", "--host", "0.0.0.0", "--port", "8000"]
