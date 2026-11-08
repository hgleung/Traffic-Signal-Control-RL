# Reproducible environment for the traffic signal RL project.
#   docker build -t traffic-rl .
#   docker run --rm traffic-rl                     # runs the smoke test
#   docker run --rm -it traffic-rl bash            # interactive shell
FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

# CityFlow is not published on PyPI, so build it from a pinned commit.
ARG CITYFLOW_COMMIT=81ee0f47659ca66177a71f81676691c58ee89184
RUN pip install --no-cache-dir "git+https://github.com/cityflow-project/CityFlow.git@${CITYFLOW_COMMIT}"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["pytest", "-q", "tests"]
