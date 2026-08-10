FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1


RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-distutils \
        git curl build-essential ca-certificates \
        libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 xformers==0.0.30 \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

WORKDIR /code

# Install the project itself so `import dtof_dmdc` and the `dtof-dmdc` CLI resolve from any working directory.
COPY pyproject.toml README.md ./
COPY dtof_dmdc ./dtof_dmdc
RUN pip install -e . --no-deps

CMD ["/bin/bash"]