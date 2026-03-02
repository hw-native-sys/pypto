FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10, g++-15, git
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y \
      python3.10 python3.10-dev python3.10-venv \
      g++-15 git curl && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Pre-install heavy Python packages
RUN pip install --no-cache-dir \
      torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
      "pytest>=7.0.0" "pytest-forked>=1.0" pytest-xdist \
      "numpy>=2.0" "scikit-build-core>=0.10.0" "nanobind>=2.0.0" \
      "ninja>=1.11.0" "cmake>=3.15"
