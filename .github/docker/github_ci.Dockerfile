FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System: Python 3.10, gcc/g++-15, build essentials
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3.10-venv \
      gcc-15 g++-15 build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set gcc/g++-15 as default compilers
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-15 1 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-15 1 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-15 1 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-15 1

# Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Build tools (needed by pip install -e .)
RUN pip install --no-cache-dir \
      "scikit-build-core>=0.10.0" "nanobind>=2.0.0" \
      "ninja>=1.11.0" "cmake>=3.15"

# Project runtime + dev dependencies
RUN pip install --no-cache-dir \
      "numpy>=2.0" \
      "pytest>=7.0.0" "pytest-forked>=1.0" "pytest-xdist==3.8.0" \
      "pyright==1.1.407" "ruff==0.14.8" "clang-tidy==21.1.0" \
      "pre-commit==4.5.1"

# torch CPU (heaviest package, separate layer for caching)
RUN pip install --no-cache-dir \
      "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu

# Fix pip cache permissions for GitHub Actions containers
# Set compilers explicitly (pip build isolation looks for x86_64-linux-gnu-gcc)
ENV PIP_CACHE_DIR=/tmp/pip-cache \
    CC=gcc-15 \
    CXX=g++-15
