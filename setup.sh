#!/bin/bash
set -e

# Install system dependencies
apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    libopencv-core-dev \
    libopencv-highgui-dev \
    libopencv-imgproc-dev

# Force Python 3.12 environment
python3.12 -m pip install --upgrade pip
python3.12 -m pip install https://files.pythonhosted.org/packages/cp312/m/mediapipe/mediapipe-0.10.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
