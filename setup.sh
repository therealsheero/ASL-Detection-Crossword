#!/bin/bash
# Install MediaPipe from source
git clone https://github.com/google/mediapipe.git
cd mediapipe
python setup.py install
cd ..
rm -rf mediapipe  # Clean up after installation
