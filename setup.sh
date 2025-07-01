!/bin/bash
 -*- mode: shell-script -*-

# Install MediaPipe with explicit Python 3.12 wheel
pip install https://files.pythonhosted.org/packages/cp312/m/mediapipe/mediapipe-0.10.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Verify installation
python -c "import mediapipe; print(f'Successfully installed MediaPipe {mediapipe.__version__}')"
