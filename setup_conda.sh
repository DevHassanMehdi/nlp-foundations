#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="ties4200-nlp"
PY_VERSION="3.10"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda/Anaconda first."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists. Activating and reinstalling deps..."
  conda activate "${ENV_NAME}"
else
  echo "Creating conda env '${ENV_NAME}' with Python ${PY_VERSION}..."
  conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
  conda activate "${ENV_NAME}"
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Done. Activated env: ${ENV_NAME}"
