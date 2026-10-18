#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Boxin Bu
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Download links: https://web.mit.edu/sparklab/datasets/uHumans2/
# To select another sequence, replace NAME (bag filename without .bag) and URL.
# Or download the bag manually from that page and place it in DATA below.
DATA=data/uhumans2
WORKERS=${WORKERS:-8}
NAME=uHumans2_apartment_s1_00h
URL='https://drive.google.com/uc?id=1kU_drpyG7glQ8pJyeztpiy214WbBEtbM'
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd -- "$REPO"
SCENE=${NAME#uHumans2_}
SCENE=${SCENE%%_*}
SEQUENCE=$DATA/${SCENE}_scene/$NAME
BAG=$DATA/$NAME.bag

if [[ -s "$SEQUENCE/mav0/.uhumans2_rgbd_prepare.json" ]]; then
  echo "[uHumans2] Already prepared, skipping: $SEQUENCE"
  exit 0
fi

if [[ ! -s "$SEQUENCE/metadata/export_complete.json" ]]; then
  TOOLS=$DATA/.venv
  if [[ ! -x "$TOOLS/bin/python" ]]; then
    "${PYTHON:-python3}" -m venv "$TOOLS"
  fi
  if ! "$TOOLS/bin/python" -c 'import gdown, rosbags.highlevel, cv2, numpy, tqdm' 2>/dev/null; then
    PIP_CONFIG_FILE=/dev/null "$TOOLS/bin/python" -m pip --isolated install \
      --index-url https://pypi.org/simple --no-cache-dir \
      'gdown>=5.2,<6' 'rosbags>=0.10,<0.11' 'opencv-python-headless>=4.8,<5'
  fi
  if [[ ! -s "$BAG" ]]; then
    if ! "$TOOLS/bin/python" -m gdown --no-cookies --continue "$URL" -O "$BAG.part"; then
      echo "Download $NAME.bag manually from https://web.mit.edu/sparklab/datasets/uHumans2/" >&2
      echo "Save it as $BAG, then rerun this script." >&2
      exit 1
    fi
    mv -- "$BAG.part" "$BAG"
  fi
  "$TOOLS/bin/python" "$REPO/tools/extract_uhumans2_rgbd.py" "$BAG" "$SEQUENCE" --workers "$WORKERS"
fi

"${PYTHON:-python3}" "$REPO/tools/prepare_uhumans2_for_okvis_rgbd.py" \
  --sequence "$SEQUENCE" --output-mav0 "$SEQUENCE/mav0" --imu-source noisy --force
echo "Run: bash scripts/bash/datasets/uhumans2/run_uhumans2_apartment_s1_00h.sh $SEQUENCE --headless"
