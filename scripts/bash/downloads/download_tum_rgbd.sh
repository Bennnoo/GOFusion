#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Boxin Bu
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail

DATA=data/tum_rgbd
SEQUENCES=(
  rgbd_dataset_freiburg1_desk
  rgbd_dataset_freiburg2_xyz
  rgbd_dataset_freiburg3_long_office_household
)
# No arguments: download all three. Pass sequence names to select a subset.
if (( $# > 0 )); then SEQUENCES=("$@"); fi

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd -- "$REPO"
mkdir -p "$DATA"
for SEQUENCE in "${SEQUENCES[@]}"; do
  case "$SEQUENCE" in
    rgbd_dataset_freiburg[123]_*) ;;
    *) echo "Invalid TUM sequence: $SEQUENCE" >&2; exit 2 ;;
  esac
  DEST=$DATA/$SEQUENCE
  if [[ -s "$DEST/mav0/gt_trajectory.csv" ]]; then
    echo "[TUM] Already prepared, skipping: $SEQUENCE"
    continue
  fi

  if [[ -s "$DEST/rgb.txt" && -s "$DEST/depth.txt" && -s "$DEST/groundtruth.txt" \
        && -d "$DEST/rgb" && -d "$DEST/depth" ]]; then
    echo "[TUM] Using existing data: $SEQUENCE"
  else
    ARCHIVE=$DATA/$SEQUENCE.tgz
    if [[ ! -s "$ARCHIVE" ]]; then
      FAMILY=${SEQUENCE#rgbd_dataset_}
      FAMILY=${FAMILY%%_*}
      curl -fL -C - "https://vision.in.tum.de/rgbd/dataset/$FAMILY/$SEQUENCE.tgz" \
        -o "$ARCHIVE.part"
      mv -- "$ARCHIVE.part" "$ARCHIVE"
    fi
    tar -xzf "$ARCHIVE" -C "$DATA"
  fi
  "${PYTHON:-python3}" "$REPO/tools/prepare_tum_rgbd_gtpose.py" "$DEST" "$DEST/mav0"
done
