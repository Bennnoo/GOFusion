#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Anonymous GOFusion author
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# bash scripts/bash/datasets/tum_rgbd/run_tum_rgbd_gtpose.sh rgbd_dataset_freiburg1_desk --headless
# bash scripts/bash/datasets/tum_rgbd/run_tum_rgbd_gtpose.sh rgbd_dataset_freiburg2_xyz --headless
# bash scripts/bash/datasets/tum_rgbd/run_tum_rgbd_gtpose.sh rgbd_dataset_freiburg3_long_office_household --headless
SEQUENCE=${SEQUENCE:-rgbd_dataset_freiburg1_desk}
APP_ARGS=(--gt-poses)
DATASET_ARG=""
for arg in "$@"; do
  case "$arg" in
    --headless) APP_ARGS+=(--headless) ;;
    -h|--help)
      echo "Usage: bash $0 [sequence-name | /path/to/TUM-sequence] [--headless]"
      exit 0 ;;
    --*) echo "Unknown option: $arg" >&2; exit 2 ;;
    *)
      if [[ -n "$DATASET_ARG" ]]; then
        echo "Only one sequence may be supplied." >&2; exit 2
      fi
      DATASET_ARG=$arg ;;
  esac
done

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
DATASET=${DATASET_ARG:-$SEQUENCE}
if [[ "$DATASET" != */* && ! -d "$DATASET" ]]; then DATASET=$REPO/data/tum_rgbd/$DATASET; fi
DATASET=$(realpath -e -- "$DATASET")
BUILD=$(realpath -m -- "${BUILD:-$REPO/build}")
OUT=$(realpath -m -- "${OUT:-$REPO/output/$(basename -- "$DATASET")_gtpose_$(date +%Y%m%d_%H%M%S)}")
CONFIG=$REPO/config/tum_rgbd
APP=$BUILD/okvis2x_app_synchronous
if [[ ! -x "$APP" || ! -s "$BUILD/small_voc.yml.gz" ]]; then
  echo "Build the project first; expected executable and vocabulary in $BUILD" >&2; exit 1
fi
if [[ -e "$OUT/run.log" ]]; then
  echo "Output already contains a run: $OUT. Set OUT to a new directory." >&2; exit 1
fi
mkdir -p "$OUT"
PREPARED=$DATASET/mav0
if [[ ! -s "$PREPARED/gt_trajectory.csv" ]]; then
  PREPARED=$OUT/tum_rgbd_like/mav0
  "${PYTHON:-python3}" "$REPO/tools/prepare_tum_rgbd_gtpose.py" \
    "$DATASET" "$PREPARED" |& tee "$OUT/prepare.log"
fi

export GLOG_logtostderr=1
export OPENCV_LOG_LEVEL=ERROR
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12}
echo "[TUM GT pose] Output: $OUT"
(cd -- "$OUT"; exec "$APP" "${APP_ARGS[@]}" "$CONFIG/okvis2.yaml" \
  "$CONFIG/se2.yaml" "$PREPARED" "$OUT") |& tee "$OUT/run.log"
