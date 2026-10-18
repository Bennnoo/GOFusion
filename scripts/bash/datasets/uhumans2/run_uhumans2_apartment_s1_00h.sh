#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Boxin Bu
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

APP_ARGS=()
SEQUENCE_ARG=""
for arg in "$@"; do
  case "$arg" in
    --headless) APP_ARGS+=(--headless) ;;
    -h|--help)
      echo "Usage: bash $0 [/path/to/uHumans2_apartment_s1_00h] [--headless]"
      exit 0 ;;
    --*) echo "Unknown option: $arg" >&2; exit 2 ;;
    *)
      if [[ -n "$SEQUENCE_ARG" ]]; then
        echo "Only one dataset directory may be supplied." >&2
        exit 2
      fi
      SEQUENCE_ARG=$arg ;;
  esac
done

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
BUILD=${BUILD:-$REPO/build}
UHUMANS2_SEQUENCE=${SEQUENCE_ARG:-${UHUMANS2_SEQUENCE:-$REPO/data/uhumans2/apartment_scene/uHumans2_apartment_s1_00h}}
OUT=${OUT:-$REPO/output/uhumans2_apartment_s1_00h_rgbd_$(date +%Y%m%d_%H%M%S)}
OKVIS_CFG=${OKVIS_CFG:-$REPO/config/uhumans2/okvis2.yaml}
SE2_CFG=${SE2_CFG:-$REPO/config/uhumans2/se2.yaml}
PYTHON=${PYTHON:-python3}
IMU_SOURCE=noisy
PREPARE_ONLY=${PREPARE_ONLY:-0}

UHUMANS2_SEQUENCE=$(realpath -e -- "$UHUMANS2_SEQUENCE")
BUILD=$(realpath -m -- "$BUILD")
OUT=$(realpath -m -- "$OUT")
OKVIS_CFG=$(realpath -e -- "$OKVIS_CFG")
SE2_CFG=$(realpath -e -- "$SE2_CFG")
APP=$BUILD/okvis2x_app_synchronous
if [[ "$PREPARE_ONLY" != 1 && (! -x "$APP" || ! -s "$BUILD/small_voc.yml.gz") ]]; then
  echo "Build the project first; expected executable and vocabulary in $BUILD" >&2
  exit 1
fi
if [[ -e "$OUT/run.log" ]]; then
  echo "Output already contains a run: $OUT. Set OUT to a new directory." >&2
  exit 1
fi
mkdir -p "$OUT"
DATASET=$OUT/uhumans2_like/mav0
"$PYTHON" "$REPO/tools/prepare_uhumans2_for_okvis_rgbd.py" \
  --sequence "$UHUMANS2_SEQUENCE" --output-mav0 "$DATASET" \
  --imu-source "$IMU_SOURCE" --force |& tee "$OUT/prepare.log"
if [[ "$PREPARE_ONLY" == 1 ]]; then exit 0; fi

export GLOG_logtostderr=1
export OPENCV_LOG_LEVEL=ERROR
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12}

echo "[uHumans2] App: $APP"
echo "[uHumans2] Config: $OKVIS_CFG ; $SE2_CFG"
echo "[uHumans2] Output: $OUT"
(cd -- "$OUT"; exec "$APP" "${APP_ARGS[@]}" "$OKVIS_CFG" "$SE2_CFG" "$DATASET" "$OUT") |& tee "$OUT/run.log"
