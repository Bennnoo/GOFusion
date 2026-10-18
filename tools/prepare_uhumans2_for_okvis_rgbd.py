#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Boxin Bu
# SPDX-License-Identifier: BSD-3-Clause
"""Prepare an extracted uHumans2 sequence as an OKVIS2-X stereo RGB-D mav0 folder."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="Extracted uHumans2 sequence directory.")
    parser.add_argument("--output-mav0", required=True, help="Output OKVIS mav0 directory.")
    parser.add_argument("--force", action="store_true", help="Replace generated CSVs and symlinks.")
    parser.add_argument(
        "--imu-source",
        choices=("noisy", "clean"),
        default="noisy",
        help="IMU CSV to export. Default uses the realistic noisy IMU.",
    )
    parser.add_argument(
        "--no-crop-to-imu",
        action="store_true",
        help="Keep camera/depth rows outside the IMU time range.",
    )
    return parser.parse_args()


def require_dir(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def ensure_symlink(dst: Path, target: Path, force: bool) -> None:
    if dst.is_symlink():
        if dst.resolve() == target.resolve():
            return
        if not force:
            raise FileExistsError(f"Symlink {dst} points to {dst.resolve()}, expected {target}")
        dst.unlink()
    elif dst.exists():
        raise FileExistsError(f"Path exists and is not a symlink: {dst}")
    dst.symlink_to(target)


def write_text_file(path: Path, text: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Generated file already exists: {path}. Use --force to replace.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def first_last_timestamps(path: Path) -> tuple[int, int]:
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return int(rows[0]["timestamp_ns"]), int(rows[-1]["timestamp_ns"])


def build_synced_indices(sequence: Path, imu_range: tuple[int, int] | None) -> tuple[list[dict[str, str]], dict[str, int]]:
    stereo_rows = read_rows(require_file(sequence / "stereo" / "mono_pairs.csv", "stereo mono pairs"))
    rgbd_rows = read_rows(require_file(sequence / "rgbd" / "left_rgb_depth_pairs.csv", "left RGB-depth pairs"))
    rgbd_by_stamp = {int(row["timestamp_ns"]): row for row in rgbd_rows}

    dropped_missing_rgbd = 0
    dropped_before_imu = 0
    dropped_after_imu = 0
    synced: list[dict[str, str]] = []
    for stereo in stereo_rows:
        stamp = int(stereo["timestamp_ns"])
        rgbd = rgbd_by_stamp.get(stamp)
        if rgbd is None:
            dropped_missing_rgbd += 1
            continue
        if imu_range is not None:
            if stamp < imu_range[0]:
                dropped_before_imu += 1
                continue
            if stamp > imu_range[1]:
                dropped_after_imu += 1
                continue
        synced.append(
            {
                "timestamp_ns": str(stamp),
                "left_mono_file": Path(stereo["left_mono_file"]).name,
                "right_mono_file": Path(stereo["right_mono_file"]).name,
                "left_rgb_file": Path(rgbd["left_rgb_file"]).name,
                "depth_mm_file": Path(rgbd["depth_mm_file"]).name,
            }
        )

    if not synced:
        raise ValueError("No synchronized stereo/RGB-D rows remain")

    report = {
        "stereo_rows": len(stereo_rows),
        "rgbd_rows": len(rgbd_rows),
        "synced_rows": len(synced),
        "dropped_missing_rgbd": dropped_missing_rgbd,
        "dropped_before_imu": dropped_before_imu,
        "dropped_after_imu": dropped_after_imu,
    }
    return synced, report


def image_csv(rows: list[dict[str, str]], field: str) -> str:
    lines = ["#timestamp [ns],filename"]
    lines.extend(f"{row['timestamp_ns']},{row[field]}" for row in rows)
    return "\n".join(lines) + "\n"


def convert_imu_csv(path: Path) -> tuple[str, int]:
    lines = [
        "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
        "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
    ]
    rows = read_rows(path)
    for row in rows:
        lines.append(
            f"{int(row['timestamp_ns'])},"
            f"{row['angular_velocity_x']},{row['angular_velocity_y']},{row['angular_velocity_z']},"
            f"{row['linear_acceleration_x']},{row['linear_acceleration_y']},{row['linear_acceleration_z']}"
        )
    return "\n".join(lines) + "\n", len(rows)


def write_marker(output_mav0: Path, sequence: Path, report: dict[str, object]) -> None:
    marker = {
        "format": "okvis2x_uhumans2_stereo_rgbd_mav0",
        "prepared_at_unix": time.time(),
        "sequence": str(sequence),
        "report": report,
        "streams": {
            "cam0": "left_mono",
            "cam1": "right_mono",
            "rgb0": "left_rgb",
            "depth0": "left-aligned depth_mm",
            "imu0": "imu",
        },
        "depth_convention": "uint16 millimeters; depth_meters = png_value * 0.001; zero invalid",
    }
    (output_mav0 / ".uhumans2_rgbd_prepare.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    sequence = Path(args.sequence).expanduser().resolve()
    output_mav0 = Path(args.output_mav0).expanduser().resolve()

    require_dir(sequence, "uHumans2 sequence")
    left_mono_dir = require_dir(sequence / "cameras" / "left_mono" / "images", "left mono images")
    right_mono_dir = require_dir(sequence / "cameras" / "right_mono" / "images", "right mono images")
    left_rgb_dir = require_dir(sequence / "cameras" / "left_rgb" / "images", "left RGB images")
    depth_dir = require_dir(sequence / "cameras" / "depth" / "depth_mm", "depth_mm images")
    imu_csv = require_file(sequence / "imu" / f"{args.imu_source}.csv", f"{args.imu_source} IMU CSV")

    imu_range = None if args.no_crop_to_imu else first_last_timestamps(imu_csv)
    synced_rows, sync_report = build_synced_indices(sequence, imu_range)
    imu_text, imu_count = convert_imu_csv(imu_csv)

    output_mav0.mkdir(parents=True, exist_ok=True)
    for stream in ("cam0", "cam1", "rgb0", "depth0", "imu0"):
        (output_mav0 / stream).mkdir(parents=True, exist_ok=True)

    write_text_file(output_mav0 / "cam0" / "data.csv", image_csv(synced_rows, "left_mono_file"), args.force)
    write_text_file(output_mav0 / "cam1" / "data.csv", image_csv(synced_rows, "right_mono_file"), args.force)
    write_text_file(output_mav0 / "rgb0" / "data.csv", image_csv(synced_rows, "left_rgb_file"), args.force)
    write_text_file(output_mav0 / "depth0" / "data.csv", image_csv(synced_rows, "depth_mm_file"), args.force)
    write_text_file(output_mav0 / "imu0" / "data.csv", imu_text, args.force)

    ensure_symlink(output_mav0 / "cam0" / "data", left_mono_dir, args.force)
    ensure_symlink(output_mav0 / "cam1" / "data", right_mono_dir, args.force)
    ensure_symlink(output_mav0 / "rgb0" / "data", left_rgb_dir, args.force)
    ensure_symlink(output_mav0 / "depth0" / "data", depth_dir, args.force)

    report = {
        "sequence": str(sequence),
        "imu_source": args.imu_source,
        "crop_to_imu": not args.no_crop_to_imu,
        "imu_rows": imu_count,
        **sync_report,
    }
    write_marker(output_mav0, sequence, report)

    print(f"[prepare_uhumans2_for_okvis_rgbd] sequence={sequence}")
    print(f"[prepare_uhumans2_for_okvis_rgbd] output_mav0={output_mav0}")
    print(
        "[prepare_uhumans2_for_okvis_rgbd] sync: "
        f"stereo={sync_report['stereo_rows']} rgbd={sync_report['rgbd_rows']} "
        f"synced={sync_report['synced_rows']} "
        f"dropped_missing_rgbd={sync_report['dropped_missing_rgbd']} "
        f"dropped_before_imu={sync_report['dropped_before_imu']} "
        f"dropped_after_imu={sync_report['dropped_after_imu']}"
    )
    print(f"[prepare_uhumans2_for_okvis_rgbd] imu_source={args.imu_source} imu_rows={imu_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
