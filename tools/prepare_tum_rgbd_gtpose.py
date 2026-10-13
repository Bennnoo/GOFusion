#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Anonymous GOFusion author
# SPDX-License-Identifier: BSD-3-Clause
"""Associate TUM RGB-D frames and interpolate camera GT poses at depth timestamps."""

import argparse
import bisect
import csv
import math
from pathlib import Path


def read_rows(path):
    with path.open() as stream:
        return sorted(line.split() for line in stream
                      if line.strip() and not line.lstrip().startswith("#"))


def normalize(q):
    norm = math.sqrt(sum(x * x for x in q))
    return [x / norm for x in q]


def slerp(a, b, t):
    a, b = normalize(a), normalize(b)
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b, dot = [-x for x in b], -dot
    if dot > 0.9995:
        return normalize([(1 - t) * x + t * y for x, y in zip(a, b)])
    angle = math.acos(min(1.0, dot))
    return normalize([(math.sin((1 - t) * angle) * x + math.sin(t * angle) * y)
                      / math.sin(angle) for x, y in zip(a, b)])


def prepare(sequence, output):
    rgb = {float(row[0]): row[1] for row in read_rows(sequence / "rgb.txt")}
    depth = {float(row[0]): row[1] for row in read_rows(sequence / "depth.txt")}
    gt = {float(row[0]): [float(x) for x in row[1:]]
          for row in read_rows(sequence / "groundtruth.txt")}
    depth_times, gt_times = sorted(depth), sorted(gt)

    candidates = []
    for rgb_time in rgb:
        begin = bisect.bisect_left(depth_times, rgb_time - 0.02)
        end = bisect.bisect_right(depth_times, rgb_time + 0.02)
        for depth_time in depth_times[begin:end]:
            difference = abs(rgb_time - depth_time)
            if difference < 0.02:
                candidates.append((difference, rgb_time, depth_time))
    used_rgb, used_depth, pairs = set(), set(), []
    for _, rgb_time, depth_time in sorted(candidates):
        if rgb_time not in used_rgb and depth_time not in used_depth:
            used_rgb.add(rgb_time)
            used_depth.add(depth_time)
            pairs.append((rgb_time, depth_time))

    frames = []
    for rgb_time, depth_time in sorted(pairs):
        index = bisect.bisect_left(gt_times, depth_time)
        if depth_time in gt:
            pose = gt[depth_time][:3] + normalize(gt[depth_time][3:])
        elif index == 0 or index == len(gt_times):
            continue
        else:
            before, after = gt_times[index - 1:index + 1]
            if after - before > 0.2:
                continue
            t = (depth_time - before) / (after - before)
            a, b = gt[before], gt[after]
            pose = [(1 - t) * x + t * y for x, y in zip(a[:3], b[:3])]
            pose += slerp(a[3:], b[3:], t)
        frames.append((round(depth_time * 1e9), rgb[rgb_time], depth[depth_time], pose))
    if not frames:
        raise ValueError(f"No RGB-D frames with valid GT poses in {sequence}")

    def link(path, target):
        if path.is_symlink():
            path.unlink()
        path.symlink_to(target)

    for stream in ("cam0", "cam1", "rgb0", "depth0"):
        (output / stream).mkdir(parents=True, exist_ok=True)
    for stream in ("rgb0", "depth0"):
        (output / stream / "data").mkdir(exist_ok=True)
    for stream in ("cam0", "cam1"):
        link(output / stream / "data", Path("../rgb0/data"))

    rgb_rows, depth_rows, pose_rows = [], [], []
    for i, (timestamp, rgb_file, depth_file, pose) in enumerate(frames):
        rgb_name, depth_name = f"frame{i:06d}.png", f"depth{i:06d}.png"
        for stream, name, source in (("rgb0", rgb_name, rgb_file),
                                     ("depth0", depth_name, depth_file)):
            target = sequence / source
            if not target.is_file():
                raise FileNotFoundError(target)
            link(output / stream / "data" / name, target)
        rgb_rows.append((timestamp, rgb_name))
        depth_rows.append((timestamp, depth_name))
        pose_rows.append((timestamp, *pose))
    for stream, rows in (("cam0", rgb_rows), ("cam1", rgb_rows),
                         ("rgb0", rgb_rows), ("depth0", depth_rows)):
        with (output / stream / "data.csv").open("w") as handle:
            handle.write("#timestamp [ns],filename\n")
            csv.writer(handle, lineterminator="\n").writerows(rows)
    with (output / "gt_trajectory.csv").open("w") as handle:
        handle.write("timestamp,px,py,pz,qx,qy,qz,qw\n")
        csv.writer(handle, lineterminator="\n").writerows(pose_rows)
    print(f"Prepared {len(frames)} RGB-D frames with GT poses: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.sequence.resolve(), args.output.resolve())
