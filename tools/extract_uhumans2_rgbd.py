#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Anonymous GOFusion author
# SPDX-License-Identifier: BSD-3-Clause
"""Extract uHumans2 stereo, RGB-D, noisy IMU and ground-truth odometry from a ROS1 bag."""

import argparse
import csv
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm


IMAGES = {
    "/tesse/left_cam/mono/image_raw": ("left_mono", "cameras/left_mono/images"),
    "/tesse/right_cam/mono/image_raw": ("right_mono", "cameras/right_mono/images"),
    "/tesse/left_cam/rgb/image_raw": ("left_rgb", "cameras/left_rgb/images"),
    "/tesse/depth_cam/mono/image_raw": ("depth", "cameras/depth/depth_mm"),
}
IMU = "/tesse/imu/noisy/imu"
ODOM = "/tesse/odom"


def decode_image(msg):
    height, width, step = msg.height, msg.width, msg.step
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
    if msg.encoding in ("rgb8", "bgr8"):
        image = data[:, :width * 3].reshape(height, width, 3)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else image
    if msg.encoding in ("mono8", "8UC1"):
        return data[:, :width]
    endian = ">" if msg.is_bigendian else "<"
    if msg.encoding in ("mono16", "16UC1"):
        return data.view(endian + "u2")[:, :width].astype(np.uint16, copy=False)
    if msg.encoding == "32FC1":
        depth = data.view(endian + "f4")[:, :width]
        image = np.zeros((height, width), dtype=np.uint16)
        valid = np.isfinite(depth) & (depth > 0)
        image[valid] = np.rint(np.clip(depth[valid] * 1000.0, 0, 65535)).astype(np.uint16)
        return image
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def write_image(path, msg):
    if not cv2.imwrite(str(path), decode_image(msg), [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Cannot write {path}")


def extract(bag, output, workers):
    cv2.setNumThreads(1)
    records = {name: {} for name, _ in IMAGES.values()}
    for _, directory in IMAGES.values():
        (output / directory).mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        def csv_file(path, fields):
            path = output / path
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("w", newline=""))
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(fields)
            return writer

        imu_writer = csv_file("imu/noisy.csv", [
            "timestamp_ns", "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
            "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
        ])
        odom_writer = csv_file("odom/tesse_odom.csv", [
            "timestamp_ns", "frame_id", "child_frame_id", "px", "py", "pz", "qx", "qy", "qz", "qw",
        ])
        imu_count = 0
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers))
        pending = deque()
        with AnyReader([bag], default_typestore=get_typestore(Stores.ROS1_NOETIC)) as reader:
            connections = [c for c in reader.connections if c.topic in IMAGES or c.topic in (IMU, ODOM)]
            progress = stack.enter_context(tqdm(
                total=sum(c.msgcount for c in connections), desc="Extract uHumans2",
                unit="msg", mininterval=0.5, dynamic_ncols=True))
            for connection, _, raw in reader.messages(connections=connections):
                msg = reader.deserialize(raw, connection.msgtype)
                stamp = msg.header.stamp
                timestamp = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
                if connection.topic in IMAGES:
                    name, directory = IMAGES[connection.topic]
                    filename = f"{len(records[name]):06d}.png"
                    path = Path(directory) / filename
                    pending.append(pool.submit(write_image, output / path, msg))
                    records[name][timestamp] = str(path)
                    if len(pending) >= workers * 2:
                        pending.popleft().result()
                        progress.update()
                elif connection.topic == IMU:
                    w, a = msg.angular_velocity, msg.linear_acceleration
                    imu_writer.writerow((timestamp, w.x, w.y, w.z, a.x, a.y, a.z))
                    imu_count += 1
                    progress.update()
                else:
                    p, q = msg.pose.pose.position, msg.pose.pose.orientation
                    odom_writer.writerow((timestamp, msg.header.frame_id, msg.child_frame_id,
                                          p.x, p.y, p.z, q.x, q.y, q.z, q.w))
                    progress.update()
            while pending:
                pending.popleft().result()
                progress.update()
            progress.close()

        if not imu_count or any(not rows for rows in records.values()):
            raise ValueError("Bag is missing required stereo/RGB-D or noisy IMU data")
        for path, streams, columns in (
            ("stereo/mono_pairs.csv", ("left_mono", "right_mono"),
             ("left_mono_file", "right_mono_file")),
            ("rgbd/left_rgb_depth_pairs.csv", ("left_rgb", "depth"),
             ("left_rgb_file", "depth_mm_file")),
        ):
            timestamps = sorted(records[streams[0]].keys() & records[streams[1]].keys())
            if not timestamps:
                raise ValueError(f"No synchronized frames for {path}")
            writer = csv_file(path, ("timestamp_ns", *columns))
            writer.writerows((t, *(records[s][t] for s in streams)) for t in timestamps)

    counts = {name: len(rows) for name, rows in records.items()}
    (output / "metadata").mkdir(exist_ok=True)
    (output / "metadata/export_complete.json").write_text(json.dumps(counts) + "\n")
    print(f"[uHumans2] Exported {counts}; noisy IMU: {imu_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8, help="Parallel image export threads.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    extract(args.bag.resolve(), args.output.resolve(), args.workers)
