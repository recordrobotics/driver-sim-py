from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Iterable

import numpy as np

from config import Rotation


@dataclass(frozen=True)
class Pose3d:
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


def degrees_to_radians(degrees: float) -> float:
    return degrees * (np.pi / 180.0)


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float32,
    )


def axis_angle_to_quat(axis: str, degrees: float) -> np.ndarray:
    angle = degrees_to_radians(degrees)
    half = angle * 0.5
    s = sin(half)
    if axis == "x":
        return np.array([cos(half), s, 0.0, 0.0], dtype=np.float32)
    if axis == "y":
        return np.array([cos(half), 0.0, s, 0.0], dtype=np.float32)
    return np.array([cos(half), 0.0, 0.0, s], dtype=np.float32)


def rotation_sequence_to_quat(rotations: Iterable[Rotation]) -> np.ndarray:
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for rotation in rotations:
        quat = quat_multiply(axis_angle_to_quat(rotation.axis, rotation.degrees), quat)
    return quat


def quat_to_mat4(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def pose_matrix(pose: Pose3d) -> np.ndarray:
    matrix = quat_to_mat4(np.array(pose.rotation, dtype=np.float32))
    matrix[:3, 3] = np.array(pose.translation, dtype=np.float32)
    return matrix


def pose2d_matrix(x: float, y: float, theta: float) -> np.ndarray:
    c = cos(theta)
    s = sin(theta)
    return np.array(
        [
            [c, -s, 0.0, x],
            [s, c, 0.0, y],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def translate_matrix(translation: tuple[float, float, float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.array(translation, dtype=np.float32)
    return matrix


def scale_matrix(scale: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = scale
    matrix[1, 1] = scale
    matrix[2, 2] = scale
    return matrix


def compose(*matrices: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float32)
    for matrix in matrices:
        result = result @ matrix
    return result
