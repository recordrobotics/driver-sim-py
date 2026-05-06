from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Iterable
from wpimath.geometry import Pose3d, Quaternion

import numpy as np

from config import Rotation

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


def rotation_sequence_to_quat(rotations: Iterable[Rotation]) -> Quaternion:
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for rotation in rotations:
        quat = quat_multiply(axis_angle_to_quat(rotation.axis, rotation.degrees), quat)
    return Quaternion(quat[0], quat[1], quat[2], quat[3])

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
