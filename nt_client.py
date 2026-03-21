from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ntcore import NetworkTableInstance, PubSubOptions
from wpimath.geometry import Pose2d, Pose3d

POSE2D_PATH = "/AdvantageKit/RealOutputs/RobotModel/Robot"
MECHANISM_PATH = "/AdvantageKit/RealOutputs/RobotModel/MechanismPoses"
FUEL_PATH = "/AdvantageKit/RealOutputs/RobotModel/FuelPositions"
MATCH_TIME_PATH = "/AdvantageKit/DriverStation/MatchTime"

RED_HUB_ACTIVE_PATH = (
    "/SmartDashboard/MapleSim/MatchData/Breakdown/Red Alliance/Improved Active"
)
BLUE_HUB_ACTIVE_PATH = (
    "/SmartDashboard/MapleSim/MatchData/Breakdown/Blue Alliance/Improved Active"
)

RED_SCORE_PATH = (
    "/SmartDashboard/MapleSim/MatchData/Breakdown/Red Alliance/Improved Score"
)
BLUE_SCORE_PATH = (
    "/SmartDashboard/MapleSim/MatchData/Breakdown/Blue Alliance/Improved Score"
)


@dataclass(frozen=True)
class Pose2dData:
    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class Pose3dData:
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float


class NetworkTablesClient:
    def __init__(self, server: str = "127.0.0.1", period: float = 0.02) -> None:
        self._instance = NetworkTableInstance.getDefault()
        self._instance.startClient4("advantage-viewer")
        self._instance.setServer(server)
        self._instance.startDSClient()

        options = PubSubOptions(periodic=period, sendAll=True, keepDuplicates=True)
        self._pose2d_sub = self._instance.getStructTopic(POSE2D_PATH, Pose2d).subscribe(
            Pose2d(), options
        )
        self._mechanism_sub = self._instance.getStructArrayTopic(
            MECHANISM_PATH, Pose3d
        ).subscribe([], options)
        self._fuel_sub = self._instance.getStructArrayTopic(
            FUEL_PATH, Pose3d
        ).subscribe([], options)
        self._red_hub_active_sub = self._instance.getBooleanTopic(
            RED_HUB_ACTIVE_PATH
        ).subscribe(False, options)
        self._blue_hub_active_sub = self._instance.getBooleanTopic(
            BLUE_HUB_ACTIVE_PATH
        ).subscribe(False, options)
        self._red_score_sub = self._instance.getDoubleTopic(RED_SCORE_PATH).subscribe(
            0.0, options
        )
        self._blue_score_sub = self._instance.getDoubleTopic(BLUE_SCORE_PATH).subscribe(
            0.0, options
        )

    def get_pose2d(self) -> Pose2dData:
        pose = self._pose2d_sub.get()
        if isinstance(pose, Pose2d):
            return Pose2dData(
                float(pose.X()), float(pose.Y()), float(pose.rotation().radians())
            )
        return Pose2dData(0.0, 0.0, 0.0)

    def get_mechanism_poses(self) -> list[Pose3dData]:
        poses = self._mechanism_sub.get()
        return _pose3d_structs_to_data(poses)

    def get_fuel_positions(self) -> list[tuple[float, float, float]]:
        poses = self._fuel_sub.get()
        return [(pose.x, pose.y, pose.z) for pose in _pose3d_structs_to_data(poses)]

    def get_red_hub_active(self) -> bool:
        return bool(self._red_hub_active_sub.get())

    def get_blue_hub_active(self) -> bool:
        return bool(self._blue_hub_active_sub.get())

    def get_red_score(self) -> float:
        return float(self._red_score_sub.get())

    def get_blue_score(self) -> float:
        return float(self._blue_score_sub.get())


def _pose3d_structs_to_data(values: Iterable[Pose3d]) -> list[Pose3dData]:
    poses: list[Pose3dData] = []
    for pose in values:
        if not isinstance(pose, Pose3d):
            continue
        rotation = pose.rotation()
        quat = rotation.getQuaternion()
        poses.append(
            Pose3dData(
                x=float(pose.X()),
                y=float(pose.Y()),
                z=float(pose.Z()),
                qw=float(quat.W()),
                qx=float(quat.X()),
                qy=float(quat.Y()),
                qz=float(quat.Z()),
            )
        )
    return poses
