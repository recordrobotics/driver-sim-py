from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ntcore import NetworkTableInstance, PubSubOptions
from wpimath.geometry import Pose3d

POSE3D_PATH = "/AdvantageKit/RealOutputs/RobotModel/Robot"
MECHANISM_PATH = "/AdvantageKit/RealOutputs/RobotModel/MechanismPoses"
FUEL_PATH = "/AdvantageKit/RealOutputs/RobotModel/FuelPositions"
MATCH_TIME_PATH = "/AdvantageKit/DriverStation/MatchTime"
AUTONOMOUS_PATH = "/AdvantageKit/DriverStation/Autonomous"
ALLIANCE_STATION_PATH = "/AdvantageKit/DriverStation/AllianceStation"

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

class NetworkTablesClient:

    def __init__(self, server: str = "127.0.0.1", period: float = 0.02) -> None:
        self._instance = NetworkTableInstance.getDefault()
        self._instance.startClient4("advantage-viewer")
        self._instance.setServer(server)
        self._instance.startDSClient()

        options = PubSubOptions(periodic=period, sendAll=True, keepDuplicates=True)
        self._pose3d_sub = self._instance.getStructTopic(POSE3D_PATH, Pose3d).subscribe(
            Pose3d(), options
        )
        self._mechanism_sub = self._instance.getStructArrayTopic(
            MECHANISM_PATH, Pose3d
        ).subscribe([], options)
        self._fuel_sub = self._instance.getStructArrayTopic(
            FUEL_PATH, Pose3d
        ).subscribe([], options)
        self._match_time_sub = self._instance.getDoubleTopic(MATCH_TIME_PATH).subscribe(
            0.0, options
        )
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
        self._autonomous_sub = self._instance.getBooleanTopic(
            AUTONOMOUS_PATH
        ).subscribe(False, options)

        self._alliance_station_sub = self._instance.getIntegerTopic(
            ALLIANCE_STATION_PATH
        ).subscribe(1, options)

    def get_autonomous(self) -> bool:
        return bool(self._autonomous_sub.get())

    def get_alliance_station(self) -> int:
        return int(self._alliance_station_sub.get())

    def get_pose3d(self) -> Pose3d:
        return self._pose3d_sub.get()

    def get_mechanism_poses(self) -> list[Pose3d]:
        return self._mechanism_sub.get()

    def get_fuel_positions(self) -> list[Pose3d]:
        return self._fuel_sub.get()

    def get_match_time(self) -> float:
        return float(self._match_time_sub.get())

    def get_red_hub_active(self) -> bool:
        return bool(self._red_hub_active_sub.get())

    def get_blue_hub_active(self) -> bool:
        return bool(self._blue_hub_active_sub.get())

    def get_red_score(self) -> float:
        return float(self._red_score_sub.get())

    def get_blue_score(self) -> float:
        return float(self._blue_score_sub.get())
