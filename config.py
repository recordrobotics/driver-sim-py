from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Rotation:
    axis: str
    degrees: float


@dataclass(frozen=True)
class GamePieceConfig:
    name: str
    rotations: list[Rotation]
    position: tuple[float, float, float]
    staged_objects: list[str]


@dataclass(frozen=True)
class AprilTagConfig:
    variant: str
    tag_id: int
    rotations: list[Rotation]
    position: tuple[float, float, float]


@dataclass(frozen=True)
class FieldConfig:
    name: str
    is_ftc: bool
    coordinate_system: str
    rotations: list[Rotation]
    position: tuple[float, float, float]
    width_inches: float
    height_inches: float
    driver_stations: list[tuple[float, float]]
    april_tags: list[AprilTagConfig]
    game_pieces: list[GamePieceConfig]


@dataclass(frozen=True)
class RobotComponentConfig:
    zeroed_rotations: list[Rotation]
    zeroed_position: tuple[float, float, float]


@dataclass(frozen=True)
class RobotConfig:
    name: str
    rotations: list[Rotation]
    position: tuple[float, float, float]
    components: list[RobotComponentConfig]
    is_ftc: bool = False


@dataclass(frozen=True)
class AssetsConfig:
    field: FieldConfig
    robot: RobotConfig
    field_dir: Path
    robot_dir: Path


class ConfigError(RuntimeError):
    pass


def _rotation_list(raw: Any) -> list[Rotation]:
    if not isinstance(raw, list):
        return []
    rotations: list[Rotation] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        axis = entry.get("axis")
        degrees = entry.get("degrees")
        if axis in {"x", "y", "z"} and isinstance(degrees, (int, float)):
            rotations.append(Rotation(axis=axis, degrees=float(degrees)))
    return rotations


def _position(raw: Any) -> tuple[float, float, float]:
    if not isinstance(raw, Iterable):
        return (0.0, 0.0, 0.0)
    values = list(raw)
    if len(values) != 3 or not all(isinstance(v, (int, float)) for v in values):
        return (0.0, 0.0, 0.0)
    return (float(values[0]), float(values[1]), float(values[2]))


def _driver_stations(raw: Any) -> list[tuple[float, float]]:
    if not isinstance(raw, list):
        return []
    stations: list[tuple[float, float]] = []
    for entry in raw:
        if not isinstance(entry, Iterable):
            continue
        values = list(entry)
        if len(values) != 2 or not all(isinstance(v, (int, float)) for v in values):
            continue
        stations.append((float(values[0]), float(values[1])))
    return stations


def parse_field_config(config: dict[str, Any]) -> FieldConfig:
    name = str(config.get("name", ""))
    is_ftc = bool(config.get("isFTC", False))
    coordinate_system = str(config.get("coordinateSystem", "center-red"))
    rotations = _rotation_list(config.get("rotations", []))
    position = _position(config.get("position", (0, 0, 0)))
    width_inches = float(config.get("widthInches", 0.0))
    height_inches = float(config.get("heightInches", 0.0))

    april_tags: list[AprilTagConfig] = []
    for entry in config.get("aprilTags", []) if isinstance(config.get("aprilTags"), list) else []:
        if not isinstance(entry, dict):
            continue
        variant = str(entry.get("variant", ""))
        tag_id = int(entry.get("id", -1))
        april_tags.append(
            AprilTagConfig(
                variant=variant,
                tag_id=tag_id,
                rotations=_rotation_list(entry.get("rotations", [])),
                position=_position(entry.get("position", (0, 0, 0)))
            )
        )

    game_pieces: list[GamePieceConfig] = []
    for entry in config.get("gamePieces", []) if isinstance(config.get("gamePieces"), list) else []:
        if not isinstance(entry, dict):
            continue
        game_pieces.append(
            GamePieceConfig(
                name=str(entry.get("name", "")),
                rotations=_rotation_list(entry.get("rotations", [])),
                position=_position(entry.get("position", (0, 0, 0))),
                staged_objects=[str(x) for x in entry.get("stagedObjects", []) if isinstance(x, str)],
            )
        )

    driver_stations = _driver_stations(config.get("driverStations", []))

    if not name:
        raise ConfigError("Field config missing name.")
    return FieldConfig(
        name=name,
        is_ftc=is_ftc,
        coordinate_system=coordinate_system,
        rotations=rotations,
        position=position,
        width_inches=width_inches,
        height_inches=height_inches,
        driver_stations=driver_stations,
        april_tags=april_tags,
        game_pieces=game_pieces,
    )


def parse_robot_config(config: dict[str, Any]) -> RobotConfig:
    name = str(config.get("name", ""))
    rotations = _rotation_list(config.get("rotations", []))
    position = _position(config.get("position", (0, 0, 0)))
    components: list[RobotComponentConfig] = []
    for entry in config.get("components", []) if isinstance(config.get("components"), list) else []:
        if not isinstance(entry, dict):
            continue
        components.append(
            RobotComponentConfig(
                zeroed_rotations=_rotation_list(entry.get("zeroedRotations", [])),
                zeroed_position=_position(entry.get("zeroedPosition", (0, 0, 0))),
            )
        )

    if not name:
        raise ConfigError("Robot config missing name.")
    return RobotConfig(name=name, rotations=rotations, position=position, components=components)


def load_assets_config(assets_dir: Path) -> AssetsConfig:
    field_dir = assets_dir / "Field3d_2026FRCFieldV1"
    robot_dir = assets_dir / "Robot_M2026"

    field_path = field_dir / "config.json"
    robot_path = robot_dir / "config.json"
    if not field_path.exists() or not robot_path.exists():
        raise ConfigError(f"Missing config.json in {field_dir} or {robot_dir}.")

    field_raw = json.loads(field_path.read_text(encoding="utf-8"))
    robot_raw = json.loads(robot_path.read_text(encoding="utf-8"))
    field_config = parse_field_config(field_raw)
    robot_config = parse_robot_config(robot_raw)

    return AssetsConfig(
        field=field_config,
        robot=robot_config,
        field_dir=field_dir,
        robot_dir=robot_dir,
    )
