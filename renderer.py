from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2
from pathlib import Path
from typing import Any, Callable

import imgui
import moderngl
import moderngl_window as mglw
import numpy as np
import glm
from moderngl_window.resources.programs import programs as _mglw_programs
from moderngl_window.integrations.imgui import ModernGLRenderer, ModernglWindowMixin
from moderngl_window.scene.camera import OrbitCamera
from PIL import Image

from config import Rotation, load_assets_config
from math_utils import Pose3d, compose, pose2d_matrix, pose_matrix, rotation_sequence_to_quat
from nt_client import NetworkTablesClient


ASSETS_ROOT = Path(__file__).resolve().parent / "assets"
TEXTURE_ROOT = Path(__file__).resolve().parent / "textures"

DS_CAMERA_HEIGHT = 62 * 0.0254
DS_CAMERA_OFFSET_FRC = 1.5
ORBIT_FIELD_FRC_DEFAULT_TARGET = np.array([0.0, 0.5, 0.0], dtype=np.float32)
ORBIT_ROBOT_FRC_DEFAULT_TARGET = np.array([0.0, 0.5, 0.0], dtype=np.float32)
ORBIT_FIELD_FRC_DEFAULT_POSITION = np.array([0.0, 6.0, -12.0], dtype=np.float32)
ORBIT_ROBOT_FRC_DEFAULT_POSITION = np.array([2.0, 1.0, 1.0], dtype=np.float32)

WPILIB_ROTATION = rotation_sequence_to_quat(
    [Rotation(axis="x", degrees=-90), Rotation(axis="y", degrees=180)]
)


def _install_program_cache() -> None:
    if getattr(_mglw_programs, "_cached_load_installed", False):
        return
    _mglw_programs._cached_load_installed = True
    _mglw_programs._load_cache = {}
    _mglw_programs._original_load = _mglw_programs.load

    def _cache_key(meta) -> tuple:
        return (
            meta.kind,
            meta.path,
            meta.vertex_shader,
            meta.fragment_shader,
            meta.geometry_shader,
            meta.tess_control_shader,
            meta.tess_evaluation_shader,
            meta.compute_shader,
            tuple(sorted((meta.defines or {}).items())),
            tuple(meta.varyings or []),
        )

    def _cached_load(meta):
        key = _cache_key(meta)
        cache = _mglw_programs._load_cache
        if key in cache:
            return cache[key]
        program = _mglw_programs._original_load(meta)
        cache[key] = program
        return program

    _mglw_programs.load = _cached_load


_install_program_cache()


class ViewMode(Enum):
    ORBIT_FIELD = "Orbit Field"
    ORBIT_ROBOT = "Orbit Robot"
    DS_1 = "Driver Station 1"
    DS_2 = "Driver Station 2"
    DS_3 = "Driver Station 3"
    DS_4 = "Driver Station 4"
    DS_5 = "Driver Station 5"
    DS_6 = "Driver Station 6"


@dataclass
class FuelInstance:
    matrix: np.ndarray


class AprilTagRenderer:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.program = ctx.program(
            vertex_shader="""
            #version 330
            uniform mat4 mvp;
            in vec3 in_pos;
            in vec2 in_uv;
            out vec2 v_uv;
            void main() {
                gl_Position = mvp * vec4(in_pos, 1.0);
                v_uv = in_uv;
            }
            """,
            fragment_shader="""
            #version 330
            uniform sampler2D tag_tex;
            in vec2 v_uv;
            out vec4 f_color;
            void main() {
                vec4 color = texture(tag_tex, v_uv);
                f_color = color;
            }
            """,
        )
        vertices = np.array(
            [
                -0.005,
                -0.5,
                -0.5,
                0.0,
                0.0,
                -0.005,
                0.5,
                -0.5,
                1.0,
                0.0,
                -0.005,
                0.5,
                0.5,
                1.0,
                1.0,
                -0.005,
                -0.5,
                0.5,
                0.0,
                1.0,
            ],
            dtype="f4",
        )
        indices = np.array([0, 1, 2, 0, 2, 3], dtype="i4")
        self.vbo = ctx.buffer(vertices.tobytes())
        self.ibo = ctx.buffer(indices.tobytes())
        self.vao = ctx.vertex_array(
            self.program,
            [(self.vbo, "3f 2f", "in_pos", "in_uv")],
            self.ibo,
        )
        self.textures: dict[int, moderngl.Texture] = {}

    def load_texture(self, tag_id: int) -> moderngl.Texture:
        if tag_id in self.textures:
            return self.textures[tag_id]
        texture_path = TEXTURE_ROOT / f"{tag_id:03d}.png"
        if not texture_path.exists():
            texture_path = TEXTURE_ROOT / "smile.png"
        image = Image.open(texture_path).convert("RGBA")
        texture = self.ctx.texture(image.size, 4, image.tobytes())
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.textures[tag_id] = texture
        return texture

    def render(self, mvp_bytes: bytes, tag_id: int) -> None:
        texture = self.load_texture(tag_id)
        texture.use(location=0)
        self.program["mvp"].write(mvp_bytes)
        self.vao.render()


class ModernGLImGui(ModernglWindowMixin, ModernGLRenderer):
    pass


class ImGuiBridge(ModernglWindowMixin):
    def __init__(self, wnd: mglw.WindowConfig, ctx: moderngl.Context) -> None:
        self.wnd = wnd
        self.renderer = ModernGLImGui(wnd=wnd, ctx=ctx)
        self.io = self.renderer.io
        keys = self.wnd.keys
        self.REVERSE_KEY_MAP = {
            keys.TAB: imgui.KEY_TAB,
            keys.LEFT: imgui.KEY_LEFT_ARROW,
            keys.RIGHT: imgui.KEY_RIGHT_ARROW,
            keys.UP: imgui.KEY_UP_ARROW,
            keys.DOWN: imgui.KEY_DOWN_ARROW,
            keys.PAGE_UP: imgui.KEY_PAGE_UP,
            keys.PAGE_DOWN: imgui.KEY_PAGE_DOWN,
            keys.HOME: imgui.KEY_HOME,
            keys.END: imgui.KEY_END,
            keys.INSERT: imgui.KEY_INSERT,
            keys.DELETE: imgui.KEY_DELETE,
            keys.BACKSPACE: imgui.KEY_BACKSPACE,
            keys.SPACE: imgui.KEY_SPACE,
            keys.ENTER: imgui.KEY_ENTER,
            keys.ESCAPE: imgui.KEY_ESCAPE,
            keys.A: imgui.KEY_A,
            keys.C: imgui.KEY_C,
            keys.V: imgui.KEY_V,
            keys.X: imgui.KEY_X,
            keys.Y: imgui.KEY_Y,
            keys.Z: imgui.KEY_Z,
        }

    def render(self, draw_data) -> None:
        self.renderer.render(draw_data)

    def shutdown(self) -> None:
        self.renderer.shutdown()

    def key_event(self, key, action, modifiers) -> None:
        if hasattr(self.io, "key_ctrl"):
            self.io.key_ctrl = bool(getattr(modifiers, "ctrl", False))
        if hasattr(self.io, "key_shift"):
            self.io.key_shift = bool(getattr(modifiers, "shift", False))
        if hasattr(self.io, "key_alt"):
            self.io.key_alt = bool(getattr(modifiers, "alt", False))
        super().key_event(key, action, modifiers)

    def mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
        if hasattr(self.io, "mouse_wheel_h"):
            self.io.mouse_wheel_h = x_offset
        self.io.mouse_wheel = y_offset


class RealtimeRenderer(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Realtime Simulation Viewer"
    window_size = (1600, 900)
    aspect_ratio = None
    resource_dir = str(Path(__file__).resolve().parents[1])
    resizable = True

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.assets = load_assets_config(ASSETS_ROOT)
        self.nt_client = NetworkTablesClient()
        imgui.create_context()
        self.imgui = ImGuiBridge(self.wnd, self.ctx)
        self.wnd.swap_interval = 0
        self._field_length_m = self.assets.field.width_inches * 0.0254
        self._field_width_m = self.assets.field.height_inches * 0.0254
        self._coordinate_system = self.assets.field.coordinate_system
        self.camera = OrbitCamera(
            target=(
                float(ORBIT_FIELD_FRC_DEFAULT_TARGET[0]),
                float(ORBIT_FIELD_FRC_DEFAULT_TARGET[1]),
                float(ORBIT_FIELD_FRC_DEFAULT_TARGET[2]),
            ),
            radius=float(np.linalg.norm(ORBIT_FIELD_FRC_DEFAULT_POSITION - ORBIT_FIELD_FRC_DEFAULT_TARGET)),
        )
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio, fov=50.0, near=0.1, far=100.0)
        self.view_mode = ViewMode.ORBIT_FIELD
        self._last_view_mode = self.view_mode
        self.last_robot_pose = np.eye(4, dtype=np.float32)
        self._keys_pressed = set()
        self.field_scene: mglw.scene.Scene | None = None
        self.robot_scene: mglw.scene.Scene | None = None
        self.robot_components: list[mglw.scene.Scene] = []
        self.fuel_scene: mglw.scene.Scene | None = None
        self._scenes_loaded = False
        self.tag_renderer = AprilTagRenderer(self.ctx)
        self.wpilib_matrix = pose_matrix(Pose3d((0.0, 0.0, 0.0), tuple(WPILIB_ROTATION.tolist())))
        self.field_base_matrix = compose(
            self.wpilib_matrix,
            pose_matrix(Pose3d(self.assets.field.position, tuple(rotation_sequence_to_quat(self.assets.field.rotations)))),
        )
        self.robot_base_matrix = pose_matrix(
            Pose3d(self.assets.robot.position, tuple(rotation_sequence_to_quat(self.assets.robot.rotations)))
        )
        self._light_colors = np.array(
            [
                (0.25, 0.45, 1.0),
                (1.0, 0.25, 0.25),
                (1.0, 1.0, 1.0),
            ],
            dtype=np.float32,
        )
        if self._coordinate_system in {"wall-blue", "wall-alliance"}:
            field_center_x = self._field_length_m / 2.0
            field_center_y = self._field_width_m / 2.0
        else:
            field_center_x = 0.0
            field_center_y = 0.0
        light_positions_wpilib = [
            (field_center_x - self._field_length_m / 4.0, field_center_y, 6.0),
            (field_center_x + self._field_length_m / 4.0, field_center_y, 6.0),
            (field_center_x, field_center_y, 6.5),
        ]
        self._light_positions_world = np.array(
            [
                self._transform_point(
                    self.wpilib_matrix,
                    np.array(self._convert_translation(pos[0], pos[1], pos[2]), dtype=np.float32),
                )
                for pos in light_positions_wpilib
            ],
            dtype=np.float32,
        )
        self._light_positions_view = self._light_positions_world.copy()
        self._light_program_cache: set[int] = set()
        self._staged_fuel_nodes: dict[Any, Any] = {}

    def _to_glm(self, matrix: np.ndarray) -> glm.mat4:
        return glm.mat4(*matrix.flatten(order="F").tolist())

    def _normalize_mesh_name(self, name: str) -> str:
        return (
            name.replace(":", "")
            .replace(" ", "_")
            .replace("__", "_")
            .lower()
        )

    def _load_scenes(self) -> None:
        if self._scenes_loaded:
            return
        self.field_scene = self.load_scene(str(self.assets.field_dir / "model.glb"))
        self.robot_scene = self.load_scene(str(self.assets.robot_dir / "model.glb"))
        self.robot_components = [
            self.load_scene(str(self.assets.robot_dir / f"model_{index}.glb"))
            for index in range(len(self.assets.robot.components))
        ]
        self.fuel_scene = self.load_scene(str(self.assets.field_dir / "model_0.glb"))
        staged_objects: list[str] = []
        for game_piece in self.assets.field.game_pieces:
            if game_piece.name.lower() == "fuel":
                staged_objects.extend(game_piece.staged_objects)
        if staged_objects and self.field_scene is not None:
            staged_set = {self._normalize_mesh_name(name) for name in staged_objects}
            for node in self.field_scene.nodes:
                if node.mesh is None or not node.name:
                    continue
                if self._normalize_mesh_name(node.name) in staged_set:
                    self._staged_fuel_nodes[node] = node.mesh
        self._scenes_loaded = True

    def _camera_from_driver_station(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        stations = self.assets.field.driver_stations
        if index < 0 or index >= len(stations):
            return ORBIT_FIELD_FRC_DEFAULT_POSITION, ORBIT_FIELD_FRC_DEFAULT_TARGET
        pos_x, pos_y = stations[index]
        heading = atan2(-pos_y, -pos_x)
        forward = np.array([np.cos(heading), np.sin(heading), 0.0], dtype=np.float32)
        camera_pos = np.array([pos_x, pos_y, 0.0], dtype=np.float32) + forward * -DS_CAMERA_OFFSET_FRC
        camera_pos[2] = DS_CAMERA_HEIGHT
        target = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        camera_pos = self._transform_point(self.wpilib_matrix, camera_pos)
        target = self._transform_point(self.wpilib_matrix, target)
        return camera_pos, target

    def _convert_pose2d(self, x: float, y: float, theta: float) -> tuple[float, float, float]:
        if self._coordinate_system == "wall-blue":
            return (
                self._field_length_m / 2.0 - x,
                self._field_width_m / 2.0 - y,
                theta + np.pi,
            )
        if self._coordinate_system == "center-rotated":
            return (y, -x, theta - np.pi / 2.0)
        if self._coordinate_system == "wall-alliance":
            return (
                self._field_length_m / 2.0 - x,
                self._field_width_m / 2.0 - y,
                theta + np.pi,
            )
        return (x, y, theta)

    def _convert_translation(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        if self._coordinate_system == "wall-blue":
            return (
                self._field_length_m / 2.0 - x,
                self._field_width_m / 2.0 - y,
                z,
            )
        if self._coordinate_system == "center-rotated":
            return (y, -x, z)
        if self._coordinate_system == "wall-alliance":
            return (
                self._field_length_m / 2.0 - x,
                self._field_width_m / 2.0 - y,
                z,
            )
        return (x, y, z)

    def _camera_projection(self):
        return self.camera.projection.matrix

    def _camera_view(self):
        return self.camera.matrix

    def _update_camera(self) -> None:
        if self.view_mode != self._last_view_mode:
            if self.view_mode == ViewMode.ORBIT_FIELD:
                field_pos = self._transform_point(self.wpilib_matrix, ORBIT_FIELD_FRC_DEFAULT_POSITION)
                field_target = self._transform_point(self.wpilib_matrix, ORBIT_FIELD_FRC_DEFAULT_TARGET)
                self._set_orbit_from_position(field_pos, field_target)
            elif self.view_mode == ViewMode.ORBIT_ROBOT:
                robot_pos = np.array(self.last_robot_pose[:3, 3], dtype=np.float32)
                robot_offset = ORBIT_ROBOT_FRC_DEFAULT_POSITION - ORBIT_ROBOT_FRC_DEFAULT_TARGET
                robot_cam_pos = robot_pos + robot_offset
                self._set_orbit_from_position(robot_cam_pos, robot_pos)
            else:
                index = list(ViewMode).index(self.view_mode) - 2
                camera_pos, target = self._camera_from_driver_station(index)
                self._set_orbit_from_position(camera_pos, target)
            self._last_view_mode = self.view_mode
        else:
            if self.view_mode == ViewMode.ORBIT_ROBOT:
                robot_pos = self.last_robot_pose[:3, 3]
                self.camera.target = glm.vec3(float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2]))
            elif self.view_mode not in {ViewMode.ORBIT_FIELD, ViewMode.ORBIT_ROBOT}:
                index = list(ViewMode).index(self.view_mode) - 2
                camera_pos, target = self._camera_from_driver_station(index)
                self._set_orbit_from_position(camera_pos, target)

    def _set_orbit_from_position(self, position: np.ndarray, target: np.ndarray) -> None:
        vector = np.array(position, dtype=np.float32) - np.array(target, dtype=np.float32)
        radius = float(np.linalg.norm(vector))
        if radius <= 0:
            return
        angle_y = -float(np.degrees(np.arccos(np.clip(vector[1] / radius, -1.0, 1.0))))
        sin_y = np.sin(np.radians(angle_y))
        angle_x = 90.0 if abs(sin_y) < 1e-6 else float(np.degrees(np.arcsin(np.clip(vector[2] / (radius * sin_y), -1.0, 1.0))))
        self.camera.target = glm.vec3(float(target[0]), float(target[1]), float(target[2]))
        self.camera.radius = radius
        self.camera.angle_x = angle_x
        self.camera.angle_y = angle_y

    def _transform_point(self, matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
        vec = np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
        result = matrix @ vec
        return result[:3]

    def _robot_pose_matrix(self) -> np.ndarray:
        pose2d = self.nt_client.get_pose2d()
        x, y, theta = self._convert_pose2d(pose2d.x, pose2d.y, pose2d.theta)
        base_pose = pose2d_matrix(x, y, theta)
        return self.wpilib_matrix @ base_pose @ self.robot_base_matrix

    def _component_matrices(self) -> list[np.ndarray]:
        robot_pose = self.nt_client.get_pose2d()
        x, y, theta = self._convert_pose2d(robot_pose.x, robot_pose.y, robot_pose.theta)
        robot_pose_matrix = pose2d_matrix(x, y, theta)
        component_poses = self.nt_client.get_mechanism_poses()
        matrices: list[np.ndarray] = []
        for index, component in enumerate(self.assets.robot.components):
            if index < len(component_poses):
                pose = component_poses[index]
                user_pose = pose_matrix(
                    Pose3d((pose.x, pose.y, pose.z), (pose.qw, pose.qx, pose.qy, pose.qz))
                )
            else:
                user_pose = np.eye(4, dtype=np.float32)
            config_pose = pose_matrix(
                Pose3d(
                    component.zeroed_position,
                    tuple(rotation_sequence_to_quat(component.zeroed_rotations)),
                )
            )
            matrices.append(self.wpilib_matrix @ robot_pose_matrix @ user_pose @ config_pose)
        return matrices

    def _fuel_instances(self, positions: list[tuple[float, float, float]]) -> list[FuelInstance]:
        fuels = []
        for position in positions:
            converted = self._convert_translation(position[0], position[1], position[2])
            matrix = self.wpilib_matrix @ pose_matrix(Pose3d(converted, (1.0, 0.0, 0.0, 0.0)))
            fuels.append(FuelInstance(matrix=matrix))
        return fuels

    def _set_staged_fuel_visible(self, visible: bool) -> None:
        for node, mesh in self._staged_fuel_nodes.items():
            node.mesh = mesh if visible else None

    def _mesh_requires_blend(self, mesh) -> bool:
        material = getattr(mesh, "material", None)
        if material is None:
            return False
        color = getattr(material, "color", None)
        if color is None:
            return False
        return len(color) >= 4 and color[3] < 0.999

    def _update_light_positions_view(self, camera: glm.mat4) -> None:
        view_positions = []
        for pos in self._light_positions_world:
            view = np.array(
                camera * glm.vec4(float(pos[0]), float(pos[1]), float(pos[2]), 1.0),
                dtype=np.float32,
            )
            view_positions.append((float(view[0]), float(view[1]), float(view[2])))
        self._light_positions_view = np.array(view_positions, dtype=np.float32)

    def _apply_lights_to_mesh(self, mesh) -> None:
        mesh_program = getattr(mesh, "mesh_program", None)
        if mesh_program is None:
            return
        program = getattr(mesh_program, "program", None)
        if program is None:
            return
        program_id = id(program)
        if program_id in self._light_program_cache:
            return
        self._light_program_cache.add(program_id)
        try:
            program["u_light_pos"].write(self._light_positions_view.tobytes())
            program["u_light_color"].write(self._light_colors.tobytes())
            program["u_specular_strength"].value = 0.6
            program["u_shininess"].value = 32.0
        except KeyError:
            return

    def _collect_drawables(
        self, node, drawables: list[tuple[Callable[..., None], Any, bool]]
    ) -> None:
        mesh = getattr(node, "mesh", None)
        if mesh is not None:
            def _draw_fn(projection: glm.mat4, model: glm.mat4, camera: glm.mat4) -> None:
                self._apply_lights_to_mesh(mesh)
                mesh.draw(projection_matrix=projection, model_matrix=model, camera_matrix=camera)

            drawables.append((_draw_fn, node.matrix_global, self._mesh_requires_blend(mesh)))
        for child in getattr(node, "children", []):
            self._collect_drawables(child, drawables)

    def _collect_scene_drawables(
        self, scene: mglw.scene.Scene, matrix: np.ndarray, drawables: list[tuple[Callable[..., None], Any, bool]]
    ) -> None:
        scene.matrix = self._to_glm(matrix)
        for node in scene.root_nodes:
            self._collect_drawables(node, drawables)

    def _collect_apriltag_drawables(self, drawables: list[tuple[Callable[..., None], Any, bool]]) -> None:
        wpilib_glm = self._to_glm(self.wpilib_matrix)
        for tag in self.assets.field.april_tags:
            size = self._apriltag_size(tag.variant)
            scale = self._glm_scale(0.01, size, size)
            tag_model = wpilib_glm * self._to_glm(
                pose_matrix(Pose3d(tag.position, tuple(rotation_sequence_to_quat(tag.rotations))))
            )

            def _draw_fn(
                projection: glm.mat4,
                model: glm.mat4,
                camera: glm.mat4,
                scale=scale,
                tag_id=tag.tag_id,
            ) -> None:
                mvp = projection * camera * model * scale
                self.tag_renderer.render(self._glm_to_bytes(glm.mat4(mvp)), tag_id)

            drawables.append((_draw_fn, tag_model, True))

    def _render_drawables(self, drawables: list[tuple[Callable[..., None], Any, bool]]) -> None:
        if not drawables:
            return
        projection = self._camera_projection()
        camera = self._camera_view()
        self._update_light_positions_view(camera)
        self._light_program_cache.clear()
        opaque = [(draw_fn, model) for draw_fn, model, is_transparent in drawables if not is_transparent]
        transparent = [(draw_fn, model) for draw_fn, model, is_transparent in drawables if is_transparent]

        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True
        for draw_fn, model in opaque:
            draw_fn(projection, model, camera)

        if transparent:
            cam_pos = glm.vec3(glm.inverse(camera)[3])

            def _distance_sq(model: glm.mat4) -> float:
                world_pos = glm.vec3(model[3])
                diff = world_pos - cam_pos
                return float(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z)

            transparent.sort(key=lambda item: _distance_sq(item[1]), reverse=True)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.ctx.depth_mask = False
            for draw_fn, model in transparent:
                draw_fn(projection, model, camera)
            self.ctx.depth_mask = True
            self.ctx.disable(moderngl.BLEND)

    def render(self, time: float, frame_time: float) -> None:
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.clear(0.18, 0.18, 0.2)
        self._update_camera()
        self._apply_keyboard_controls(frame_time)

        self._load_scenes()
        if not self._scenes_loaded:
            return

        fuel_positions = self.nt_client.get_fuel_positions()
        self._set_staged_fuel_visible(len(fuel_positions) == 0)

        if self.field_scene is None or self.robot_scene is None or self.fuel_scene is None:
            return

        drawables: list[tuple[Callable[..., None], Any, bool]] = []
        self._collect_scene_drawables(self.field_scene, self.field_base_matrix, drawables)
        self.last_robot_pose = self._robot_pose_matrix()
        self._collect_scene_drawables(self.robot_scene, self.last_robot_pose, drawables)

        for component_scene, matrix in zip(self.robot_components, self._component_matrices()):
            self._collect_scene_drawables(component_scene, matrix, drawables)

        for fuel in self._fuel_instances(fuel_positions):
            self._collect_scene_drawables(self.fuel_scene, fuel.matrix, drawables)

        self._collect_apriltag_drawables(drawables)

        self._render_drawables(drawables)
        self._render_ui()

    def _apply_keyboard_controls(self, frame_time: float) -> None:
        if self.view_mode not in {ViewMode.ORBIT_FIELD, ViewMode.ORBIT_ROBOT}:
            return
        keys = self.wnd.keys
        speed = 6.0 * frame_time
        orbit_speed = 120.0 * frame_time
        zoom_speed = 6.0 * frame_time
        if keys.UP in self._keys_pressed:
            self.camera.angle_y -= orbit_speed
        if keys.DOWN in self._keys_pressed:
            self.camera.angle_y += orbit_speed
        if keys.LEFT in self._keys_pressed:
            self.camera.angle_x -= orbit_speed
        if keys.RIGHT in self._keys_pressed:
            self.camera.angle_x += orbit_speed
        if keys.W in self._keys_pressed:
            self.camera.radius = max(0.5, self.camera.radius - zoom_speed)
        if keys.S in self._keys_pressed:
            self.camera.radius += zoom_speed
        if keys.A in self._keys_pressed:
            self._nudge_target(-speed, 0.0, 0.0)
        if keys.D in self._keys_pressed:
            self._nudge_target(speed, 0.0, 0.0)
        if keys.Q in self._keys_pressed:
            self._nudge_target(0.0, speed, 0.0)
        if keys.E in self._keys_pressed:
            self._nudge_target(0.0, -speed, 0.0)

    def on_render(self, time: float, frame_time: float) -> None:
        self.render(time, frame_time)

    def _render_apriltags(self) -> None:
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.depth_mask = False
        for tag in self.assets.field.april_tags:
            size = self._apriltag_size(tag.variant)
            scale = self._glm_scale(0.01, size, size)
            mvp = (
                self._camera_projection()
                * self._camera_view()
                * self._to_glm(self.wpilib_matrix)
                * self._to_glm(pose_matrix(Pose3d(tag.position, tuple(rotation_sequence_to_quat(tag.rotations)))))
                * scale
            )
            self.tag_renderer.render(self._glm_to_bytes(glm.mat4(mvp)), tag.tag_id)
        self.ctx.depth_mask = True
        self.ctx.disable(moderngl.BLEND)

    def _apriltag_size(self, variant: str) -> float:
        parts = variant.split("-")
        if len(parts) >= 2 and parts[1].endswith("in"):
            inches = float(parts[1].replace("in", ""))
        else:
            inches = 6.5
        size = inches * 0.0254
        if parts[0] == "36h11":
            size *= 10 / 8
        else:
            size *= 8 / 6
        return size

    def _glm_to_bytes(self, matrix: glm.mat4) -> bytes:
        return np.array(matrix, dtype=np.float32).tobytes(order="F")

    def _glm_scale(self, x: float, y: float, z: float) -> glm.mat4:
        matrix = glm.mat4(1.0)
        matrix[0][0] = x
        matrix[1][1] = y
        matrix[2][2] = z
        return matrix

    def _render_ui(self) -> None:
        imgui.new_frame()
        imgui.set_next_window_position(16, 16, imgui.FIRST_USE_EVER)
        imgui.begin("View", True, imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        current_index = list(ViewMode).index(self.view_mode)
        changed, new_index = imgui.combo(
            "Camera",
            current_index,
            [mode.value for mode in ViewMode],
        )
        if changed:
            self.view_mode = list(ViewMode)[new_index]
        imgui.text("NetworkTables: 127.0.0.1")
        imgui.end()
        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        self.imgui.mouse_drag_event(x, y, dx, dy)
        if self.view_mode in {ViewMode.ORBIT_FIELD, ViewMode.ORBIT_ROBOT}:
            mouse_states = getattr(self.wnd, "mouse_states", None)
            if mouse_states is None or mouse_states.left:
                self.camera.rot_state(dx, dy)
            if mouse_states is not None and mouse_states.right:
                self._pan_target(dx, dy)

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        self.mouse_drag_event(x, y, dx, dy)

    def mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
        self.imgui.mouse_scroll_event(x_offset, y_offset)
        if self.view_mode in {ViewMode.ORBIT_FIELD, ViewMode.ORBIT_ROBOT}:
            self.camera.zoom_state(y_offset)

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
        self.mouse_scroll_event(x_offset, y_offset)

    def _pan_target(self, dx: float, dy: float) -> None:
        scale = 0.01
        offset = glm.vec3(-dx * scale, dy * scale, 0.0)
        self.camera.target += offset

    def _nudge_target(self, dx: float, dy: float, dz: float) -> None:
        self.camera.target += glm.vec3(dx, dy, dz)

    def resize(self, width: int, height: int) -> None:
        self.camera.projection.update(aspect_ratio=width / height, near=0.1, far=100.0)
        self.imgui.resize(width, height)

    def on_resize(self, width: int, height: int) -> None:
        self.resize(width, height)

    def close(self) -> None:
        self.imgui.shutdown()

    def on_close(self) -> None:
        self.close()

    def mouse_position_event(self, x: int, y: int, dx: int, dy: int) -> None:
        self.imgui.mouse_position_event(x, y, dx, dy)

    def on_mouse_position_event(self, x: int, y: int, dx: int, dy: int) -> None:
        self.mouse_position_event(x, y, dx, dy)

    def mouse_press_event(self, x: int, y: int, button: int) -> None:
        self.imgui.mouse_press_event(x, y, button)

    def on_mouse_press_event(self, x: int, y: int, button: int) -> None:
        self.mouse_press_event(x, y, button)

    def mouse_release_event(self, x: int, y: int, button: int) -> None:
        self.imgui.mouse_release_event(x, y, button)

    def on_mouse_release_event(self, x: int, y: int, button: int) -> None:
        self.mouse_release_event(x, y, button)

    def key_event(self, key: int, action: int, modifiers) -> None:
        self.imgui.key_event(key, action, modifiers)
        if action == self.wnd.keys.ACTION_PRESS:
            self._keys_pressed.add(key)
        elif action == self.wnd.keys.ACTION_RELEASE:
            self._keys_pressed.discard(key)

    def on_key_event(self, key: int, action: int, modifiers) -> None:
        self.key_event(key, action, modifiers)

    def char_event(self, char: str) -> None:
        self.imgui.unicode_char_entered(char)

    def on_unicode_char_entered(self, char: str) -> None:
        self.char_event(char)
