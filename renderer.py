from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2
from pathlib import Path
from typing import Any, cast

import imgui
import moderngl
import moderngl_window as mglw
import numpy as np
import glm
from moderngl_window.resources.programs import programs as _mglw_programs
from moderngl_window.conf import settings
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


@dataclass
class Renderable:
    mesh: Any
    program: moderngl.Program | None
    base_matrix: glm.mat4
    model_matrix: glm.mat4
    is_transparent: bool
    active: bool = True


@dataclass
class ProgramUniformCache:
    program: moderngl.Program
    light_pos: Any | None
    light_color: Any | None
    specular_strength: Any | None
    shininess: Any | None
    last_version: int = -1

    def update_lights(
        self,
        positions_bytes: bytes,
        colors_bytes: bytes,
        specular_strength: float,
        shininess: float,
        version: int,
    ) -> None:
        if self.last_version == version:
            return
        if self.light_pos is not None:
            self.light_pos.write(positions_bytes)
        if self.light_color is not None:
            self.light_color.write(colors_bytes)
        if self.specular_strength is not None:
            self.specular_strength.value = specular_strength
        if self.shininess is not None:
            self.shininess.value = shininess
        self.last_version = version


@dataclass
class AprilTagDrawable:
    model_matrix: glm.mat4
    scale_matrix: glm.mat4
    tag_id: int


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
        self.mvp_uniform = cast(Any, self.program["mvp"])
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
        self.mvp_uniform.write(mvp_bytes)
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
        shader_dir = Path(__file__).resolve().parent / "shaders"
        shader_dir_str = str(shader_dir)
        settings.PROGRAM_DIRS = [shader_dir_str]
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
                (0.25, 0.45, 1.0, 1.0),
                (1.0, 0.25, 0.25, 1.0),
                (1.0, 1.0, 1.0, 1.0),
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
        self._light_positions_view = np.column_stack(
            (self._light_positions_world, np.ones(len(self._light_positions_world), dtype=np.float32))
        )
        self._light_positions_view_bytes = self._light_positions_view.tobytes()
        self._light_colors_bytes = self._light_colors.tobytes()
        self._light_state_version = 0
        self._specular_strength = 0.6
        self._shininess = 32.0
        self._program_uniforms: dict[int, ProgramUniformCache] = {}
        self._staged_fuel_nodes: dict[Any, Any] = {}
        self._camera_pos = glm.vec3(0.0, 0.0, 0.0)
        self.static_drawables: list[Renderable] = []
        self.dynamic_drawables: list[Renderable] = []
        self._opaque_drawables: list[Renderable] = []
        self._transparent_drawables: list[Renderable] = []
        self._robot_drawables: list[Renderable] = []
        self._component_drawables: list[list[Renderable]] = []
        self._fuel_templates: list[Renderable] = []
        self._fuel_drawables: list[Renderable] = []
        self._fuel_active_count = 0
        self._apriltag_drawables: list[AprilTagDrawable] = []

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
        self._build_renderables()
        self._scenes_loaded = True

    def _iter_scene_meshes(self, scene: mglw.scene.Scene) -> list[tuple[Any, glm.mat4]]:
        meshes: list[tuple[Any, glm.mat4]] = []
        stack = list(scene.root_nodes)
        while stack:
            node = stack.pop()
            mesh = getattr(node, "mesh", None)
            if mesh is not None:
                meshes.append((mesh, node.matrix_global))
            stack.extend(getattr(node, "children", []))
        return meshes

    def _register_renderable(self, renderable: Renderable, is_dynamic: bool) -> None:
        if is_dynamic:
            self.dynamic_drawables.append(renderable)
        else:
            self.static_drawables.append(renderable)
        if renderable.is_transparent:
            self._transparent_drawables.append(renderable)
        else:
            self._opaque_drawables.append(renderable)

    def _build_renderables(self) -> None:
        self.static_drawables.clear()
        self.dynamic_drawables.clear()
        self._opaque_drawables.clear()
        self._transparent_drawables.clear()
        self._robot_drawables.clear()
        self._component_drawables.clear()
        self._fuel_templates.clear()
        self._fuel_drawables.clear()
        self._fuel_active_count = 0
        self._apriltag_drawables.clear()

        field_base_glm = self._to_glm(self.field_base_matrix)

        if self.field_scene is not None:
            self.field_scene.matrix = glm.mat4(1.0)
            for mesh, node_matrix in self._iter_scene_meshes(self.field_scene):
                model = field_base_glm * node_matrix
                renderable = Renderable(
                    mesh=mesh,
                    program=getattr(getattr(mesh, "mesh_program", None), "program", None),
                    base_matrix=node_matrix,
                    model_matrix=model,
                    is_transparent=self._mesh_requires_blend(mesh),
                )
                self._register_renderable(renderable, is_dynamic=False)

        if self.robot_scene is not None:
            self.robot_scene.matrix = glm.mat4(1.0)
            for mesh, node_matrix in self._iter_scene_meshes(self.robot_scene):
                renderable = Renderable(
                    mesh=mesh,
                    program=getattr(getattr(mesh, "mesh_program", None), "program", None),
                    base_matrix=node_matrix,
                    model_matrix=node_matrix,
                    is_transparent=self._mesh_requires_blend(mesh),
                )
                self._robot_drawables.append(renderable)
                self._register_renderable(renderable, is_dynamic=True)

        for component_scene in self.robot_components:
            component_scene.matrix = glm.mat4(1.0)
            component_drawables: list[Renderable] = []
            for mesh, node_matrix in self._iter_scene_meshes(component_scene):
                renderable = Renderable(
                    mesh=mesh,
                    program=getattr(getattr(mesh, "mesh_program", None), "program", None),
                    base_matrix=node_matrix,
                    model_matrix=node_matrix,
                    is_transparent=self._mesh_requires_blend(mesh),
                )
                component_drawables.append(renderable)
                self._register_renderable(renderable, is_dynamic=True)
            self._component_drawables.append(component_drawables)

        if self.fuel_scene is not None:
            self.fuel_scene.matrix = glm.mat4(1.0)
            for mesh, node_matrix in self._iter_scene_meshes(self.fuel_scene):
                template = Renderable(
                    mesh=mesh,
                    program=getattr(getattr(mesh, "mesh_program", None), "program", None),
                    base_matrix=node_matrix,
                    model_matrix=node_matrix,
                    is_transparent=self._mesh_requires_blend(mesh),
                )
                self._fuel_templates.append(template)

        self._build_apriltag_drawables()

    def _build_apriltag_drawables(self) -> None:
        wpilib_glm = self._to_glm(self.wpilib_matrix)
        for tag in self.assets.field.april_tags:
            size = self._apriltag_size(tag.variant)
            scale = self._glm_scale(0.01, size, size)
            tag_model = wpilib_glm * self._to_glm(
                pose_matrix(Pose3d(tag.position, tuple(rotation_sequence_to_quat(tag.rotations))))
            )
            self._apriltag_drawables.append(
                AprilTagDrawable(model_matrix=tag_model, scale_matrix=scale, tag_id=tag.tag_id)
            )

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

    def _robot_pose_matrix_from_pose(self, pose2d) -> np.ndarray:
        x, y, theta = self._convert_pose2d(pose2d.x, pose2d.y, pose2d.theta)
        base_pose = pose2d_matrix(x, y, theta)
        return self.wpilib_matrix @ base_pose @ self.robot_base_matrix

    def _component_matrices_from_pose(self, pose2d) -> list[np.ndarray]:
        x, y, theta = self._convert_pose2d(pose2d.x, pose2d.y, pose2d.theta)
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
            view_positions.append((float(view[0]), float(view[1]), float(view[2]), 1.0))
        updated = np.array(view_positions, dtype=np.float32)
        updated_bytes = updated.tobytes()
        if updated_bytes != self._light_positions_view_bytes:
            self._light_positions_view = updated
            self._light_positions_view_bytes = updated_bytes
            self._light_state_version += 1

    def _get_program_cache(self, program: moderngl.Program) -> ProgramUniformCache:
        program_id = id(program)
        cache = self._program_uniforms.get(program_id)
        if cache is not None:
            return cache

        def _get_uniform(name: str) -> Any | None:
            try:
                return program[name]
            except KeyError:
                return None

        cache = ProgramUniformCache(
            program=program,
            light_pos=_get_uniform("u_light_pos"),
            light_color=_get_uniform("u_light_color"),
            specular_strength=_get_uniform("u_specular_strength"),
            shininess=_get_uniform("u_shininess"),
        )
        self._program_uniforms[program_id] = cache
        return cache

    def _apply_lights_to_program(self, program: moderngl.Program | None) -> None:
        if program is None:
            return
        cache = self._get_program_cache(program)
        cache.update_lights(
            self._light_positions_view_bytes,
            self._light_colors_bytes,
            self._specular_strength,
            self._shininess,
            self._light_state_version,
        )

    def _ensure_fuel_drawables(self, instance_count: int) -> None:
        if instance_count <= 0 or not self._fuel_templates:
            return
        template_count = len(self._fuel_templates)
        needed = instance_count * template_count
        while len(self._fuel_drawables) < needed:
            template = self._fuel_templates[len(self._fuel_drawables) % template_count]
            renderable = Renderable(
                mesh=template.mesh,
                program=template.program,
                base_matrix=template.base_matrix,
                model_matrix=template.base_matrix,
                is_transparent=template.is_transparent,
                active=False,
            )
            self._fuel_drawables.append(renderable)
            self._register_renderable(renderable, is_dynamic=True)
        self._fuel_active_count = needed

    def _update_dynamic_transforms(self, fuel_positions: list[tuple[float, float, float]]) -> None:
        pose2d = self.nt_client.get_pose2d()
        self.last_robot_pose = self._robot_pose_matrix_from_pose(pose2d)
        robot_root = self._to_glm(self.last_robot_pose)
        for renderable in self._robot_drawables:
            renderable.model_matrix = glm.mat4(robot_root * renderable.base_matrix)

        component_matrices = self._component_matrices_from_pose(pose2d)
        for index, component_drawables in enumerate(self._component_drawables):
            if index < len(component_matrices):
                component_root = self._to_glm(component_matrices[index])
            else:
                component_root = glm.mat4(1.0)
            for renderable in component_drawables:
                renderable.model_matrix = glm.mat4(component_root * renderable.base_matrix)

        instance_count = len(fuel_positions)
        if instance_count == 0 or not self._fuel_templates:
            for renderable in self._fuel_drawables:
                renderable.active = False
            self._fuel_active_count = 0
            return

        self._ensure_fuel_drawables(instance_count)
        template_count = len(self._fuel_templates)
        active_needed = instance_count * template_count
        for index, renderable in enumerate(self._fuel_drawables):
            renderable.active = index < active_needed

        for index, position in enumerate(fuel_positions):
            converted = self._convert_translation(position[0], position[1], position[2])
            fuel_matrix = self.wpilib_matrix @ pose_matrix(Pose3d(converted, (1.0, 0.0, 0.0, 0.0)))
            fuel_root = self._to_glm(fuel_matrix)
            base_index = index * template_count
            for template_index in range(template_count):
                renderable = self._fuel_drawables[base_index + template_index]
                renderable.model_matrix = glm.mat4(
                    fuel_root * self._fuel_templates[template_index].base_matrix
                )

    def _render_mesh_drawables(self, renderables: list[Renderable], projection: glm.mat4, camera: glm.mat4) -> None:
        current_program_id: int | None = None
        for renderable in renderables:
            if not renderable.active:
                continue
            program = renderable.program
            if program is not None:
                program_id = id(program)
                if program_id != current_program_id:
                    current_program_id = program_id
                    self._apply_lights_to_program(program)
            renderable.mesh.draw(
                projection_matrix=projection,
                model_matrix=renderable.model_matrix,
                camera_matrix=camera,
            )

    def _render_apriltag_drawables(self, projection: glm.mat4, camera: glm.mat4) -> None:
        if not self._apriltag_drawables:
            return
        for tag_drawable in self._apriltag_drawables:
            mvp = projection * camera * tag_drawable.model_matrix * tag_drawable.scale_matrix
            self.tag_renderer.render(self._glm_to_bytes(glm.mat4(mvp)), tag_drawable.tag_id)

    def _render_cached_drawables(self) -> None:
        projection = self._camera_projection()
        camera = self._camera_view()
        self._update_light_positions_view(camera)

        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True
        self._render_mesh_drawables(self._opaque_drawables, projection, camera)

        if self._transparent_drawables or self._apriltag_drawables:
            self._camera_pos = glm.vec3(glm.inverse(camera)[3])
            active_count = sum(1 for renderable in self._transparent_drawables if renderable.active)
            if active_count > 1:
                self._transparent_drawables.sort(key=self._distance_sq_from_renderable, reverse=True)

            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.ctx.depth_mask = False
            if self._transparent_drawables:
                self._render_mesh_drawables(self._transparent_drawables, projection, camera)
            self._render_apriltag_drawables(projection, camera)
            self.ctx.depth_mask = True
            self.ctx.disable(moderngl.BLEND)

    def _distance_sq_from_renderable(self, renderable: Renderable) -> float:
        world_pos = glm.vec3(renderable.model_matrix[3])
        diff = world_pos - self._camera_pos
        return float(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z)

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

        self._update_dynamic_transforms(fuel_positions)
        self._render_cached_drawables()
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
