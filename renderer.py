from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2
from pathlib import Path
import threading
import requests
import io
import logging
from typing import Any, Callable
import copy

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

from wpimath.geometry import Pose3d, Rotation3d

from config import Rotation, load_assets_config
from math_utils import (
    compose,
    rotation_sequence_to_quat,
)
from nt_client import NetworkTablesClient


ASSETS_ROOT = Path(__file__).resolve().parent / "assets"
TEXTURE_ROOT = Path(__file__).resolve().parent / "textures"

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

# === LOAD CONSTANTS FROM CONFIG.JSON ===
import json


import re

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
if not CONFIG_PATH.exists():
    raise RuntimeError(f"Missing config.json at {CONFIG_PATH}")


def _strip_json_comments(text):
    # Remove // comments
    text = re.sub(r"//.*", "", text)
    # Remove /* ... */ comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


with open(CONFIG_PATH, encoding="utf-8") as f:
    _config_data = json.loads(_strip_json_comments(f.read()))


def _get_required_config(key):
    if key not in _config_data:
        raise RuntimeError(f"Missing required config key: '{key}' in config.json")
    return _config_data[key]


TEAM_NUMBER_POOL = _get_required_config("team-pool")
THIS_TEAM = _get_required_config("team")
MATCH_TYPE = _get_required_config("match-type")
MATCH_NUMBER = _get_required_config("match-number")
MATCH_TOTAL = _get_required_config("match-total")
GAME_YEAR = _get_required_config("year")

import random


class TeamAssignmentManager:
    def __init__(self, nt_client):
        self.nt_client = nt_client
        self.last_alliance_station = None
        self.red_team_numbers = [None, None, None]
        self.blue_team_numbers = [None, None, None]
        self._update_team_numbers()

    def _update_team_numbers(self):
        alliance_station = self.nt_client.get_alliance_station()
        if(alliance_station < 1 or alliance_station > 6):
            alliance_station = 1  # Default to red1 if invalid
            
        if alliance_station == self.last_alliance_station:
            return
        self.last_alliance_station = alliance_station
        # Alliance station mapping: 1=red1, 2=red2, 3=red3, 4=blue1, 5=blue2, 6=blue3
        # Indices: 0=first, 1=second, 2=third
        red = [None, None, None]
        blue = [None, None, None]
        pool = TEAM_NUMBER_POOL.copy()
        # Place THIS_TEAM
        if 1 <= alliance_station <= 3:
            idx = alliance_station - 1
            red[idx] = THIS_TEAM
        elif 4 <= alliance_station <= 6:
            idx = alliance_station - 4
            blue[idx] = THIS_TEAM
        # Fill remaining spots from pool, seeded by alliance_station
        rng = random.Random(alliance_station)
        pool = [t for t in pool if t != THIS_TEAM]
        rng.shuffle(pool)
        pool_idx = 0
        for i in range(3):
            if red[i] is None:
                red[i] = pool[pool_idx]
                pool_idx += 1
        for i in range(3):
            if blue[i] is None:
                blue[i] = pool[pool_idx]
                pool_idx += 1
        self.red_team_numbers = [str(n) for n in red]
        self.blue_team_numbers = [str(n) for n in blue]

    def get_red_team_numbers(self):
        self._update_team_numbers()
        return self.red_team_numbers

    def get_blue_team_numbers(self):
        self._update_team_numbers()
        return self.blue_team_numbers


# === TEAM LOGO CONSTANTS ===
LOGO_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / f"logos_{GAME_YEAR}"
LOGO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class TeamLogoCache:
    def __init__(self):
        self._mem_cache = (
            {}
        )  # (team_number: int) -> PIL.Image or None (None=loading/failed)
        self._lock = threading.Lock()

    def get_logo(self, team_number: str, callback=None):
        # Remove leading zeros, ensure int
        try:
            team = int(team_number)
        except Exception as e:
            logging.error(f"Invalid team number '{team_number}': {e}")
            return None
        # Check memory cache
        with self._lock:
            if team in self._mem_cache:
                return self._mem_cache[team]
            # Mark as loading
            self._mem_cache[team] = None
            logging.debug(
                f"Logo for team {team} not in memory cache, starting async load."
            )
        # Start async load
        threading.Thread(
            target=self._load_logo, args=(team, callback), daemon=True
        ).start()
        return None

    def _load_logo(self, team, callback):
        # Disk cache path
        cache_path = LOGO_CACHE_DIR / f"{team}.png"
        img = None
        if cache_path.exists():
            try:
                img = Image.open(cache_path).convert("RGBA")
                logging.info(
                    f"Loaded logo for team {team} from disk cache: {cache_path}"
                )
            except Exception as e:
                logging.error(f"Failed to load logo for team {team} from disk: {e}")
                img = None
        else:
            url = f"https://www.thebluealliance.com/avatar/{GAME_YEAR}/frc{team}.png"
            try:
                logging.info(f"Downloading logo for team {team} from {url}")
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
                    img.save(cache_path)
                    logging.info(
                        f"Downloaded and cached logo for team {team} at {cache_path}"
                    )
                elif resp.status_code == 403:
                    # Forbidden: use placeholder
                    logging.warning(
                        f"Logo fetch forbidden for team {team} in {GAME_YEAR}, using placeholder."
                    )
                    import shutil

                    placeholder_path = (
                        Path(__file__).resolve().parent
                        / "logos"
                        / "FIRSTicon_RGB_withTM.png"
                    )
                    try:
                        shutil.copy(placeholder_path, cache_path)
                        img = Image.open(cache_path).convert("RGBA")
                        logging.info(
                            f"Copied placeholder logo for team {team} to {cache_path}"
                        )
                    except Exception as e:
                        logging.error(
                            f"Failed to copy placeholder logo for team {team}: {e}"
                        )
                        img = None
                else:
                    logging.warning(
                        f"Failed to download logo for team {team}: HTTP {resp.status_code}"
                    )
            except Exception as e:
                logging.error(f"Exception downloading logo for team {team}: {e}")
                img = None
        with self._lock:
            self._mem_cache[team] = img
            logging.debug(f"Logo for team {team} set in memory cache: {img}")
        if callback:
            callback(team, img)


team_logo_cache = TeamLogoCache()

DS_CAMERA_HEIGHT = 62 * 0.0254
DS_CAMERA_OFFSET_FRC = 1.5

ORBIT_FIELD_FRC_DEFAULT_TARGET = np.array(
    [0.0, 1.32, 0.0], dtype=np.float32
)  # Center of field
ORBIT_FIELD_FRC_DEFAULT_RADIUS = 11
ORBIT_ROBOT_FRC_DEFAULT_TARGET = np.array([0.0, 0.5, 0.0], dtype=np.float32)
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
        shader_dir = Path(__file__).resolve().parent / "shaders"
        shader_dir_str = str(shader_dir)
        settings.PROGRAM_DIRS = [shader_dir_str]
        self.assets = load_assets_config(ASSETS_ROOT)
        self.nt_client = NetworkTablesClient()
        self.team_assignment = TeamAssignmentManager(self.nt_client)
        imgui.create_context()
        self.imgui = ImGuiBridge(self.wnd, self.ctx)

        io = imgui.get_io()

        self._ui_font_base_size = 128
        self._ui_font_target_size = 32
        self._ui_font_small_size = int(self._ui_font_base_size / 1.5)
        self._ui_font_big_size = int(self._ui_font_base_size * 1.5)
        self._ui_font_large_size = int(self._ui_font_base_size * 1.4)
        self._ui_font = io.fonts.add_font_from_file_ttf(
            "fonts/Roboto-Bold.ttf",
            self._ui_font_base_size,
        )
        self._ui_font_small = io.fonts.add_font_from_file_ttf(
            "fonts/Roboto-Bold.ttf",
            self._ui_font_small_size,
        )
        self._ui_font_big = io.fonts.add_font_from_file_ttf(
            "fonts/Roboto-Bold.ttf",
            self._ui_font_big_size,
        )
        self._ui_font_large = io.fonts.add_font_from_file_ttf(
            "fonts/Roboto-Bold.ttf",
            self._ui_font_large_size,
        )
        self.imgui.renderer.refresh_font_texture()

        self.wnd.swap_interval = 0
        self._field_length_m = self.assets.field.width_inches * 0.0254
        self._field_width_m = self.assets.field.height_inches * 0.0254
        self._coordinate_system = self.assets.field.coordinate_system

        # Bumper foam dynamic color state
        self._foam_mesh = None
        self._foam_material = None
        self._foam_original_color = None
        self.camera = OrbitCamera(
            target=(
                float(ORBIT_FIELD_FRC_DEFAULT_TARGET[0]),
                float(ORBIT_FIELD_FRC_DEFAULT_TARGET[1]),
                float(ORBIT_FIELD_FRC_DEFAULT_TARGET[2]),
            ),
            angles=(-90, -57.2),
            radius=ORBIT_FIELD_FRC_DEFAULT_RADIUS,
        )
        self.camera.projection.update(
            aspect_ratio=self.wnd.aspect_ratio, fov=50.0, near=0.1, far=100.0
        )
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
        self.wpilib_matrix = Pose3d(0.0, 0.0, 0.0, Rotation3d(WPILIB_ROTATION)).toMatrix()
        self.field_base_matrix = compose(
            self.wpilib_matrix,
            Pose3d(
                self.assets.field.position[0],
                self.assets.field.position[1],
                self.assets.field.position[2],
                Rotation3d(rotation_sequence_to_quat(self.assets.field.rotations)),
            ).toMatrix(),
        )
        self.robot_base_matrix = Pose3d(
                self.assets.robot.position[0],
                self.assets.robot.position[1],
                self.assets.robot.position[2],
                Rotation3d(rotation_sequence_to_quat(self.assets.robot.rotations))
            ).toMatrix()
        
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
                    np.array(
                        self._convert_translation(pos[0], pos[1], pos[2]),
                        dtype=np.float32,
                    ),
                )
                for pos in light_positions_wpilib
            ],
            dtype=np.float32,
        )
        self._light_positions_view = np.column_stack(
            (
                self._light_positions_world,
                np.ones(len(self._light_positions_world), dtype=np.float32),
            )
        )
        self._light_program_cache: set[int] = set()
        self._staged_fuel_nodes: dict[Any, Any] = {}
        self._hub_diffusers: dict[int, list[Any]] = {1: [], 2: []}
        self._hub_diffuser_original_colors: dict[int, dict[Any, tuple[float, ...]]] = {
            1: {},
            2: {},
        }
        self._hub_glow_colors = {
            1: (1.0, 0.18, 0.18, 1.0),
            2: (0.2, 0.45, 1.0, 1.0),
        }
        self._hub_mesh_owner: dict[int, int] = {}
        self._bloom_enabled = True
        self._bloom_threshold = 2.3
        self._bloom_intensity = 1.0
        self._bloom_blur_passes = 6
        self._bloom_size: tuple[int, int] | None = None
        self._init_bloom_resources()

    def _to_glm(self, matrix: np.ndarray) -> glm.mat4:
        return glm.mat4(*matrix.flatten(order="F").tolist())

    def _init_bloom_resources(self) -> None:
        quad_vertices = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype="f4",
        )
        self._quad_vbo = self.ctx.buffer(quad_vertices.tobytes())
        self._bloom_extract_program = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            in vec2 in_uv;
            out vec2 v_uv;
            void main() {
                v_uv = in_uv;
                gl_Position = vec4(in_pos, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            uniform sampler2D scene_tex;
            uniform float threshold;
            in vec2 v_uv;
            out vec4 f_color;
            void main() {
                vec3 color = texture(scene_tex, v_uv).rgb;
                float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
                f_color = luma > threshold ? vec4(color, 1.0) : vec4(0.0);
            }
            """,
        )
        self._bloom_blur_program = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            in vec2 in_uv;
            out vec2 v_uv;
            void main() {
                v_uv = in_uv;
                gl_Position = vec4(in_pos, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            uniform sampler2D image;
            uniform vec2 direction;
            in vec2 v_uv;
            out vec4 f_color;
            void main() {
                vec2 texel = 1.0 / vec2(textureSize(image, 0));
                float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
                vec3 result = texture(image, v_uv).rgb * weights[0];
                for (int i = 1; i < 5; i++) {
                    vec2 offset = direction * texel * float(i);
                    result += texture(image, v_uv + offset).rgb * weights[i];
                    result += texture(image, v_uv - offset).rgb * weights[i];
                }
                f_color = vec4(result, 1.0);
            }
            """,
        )
        self._bloom_combine_program = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            in vec2 in_uv;
            out vec2 v_uv;
            void main() {
                v_uv = in_uv;
                gl_Position = vec4(in_pos, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            uniform sampler2D scene_tex;
            uniform sampler2D bloom_tex;
            uniform float intensity;
            in vec2 v_uv;
            out vec4 f_color;
            vec3 tonemap_aces(vec3 x) {
                const float a = 2.51;
                const float b = 0.03;
                const float c = 2.43;
                const float d = 0.59;
                const float e = 0.14;
                return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
            }
            void main() {
                vec3 scene = texture(scene_tex, v_uv).rgb;
                vec3 bloom = texture(bloom_tex, v_uv).rgb;
                vec3 color = scene + bloom * intensity;
                color = tonemap_aces(color);
                f_color = vec4(color, 1.0);
            }
            """,
        )
        self._bloom_extract_vao = self.ctx.vertex_array(
            self._bloom_extract_program, [(self._quad_vbo, "2f 2f", "in_pos", "in_uv")]
        )
        self._bloom_blur_vao = self.ctx.vertex_array(
            self._bloom_blur_program, [(self._quad_vbo, "2f 2f", "in_pos", "in_uv")]
        )
        self._bloom_combine_vao = self.ctx.vertex_array(
            self._bloom_combine_program, [(self._quad_vbo, "2f 2f", "in_pos", "in_uv")]
        )
        self._init_bloom_targets()

    def _init_bloom_targets(self) -> None:
        width, height = self.wnd.buffer_size
        width = max(1, int(width))
        height = max(1, int(height))
        self._bloom_size = (width, height)
        bloom_width = max(1, width // 4)
        bloom_height = max(1, height // 4)
        self._scene_color_tex = self.ctx.texture((width, height), 4, dtype="f2")
        self._scene_color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._scene_depth = self.ctx.depth_renderbuffer((width, height))
        self._scene_fbo = self.ctx.framebuffer(self._scene_color_tex, self._scene_depth)
        self._bloom_extract_tex = self.ctx.texture(
            (bloom_width, bloom_height), 4, dtype="f2"
        )
        self._bloom_extract_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._bloom_extract_fbo = self.ctx.framebuffer(self._bloom_extract_tex)
        self._bloom_pingpong_tex = [
            self.ctx.texture((bloom_width, bloom_height), 4, dtype="f2"),
            self.ctx.texture((bloom_width, bloom_height), 4, dtype="f2"),
        ]
        for tex in self._bloom_pingpong_tex:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._bloom_pingpong_fbo = [
            self.ctx.framebuffer(self._bloom_pingpong_tex[0]),
            self.ctx.framebuffer(self._bloom_pingpong_tex[1]),
        ]

    def _ensure_bloom_targets(self) -> None:
        width, height = self.wnd.buffer_size
        size = (int(width), int(height))
        if self._bloom_size != size:
            self._init_bloom_targets()

    def _normalize_mesh_name(self, name: str) -> str:
        return name.replace(":", "").replace(" ", "_").replace("__", "_").lower()

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
        if self.field_scene is not None:
            self._collect_hub_diffusers()

        # Find foam mesh/material in robot_scene
        self._foam_mesh = None
        self._foam_material = None
        self._foam_original_color = None
        if self.robot_scene is not None:
            for node in getattr(self.robot_scene, "nodes", []):
                if node.mesh is not None and node.name and node.name.lower() == "foam":
                    self._foam_mesh = node.mesh
                    self._foam_material = getattr(self._foam_mesh, "material", None)
                    if self._foam_material is not None and hasattr(
                        self._foam_material, "color"
                    ):
                        # Store original color as tuple
                        self._foam_original_color = tuple(self._foam_material.color)
                    break
        self._scenes_loaded = True

    def _update_bumper_foam_color(self):
        """Set foam color to red for alliance station 1-3, else restore original blue."""
        if self._foam_material is None or self._foam_original_color is None:
            return
        alliance_station = self.nt_client.get_alliance_station()
        if 1 <= alliance_station <= 3:
            # Set to red #FF0707, RGBA
            self._foam_material.color = (1.0, 0.027, 0.027, 1.0)
        else:
            # Restore original blue, ensure 4-tuple
            orig = self._foam_original_color
            if len(orig) == 3:
                self._foam_material.color = (orig[0], orig[1], orig[2], 1.0)
            else:
                self._foam_material.color = orig

    def _collect_hub_diffusers(self) -> None:
        hub_names = {
            1: "GE-26300: Hub <1>",
            2: "GE-26300: Hub <2>",
        }
        diffuser_names = {
            "GE-26309: Hub Front Diffuser",
            "GE-26310: Hub Rear Diffuser",
            "GE-26311: Hub Side Diffuser",
        }
        self._hub_diffusers = {1: [], 2: []}
        self._hub_diffuser_original_colors = {1: {}, 2: {}}
        self._hub_mesh_owner = {}
        for node in self.field_scene.root_nodes:
            self._walk_hub_tree(node, hub_names, diffuser_names, None)

    def _walk_hub_tree(
        self,
        node,
        hub_names: dict[int, str],
        diffuser_names: set[str],
        active_hub: int | None,
    ) -> None:
        node_name = getattr(node, "name", None)
        for hub_index, hub_name in hub_names.items():
            if node_name == hub_name:
                active_hub = hub_index
                break
        if active_hub is not None and node_name in diffuser_names:
            mesh = getattr(node, "mesh", None)
            if (
                mesh is not None
                and mesh not in self._hub_diffuser_original_colors[active_hub]
            ):
                mesh = self._ensure_unique_hub_mesh(mesh, node, active_hub)
                material = getattr(mesh, "material", None)
                color = getattr(material, "color", None) if material else None
                if color is not None:
                    self._hub_diffusers[active_hub].append(mesh)
                    self._hub_diffuser_original_colors[active_hub][mesh] = tuple(color)
        for child in getattr(node, "children", []):
            self._walk_hub_tree(child, hub_names, diffuser_names, active_hub)

    def _ensure_unique_hub_mesh(self, mesh, node, hub_index: int):
        owner = self._hub_mesh_owner.get(id(mesh))
        if owner is not None and owner != hub_index:
            if hasattr(mesh, "copy"):
                mesh = mesh.copy()
            else:
                mesh = copy.copy(mesh)
            node.mesh = mesh
        self._hub_mesh_owner[id(mesh)] = hub_index
        self._ensure_unique_material(mesh, hub_index)
        return mesh

    def _ensure_unique_material(self, mesh, hub_index: int) -> None:
        material = getattr(mesh, "material", None)
        if material is None:
            return
        material_owner = self._hub_mesh_owner.get(id(material))
        if material_owner is not None and material_owner != hub_index:
            if hasattr(material, "copy"):
                material = material.copy()
            else:
                material = copy.copy(material)
            mesh.material = material
        self._hub_mesh_owner[id(material)] = hub_index

    def _camera_from_driver_station(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        stations = self.assets.field.driver_stations
        if index < 0 or index >= len(stations):
            return ORBIT_FIELD_FRC_DEFAULT_POSITION, ORBIT_FIELD_FRC_DEFAULT_TARGET
        pos_x, pos_y = stations[index]
        heading = atan2(-pos_y, -pos_x)
        forward = np.array([np.cos(heading), np.sin(heading), 0.0], dtype=np.float32)
        camera_pos = (
            np.array([pos_x, pos_y, 0.0], dtype=np.float32)
            + forward * -DS_CAMERA_OFFSET_FRC
        )
        camera_pos[2] = DS_CAMERA_HEIGHT
        target = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        camera_pos = self._transform_point(self.wpilib_matrix, camera_pos)
        target = self._transform_point(self.wpilib_matrix, target)
        return camera_pos, target

    def _convert_pose3d(self, pose3d: Pose3d) -> Pose3d:
        if self._coordinate_system == "wall-blue":
            return Pose3d(
                self._field_length_m / 2.0 - pose3d.X(),
                self._field_width_m / 2.0 - pose3d.Y(),
                pose3d.Z(),
                pose3d.rotation().rotateBy(Rotation3d(0,0,np.pi)),
            )
        if self._coordinate_system == "center-rotated":
            return Pose3d(pose3d.Y(), -pose3d.X(), pose3d.Z(), pose3d.rotation().rotateBy(Rotation3d(0,0,-np.pi/2)))
        if self._coordinate_system == "wall-alliance":
            return Pose3d(
                self._field_length_m / 2.0 - pose3d.X(),
                self._field_width_m / 2.0 - pose3d.Y(),
                pose3d.Z(),
                pose3d.rotation().rotateBy(Rotation3d(0,0,np.pi)),
            )
        return pose3d

    def _convert_translation(
        self, x: float, y: float, z: float
    ) -> tuple[float, float, float]:
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
                field_target = self._transform_point(
                    self.wpilib_matrix, ORBIT_FIELD_FRC_DEFAULT_TARGET
                )
                self.camera.target = glm.vec3(
                    float(field_target[0]),
                    float(field_target[1]),
                    float(field_target[2]),
                )
                self.camera.radius = ORBIT_FIELD_FRC_DEFAULT_RADIUS
                self.camera.angle_x = -90
                self.camera.angle_y = -57.2
            elif self.view_mode == ViewMode.ORBIT_ROBOT:
                robot_pos = np.array(self.last_robot_pose[:3, 3], dtype=np.float32)
                robot_offset = (
                    ORBIT_ROBOT_FRC_DEFAULT_POSITION - ORBIT_ROBOT_FRC_DEFAULT_TARGET
                )
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
                self.camera.target = glm.vec3(
                    float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])
                )
            elif self.view_mode not in {ViewMode.ORBIT_FIELD, ViewMode.ORBIT_ROBOT}:
                index = list(ViewMode).index(self.view_mode) - 2
                camera_pos, target = self._camera_from_driver_station(index)
                self._set_orbit_from_position(camera_pos, target)

    def _set_orbit_from_position(
        self, position: np.ndarray, target: np.ndarray
    ) -> None:
        vector = np.array(position, dtype=np.float32) - np.array(
            target, dtype=np.float32
        )
        radius = float(np.linalg.norm(vector))
        if radius <= 0:
            return
        angle_y = -float(np.degrees(np.arccos(np.clip(vector[1] / radius, -1.0, 1.0))))
        sin_y = np.sin(np.radians(angle_y))
        angle_x = (
            90.0
            if abs(sin_y) < 1e-6
            else float(
                np.degrees(np.arcsin(np.clip(vector[2] / (radius * sin_y), -1.0, 1.0)))
            )
        )
        self.camera.target = glm.vec3(
            float(target[0]), float(target[1]), float(target[2])
        )
        self.camera.radius = radius
        self.camera.angle_x = angle_x
        self.camera.angle_y = angle_y

    def _transform_point(self, matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
        vec = np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
        result = matrix @ vec
        return result[:3]

    def _robot_pose_matrix(self) -> np.ndarray:
        pose3d = self.nt_client.get_pose3d()
        pose3d = self._convert_pose3d(pose3d)
        base_pose = pose3d.toMatrix()
        return self.wpilib_matrix @ base_pose @ self.robot_base_matrix

    def _component_matrices(self) -> list[np.ndarray]:
        pose3d = self.nt_client.get_pose3d()
        pose3d = self._convert_pose3d(pose3d)
        robot_pose_matrix = pose3d.toMatrix()
        component_poses = self.nt_client.get_mechanism_poses()
        matrices: list[np.ndarray] = []
        for index, component in enumerate(self.assets.robot.components):
            if index < len(component_poses):
                pose = component_poses[index]
                user_pose = pose.toMatrix()
            else:
                user_pose = np.eye(4, dtype=np.float32)
            config_pose = Pose3d(
                    component.zeroed_position[0],
                    component.zeroed_position[1],
                    component.zeroed_position[2],
                    Rotation3d(rotation_sequence_to_quat(component.zeroed_rotations)),
                ).toMatrix()
            
            matrices.append(
                self.wpilib_matrix @ robot_pose_matrix @ user_pose @ config_pose
            )
        return matrices

    def _fuel_instances(
        self, positions: list[Pose3d]
    ) -> list[FuelInstance]:
        fuels = []
        for position in positions:
            converted = self._convert_translation(position.X(), position.Y(), position.Z())
            matrix = self.wpilib_matrix @ (
                Pose3d(converted[0], converted[1], converted[2], Rotation3d(0, 0, 0))
            ).toMatrix()
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
            view_positions.append((float(view[0]), float(view[1]), float(view[2]), 1.0))
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

            def _draw_fn(
                projection: glm.mat4, model: glm.mat4, camera: glm.mat4
            ) -> None:
                self._apply_lights_to_mesh(mesh)
                mesh.draw(
                    projection_matrix=projection,
                    model_matrix=model,
                    camera_matrix=camera,
                )

            drawables.append(
                (_draw_fn, node.matrix_global, self._mesh_requires_blend(mesh))
            )
        for child in getattr(node, "children", []):
            self._collect_drawables(child, drawables)

    def _collect_scene_drawables(
        self,
        scene: mglw.scene.Scene,
        matrix: np.ndarray,
        drawables: list[tuple[Callable[..., None], Any, bool]],
    ) -> None:
        scene.matrix = self._to_glm(matrix)
        for node in scene.root_nodes:
            self._collect_drawables(node, drawables)

    def _collect_apriltag_drawables(
        self, drawables: list[tuple[Callable[..., None], Any, bool]]
    ) -> None:
        wpilib_glm = self._to_glm(self.wpilib_matrix)
        for tag in self.assets.field.april_tags:
            size = self._apriltag_size(tag.variant)
            scale = self._glm_scale(0.01, size, size)
            tag_model = wpilib_glm * self._to_glm(
                (
                    Pose3d(
                        tag.position[0],
                        tag.position[1],
                        tag.position[2],
                        Rotation3d(rotation_sequence_to_quat(tag.rotations))
                    )
                ).toMatrix()
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

    def _render_drawables(
        self, drawables: list[tuple[Callable[..., None], Any, bool]]
    ) -> None:
        if not drawables:
            return
        projection = self._camera_projection()
        camera = self._camera_view()
        self._update_light_positions_view(camera)
        self._light_program_cache.clear()
        opaque = [
            (draw_fn, model)
            for draw_fn, model, is_transparent in drawables
            if not is_transparent
        ]
        transparent = [
            (draw_fn, model)
            for draw_fn, model, is_transparent in drawables
            if is_transparent
        ]

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
        self._update_camera()
        self._apply_keyboard_controls(frame_time)

        self._load_scenes()
        self._update_bumper_foam_color()
        if not self._scenes_loaded:
            return

        fuel_positions = self.nt_client.get_fuel_positions()
        self._set_staged_fuel_visible(len(fuel_positions) == 0)

        if (
            self.field_scene is None
            or self.robot_scene is None
            or self.fuel_scene is None
        ):
            return

        drawables: list[tuple[Callable[..., None], Any, bool]] = []
        self._collect_scene_drawables(
            self.field_scene, self.field_base_matrix, drawables
        )
        self.last_robot_pose = self._robot_pose_matrix()
        self._collect_scene_drawables(self.robot_scene, self.last_robot_pose, drawables)

        for component_scene, matrix in zip(
            self.robot_components, self._component_matrices()
        ):
            self._collect_scene_drawables(component_scene, matrix, drawables)

        for fuel in self._fuel_instances(fuel_positions):
            self._collect_scene_drawables(self.fuel_scene, fuel.matrix, drawables)

        self._collect_apriltag_drawables(drawables)

        if self._bloom_enabled:
            self._ensure_bloom_targets()
            self._render_scene_to_fbo(drawables)
            self._apply_bloom()
            self._render_ui()
        else:
            self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
            self.ctx.clear(0.18, 0.18, 0.2)
            self._render_drawables(drawables)
            self._render_ui()
        self._update_hub_diffusers()

    def _render_scene_to_fbo(
        self, drawables: list[tuple[Callable[..., None], Any, bool]]
    ) -> None:
        self._scene_fbo.use()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.clear(0.18, 0.18, 0.2)
        self._render_drawables(drawables)

    def _apply_bloom(self) -> None:
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self._bloom_extract_fbo.use()
        self._bloom_extract_program["scene_tex"].value = 0
        self._bloom_extract_program["threshold"].value = float(self._bloom_threshold)
        self._scene_color_tex.use(location=0)
        self._bloom_extract_vao.render(moderngl.TRIANGLE_STRIP)

        horizontal = True
        src_tex = self._bloom_extract_tex
        for _ in range(self._bloom_blur_passes):
            fbo_index = 0 if horizontal else 1
            self._bloom_pingpong_fbo[fbo_index].use()
            self._bloom_blur_program["image"].value = 0
            self._bloom_blur_program["direction"].value = (
                (
                    1.0,
                    0.0,
                )
                if horizontal
                else (
                    0.0,
                    1.0,
                )
            )
            src_tex.use(location=0)
            self._bloom_blur_vao.render(moderngl.TRIANGLE_STRIP)
            src_tex = self._bloom_pingpong_tex[fbo_index]
            horizontal = not horizontal

        self.wnd.use()
        self.ctx.clear(0.0, 0.0, 0.0)
        self._bloom_combine_program["scene_tex"].value = 0
        self._bloom_combine_program["bloom_tex"].value = 1
        self._bloom_combine_program["intensity"].value = float(self._bloom_intensity)
        self._scene_color_tex.use(location=0)
        src_tex.use(location=1)
        self._bloom_combine_vao.render(moderngl.TRIANGLE_STRIP)

    def _update_hub_diffusers(self) -> None:
        red_active = self.nt_client.get_red_hub_active()
        blue_active = self.nt_client.get_blue_hub_active()
        self._apply_hub_diffuser_state(1, red_active)
        self._apply_hub_diffuser_state(2, blue_active)

    def _apply_hub_diffuser_state(self, hub_index: int, active: bool) -> None:
        target_color = self._hub_glow_colors.get(hub_index)
        original_colors = self._hub_diffuser_original_colors.get(hub_index, {})
        for mesh in self._hub_diffusers.get(hub_index, []):
            material = getattr(mesh, "material", None)
            if material is None:
                continue
            if active and target_color is not None:
                boost = 17.0
                material.color = (
                    target_color[0] * boost,
                    target_color[1] * boost,
                    target_color[2] * boost,
                    target_color[3],
                )
            else:
                original = original_colors.get(mesh)
                if original is not None:
                    material.color = original

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
                * self._to_glm(
                    (
                        Pose3d(
                            tag.position[0],
                            tag.position[1],
                            tag.position[2],
                            Rotation3d(rotation_sequence_to_quat(tag.rotations)),
                        )
                    ).toMatrix()
                )
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

        if self._ui_font is not None:
            imgui.push_font(self._ui_font)

        io = imgui.get_io()
        width, height = io.display_size

        # === UNIFORM SCALE FACTOR (based on 1920x1080 reference) ===
        base_width = 1920.0
        base_height = 1080.0
        scale = min(width / base_width, height / base_height)
        offset_x = (width - base_width * scale) / 2.0
        offset_y = 0.0
        io.font_global_scale = scale * (
            self._ui_font_target_size / self._ui_font_base_size
        )

        def S(x, y):
            return offset_x + x * scale, offset_y + y * scale

        def rect(x, y, w, h, color):
            draw = imgui.get_background_draw_list()
            x, y = S(x, y)
            w, h = w * scale, h * scale
            draw.add_rect_filled(x, y, x + w, y + h, color, 0)

        def text(x, y, txt, size=20, color=(255, 255, 255, 255), center=False):
            draw = imgui.get_background_draw_list()
            x, y = S(x, y)

            if center:
                tw, th = imgui.calc_text_size(txt)
                x -= tw / 2
                y -= th / 2

            draw.add_text(
                x,
                y,
                imgui.get_color_u32_rgba(
                    color[0] / 255, color[1] / 255, color[2] / 255, color[3] / 255
                ),
                txt,
            )

        def text_small(x, y, txt, color=(255, 255, 255, 255), center=False):
            imgui.pop_font()
            imgui.push_font(self._ui_font_small)
            text(x, y, txt, size=20, color=color, center=center)
            imgui.pop_font()
            imgui.push_font(self._ui_font)

        def text_big(x, y, txt, color=(255, 255, 255, 255), center=False):
            imgui.pop_font()
            imgui.push_font(self._ui_font_big)
            text(x, y, txt, size=20, color=color, center=center)
            imgui.pop_font()
            imgui.push_font(self._ui_font)

        def text_large(x, y, txt, color=(255, 255, 255, 255), center=False):
            imgui.pop_font()
            imgui.push_font(self._ui_font_large)
            text(x, y, txt, size=20, color=color, center=center)
            imgui.pop_font()
            imgui.push_font(self._ui_font)

        # === COLORS ===
        BLUE = (30 / 255, 90 / 255, 200 / 255, 1)
        RED = (200 / 255, 40 / 255, 40 / 255, 1)
        DARK = (20 / 255, 20 / 255, 25 / 255, 1)
        WHITE = (1, 1, 1, 1)
        BLUE_TEAM_OUTER = (0 / 255, 64 / 255, 115 / 255, 1)
        BLUE_TEAM_CENTER = (0 / 255, 46 / 255, 84 / 255, 1)
        RED_TEAM_OUTER = (130 / 255, 12 / 255, 19 / 255, 1)
        RED_TEAM_CENTER = (97 / 255, 9 / 255, 12 / 255, 1)

        def col(c):
            return imgui.get_color_u32_rgba(*c)

        # =========================
        # CENTER TIMER PANEL (background first)
        # =========================
        rect(450, 15, 1020, 45, col(DARK))
        rect(450, 60, 1020, 70, col(WHITE))

        # =========================
        # LEFT BLUE PANEL
        # =========================
        rect(450, 60, 450, 70, col(BLUE))

        blue_score = int(self.nt_client.get_blue_score())

        # Team numbers panel (blue) 330x60, bottom-left of blue panel
        blue_team_x = 450
        blue_team_y = 70
        team_panel_w = 330
        team_panel_h = 60
        team_cell_w = team_panel_w / 3

        # Blue score: center in area to right of team numbers, same height, bottom aligned
        blue_score_x0 = blue_team_x + team_panel_w
        blue_score_x1 = 450 + 450  # end of blue panel
        blue_score_w = blue_score_x1 - blue_score_x0
        blue_score_h = team_panel_h
        blue_score_cx = blue_score_x0 + blue_score_w / 2
        blue_score_cy = blue_team_y + team_panel_h  # bottom edge
        # Center vertically in the height of the team panel, but bottom aligned
        text_big(
            blue_score_cx,
            blue_score_cy - blue_score_h / 2,
            f"{blue_score}",
            center=True,
        )

        blue_team_numbers = self.team_assignment.get_blue_team_numbers()
        for index in range(3):
            cell_x = blue_team_x + team_cell_w * index
            cell_color = BLUE_TEAM_CENTER if index == 1 else BLUE_TEAM_OUTER
            rect(cell_x, blue_team_y, team_cell_w, team_panel_h, col(cell_color))
            # --- Team logo ---
            logo_x = cell_x + 12
            logo_y = blue_team_y + 15
            logo_w = 30
            logo_h = 30
            team_num = blue_team_numbers[index]
            logo_img = team_logo_cache.get_logo(team_num)
            if logo_img is not None:
                try:
                    if not hasattr(logo_img, "_imgui_tex_id"):
                        logging.info(
                            f"Uploading logo for team {team_num} to GPU and registering with ImGui."
                        )
                        tex = self.ctx.texture(logo_img.size, 4, logo_img.tobytes())
                        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                        self.imgui.renderer.register_texture(tex)
                        logo_img._imgui_tex_id = tex.glo
                        logo_img._mgl_tex = tex  # Prevent GC
                        logging.info(
                            f"Registered logo for team {team_num} with ImGui tex_id={tex.glo}, tex={tex}"
                        )
                    imgui.get_background_draw_list().add_image(
                        logo_img._imgui_tex_id,
                        S(logo_x, logo_y),
                        S(logo_x + logo_w, logo_y + logo_h),
                    )
                except Exception as e:
                    logging.error(f"Failed to render logo for team {team_num}: {e}")
                    rect(logo_x, logo_y, logo_w, logo_h, col(WHITE))
            else:
                logging.debug(
                    f"Logo for team {team_num} not loaded yet, drawing placeholder."
                )
                rect(logo_x, logo_y, logo_w, logo_h, col(WHITE))
            text_small(
                cell_x + (team_cell_w + 30) / 2,
                blue_team_y + 30,
                team_num,
                center=True,
            )

        # =========================
        # RIGHT RED PANEL
        # =========================
        rect(1020, 60, 450, 70, col(RED))

        red_score = int(self.nt_client.get_red_score())

        # Team numbers panel (red) 330x60, bottom-right of red panel
        red_team_x = 1140
        red_team_y = 70

        # Red score: center in area to left of team numbers, same height, bottom aligned
        red_score_x0 = 1020  # start of red panel
        red_score_x1 = red_team_x  # start of team numbers
        red_score_w = red_score_x1 - red_score_x0
        red_score_h = team_panel_h
        red_score_cx = red_score_x0 + red_score_w / 2
        red_score_cy = red_team_y + team_panel_h  # bottom edge
        # Center vertically in the height of the team panel, but bottom aligned
        text_big(
            red_score_cx,
            red_score_cy - red_score_h / 2,
            f"{red_score}",
            center=True,
        )

        red_team_numbers = self.team_assignment.get_red_team_numbers()
        for index in range(3):
            cell_x = red_team_x + team_cell_w * index
            cell_color = RED_TEAM_CENTER if index == 1 else RED_TEAM_OUTER
            rect(cell_x, red_team_y, team_cell_w, team_panel_h, col(cell_color))
            # --- Team logo ---
            logo_x = cell_x + 12
            logo_y = red_team_y + 15
            logo_w = 30
            logo_h = 30
            team_num = red_team_numbers[index]
            logo_img = team_logo_cache.get_logo(team_num)
            if logo_img is not None:
                try:
                    if not hasattr(logo_img, "_imgui_tex_id"):
                        logging.info(
                            f"Uploading logo for team {team_num} to GPU and registering with ImGui."
                        )
                        tex = self.ctx.texture(logo_img.size, 4, logo_img.tobytes())
                        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                        self.imgui.renderer.register_texture(tex)
                        logo_img._imgui_tex_id = tex.glo
                        logo_img._mgl_tex = tex  # Prevent GC
                        logging.info(
                            f"Registered logo for team {team_num} with ImGui tex_id={tex.glo}, tex={tex}"
                        )
                    imgui.get_background_draw_list().add_image(
                        logo_img._imgui_tex_id,
                        S(logo_x, logo_y),
                        S(logo_x + logo_w, logo_y + logo_h),
                    )
                except Exception as e:
                    logging.error(f"Failed to render logo for team {team_num}: {e}")
                    rect(logo_x, logo_y, logo_w, logo_h, col(WHITE))
            else:
                logging.debug(
                    f"Logo for team {team_num} not loaded yet, drawing placeholder."
                )
                rect(logo_x, logo_y, logo_w, logo_h, col(WHITE))
            text_small(
                cell_x + (team_cell_w + 30) / 2,
                red_team_y + 30,
                team_num,
                center=True,
            )

        # Match time from NT
        match_time = int(self.nt_client.get_match_time())
        is_waiting_for_match = match_time < 0
        if is_waiting_for_match:
            match_time = 140
        minutes = match_time // 60
        seconds = match_time % 60

        time_str = f"{minutes}:{seconds:02d}"

        # Centered and bottom-aligned in the timer panel, same height as team number panel
        timer_panel_y = 70
        timer_panel_h = 60
        timer_panel_x = 450
        timer_panel_w = 1020
        timer_cy = timer_panel_y + timer_panel_h  # bottom edge
        text_large(
            960,
            timer_cy - timer_panel_h / 2,
            time_str,
            center=True,
            color=(0, 0, 0, 255),
        )

        # =========================
        # SHIFT TIMER PANEL (directly below time container)
        # =========================

        if not self.nt_client.get_autonomous():
            shift_panel_y = timer_panel_y + timer_panel_h + 1  # 1px border below
            shift_panel_h = 38
            # The blue and red background panels are at x=450, width=450, and x=1020, width=450
            shift_panel_x = 450 + 450  # right edge of blue panel
            shift_panel_w = (
                1020 - shift_panel_x
            )  # left edge of red panel minus right edge of blue panel
            # Draw border (1px #DDDBDC) between match time and shift timer panel
            border_col = imgui.get_color_u32_rgba(0xDD / 255, 0xDB / 255, 0xDC / 255, 1)
            draw = imgui.get_background_draw_list()
            # Horizontal divider line at top of shift panel
            x0, y0 = S(shift_panel_x, shift_panel_y - 1)
            x1, y1 = S(shift_panel_x + shift_panel_w, shift_panel_y)
            draw.add_rect_filled(x0, y0, x1, y1 + 1.0, border_col, 0)
            # Outer border for the shift timer panel
            x0b, y0b = S(shift_panel_x, shift_panel_y)
            x1b, y1b = S(shift_panel_x + shift_panel_w, shift_panel_y + shift_panel_h)
            draw.add_rect(x0b, y0b, x1b, y1b, border_col, 0, 1.0 * scale)

            shift_num = (
                1
                if match_time > 130
                else (
                    2
                    if match_time > 105
                    else (
                        3
                        if match_time > 80
                        else 4 if match_time > 55 else 5 if match_time > 30 else 6
                    )
                )
            )

            shift_time_left = match_time - (
                130
                if shift_num == 1
                else (
                    105
                    if shift_num == 2
                    else (
                        80
                        if shift_num == 3
                        else 55 if shift_num == 4 else 30 if shift_num == 5 else 0
                    )
                )
            )

            # Shift count panel (75px wide, white bg, black text)
            shift_count_w = 75
            shift_count_x = shift_panel_x
            shift_count_y = shift_panel_y
            rect(shift_count_x, shift_count_y, shift_count_w, shift_panel_h, col(WHITE))
            text(
                shift_count_x + shift_count_w / 2,
                shift_count_y + shift_panel_h / 2,
                f"{shift_num} / 6",
                color=(0, 0, 0, 255),
                center=True,
            )

            # Shift timer panel (remaining width, #DDDBDC bg, black text)
            shift_timer_x = shift_count_x + shift_count_w
            shift_timer_w = shift_panel_w - shift_count_w
            rect(shift_timer_x, shift_count_y, shift_timer_w, shift_panel_h, border_col)
            text_small(
                shift_timer_x + shift_timer_w / 2,
                shift_count_y + shift_panel_h / 2,
                f":{shift_time_left:02d}",
                color=(0, 0, 0, 255),
                center=True,
            )

        # =========================
        # TOP LABEL TEXTS
        # =========================
        text(
            960,
            37.5,
            f"{["Match", "Practice", "Qualification", "Playoff Match"][max(0, min(3, MATCH_TYPE))]} {MATCH_NUMBER} of {MATCH_TOTAL}",
            size=20,
            center=True,
        )

        # Left logo placeholder (now with static image)
        rect(450, 15, 180, 45, col((0.9, 0.9, 0.9, 1)))
        # Draw static image: logos/first_age_logo_horizontal_rgb_onecolor.png
        if not hasattr(self, "_first_age_logo_tex"):
            try:
                logo_path = (
                    Path(__file__).resolve().parent
                    / "logos"
                    / "first_age_logo_horizontal_rgb_onecolor.png"
                )
                logo_img = Image.open(logo_path).convert("RGBA")
                tex = self.ctx.texture(logo_img.size, 4, logo_img.tobytes())
                self.imgui.renderer.register_texture(tex)
                self._first_age_logo_tex = tex.glo
                self._first_age_logo_mgl = tex  # Prevent GC
            except Exception as e:
                logging.error(f"Failed to load FIRST AGE logo image: {e}")
                self._first_age_logo_tex = None
        if self._first_age_logo_tex is not None:
            imgui.get_background_draw_list().add_image(
                self._first_age_logo_tex,
                S(450, 15),
                S(450 + 180, 15 + 45),
            )
        # else: fallback to nothing (background remains white)

        # Right logo placeholder
        rect(1290, 15, 180, 45, col((0.9, 0.9, 0.9, 1)))
        # Draw static image: logos/first_age_frc_rebuilt_wordmark_rgb_black.png
        if not hasattr(self, "_rebuilt_logo_tex"):
            try:
                rebuilt_logo_path = (
                    Path(__file__).resolve().parent
                    / "logos"
                    / "first_age_frc_rebuilt_wordmark_rgb_black.png"
                )
                rebuilt_logo_img = Image.open(rebuilt_logo_path).convert("RGBA")
                rebuilt_tex = self.ctx.texture(
                    rebuilt_logo_img.size, 4, rebuilt_logo_img.tobytes()
                )
                self.imgui.renderer.register_texture(rebuilt_tex)
                self._rebuilt_logo_tex = rebuilt_tex.glo
                self._rebuilt_logo_mgl = rebuilt_tex  # Prevent GC
            except Exception as e:
                logging.error(f"Failed to load REBUILT logo image: {e}")
                self._rebuilt_logo_tex = None
        if self._rebuilt_logo_tex is not None:
            imgui.get_background_draw_list().add_image(
                self._rebuilt_logo_tex,
                S(1290, 15),
                S(1290 + 180, 15 + 45),
            )
        # else: fallback to nothing (background remains gray)

        # =========================

        # LEFT STACK COUNTER
        # =========================
        rect(35, 67, 45, 45, col(WHITE))
        # Draw blue fuel icon on top of white square
        if not hasattr(self, "_fuelblue_tex"):
            try:
                fuelblue_path = (
                    Path(__file__).resolve().parent / "logos" / "fuelblue.png"
                )
                fuelblue_img = Image.open(fuelblue_path).convert("RGBA")
                fuelblue_tex = self.ctx.texture(
                    fuelblue_img.size, 4, fuelblue_img.tobytes()
                )
                self.imgui.renderer.register_texture(fuelblue_tex)
                self._fuelblue_tex = fuelblue_tex.glo
                self._fuelblue_mgl = fuelblue_tex  # Prevent GC
                self._fuelblue_size = fuelblue_img.size
            except Exception as e:
                logging.error(f"Failed to load fuelblue icon: {e}")
                self._fuelblue_tex = None
        if self._fuelblue_tex is not None:
            # Center icon in 45x45 square at (35,67)
            icon_w, icon_h = 45, 45
            x0, y0 = S(35 + (45 - icon_w) / 2, 67 + (45 - icon_h) / 2)
            x1, y1 = S(35 + (45 + icon_w) / 2, 67 + (45 + icon_h) / 2)
            imgui.get_background_draw_list().add_image(
                self._fuelblue_tex,
                (x0, y0),
                (x1, y1),
            )
        rect(80, 67, 150, 45, col(BLUE))
        ranking_target_blue = 300 if blue_score >= 100 else 100
        text(155, 89.5, f"{blue_score} / {ranking_target_blue}", size=18, center=True)

        # Show left arrow only if blue hub is active
        if self.nt_client.get_blue_hub_active():
            rect(302, 67, 45, 45, col((1, 1, 0, 1)))
            # Draw left arrow image (flipped horizontally)
            if not hasattr(self, "_arrow_tex"):
                try:
                    arrow_path = Path(__file__).resolve().parent / "logos" / "arrow.png"
                    arrow_img = Image.open(arrow_path).convert("RGBA")
                    arrow_tex = self.ctx.texture(arrow_img.size, 4, arrow_img.tobytes())
                    self.imgui.renderer.register_texture(arrow_tex)
                    self._arrow_tex = arrow_tex.glo
                    self._arrow_mgl = arrow_tex  # Prevent GC
                    self._arrow_size = arrow_img.size
                except Exception as e:
                    logging.error(f"Failed to load arrow image: {e}")
                    self._arrow_tex = None
            if self._arrow_tex is not None:
                # Draw flipped horizontally for left arrow
                x0, y0 = S(302, 67)
                x1, y1 = S(302 + 45, 67 + 45)
                imgui.get_background_draw_list().add_image(
                    self._arrow_tex,
                    (x1, y0),  # flip x
                    (x0, y1),
                )
            # else: fallback to nothing (background remains yellow)

        # =========================

        # RIGHT STACK COUNTER
        # =========================
        if self.nt_client.get_red_hub_active():
            rect(1573, 67, 45, 45, col((1, 1, 0, 1)))
            # Draw right arrow image (normal orientation)
            if not hasattr(self, "_arrow_tex"):
                try:
                    arrow_path = Path(__file__).resolve().parent / "logos" / "arrow.png"
                    arrow_img = Image.open(arrow_path).convert("RGBA")
                    arrow_tex = self.ctx.texture(arrow_img.size, 4, arrow_img.tobytes())
                    self.imgui.renderer.register_texture(arrow_tex)
                    self._arrow_tex = arrow_tex.glo
                    self._arrow_mgl = arrow_tex  # Prevent GC
                    self._arrow_size = arrow_img.size
                except Exception as e:
                    logging.error(f"Failed to load arrow image: {e}")
                    self._arrow_tex = None
            if self._arrow_tex is not None:
                x0, y0 = S(1573, 67)
                x1, y1 = S(1573 + 45, 67 + 45)
                imgui.get_background_draw_list().add_image(
                    self._arrow_tex,
                    (x0, y0),
                    (x1, y1),
                )
            # else: fallback to nothing (background remains yellow)
        rect(1689, 67, 45, 45, col(WHITE))
        # Draw red fuel icon on top of white square
        if not hasattr(self, "_fuelred_tex"):
            try:
                fuelred_path = Path(__file__).resolve().parent / "logos" / "fuelred.png"
                fuelred_img = Image.open(fuelred_path).convert("RGBA")
                fuelred_tex = self.ctx.texture(
                    fuelred_img.size, 4, fuelred_img.tobytes()
                )
                self.imgui.renderer.register_texture(fuelred_tex)
                self._fuelred_tex = fuelred_tex.glo
                self._fuelred_mgl = fuelred_tex  # Prevent GC
                self._fuelred_size = fuelred_img.size
            except Exception as e:
                logging.error(f"Failed to load fuelred icon: {e}")
                self._fuelred_tex = None
        if self._fuelred_tex is not None:
            # Center icon in 45x45 square at (1689,67)
            icon_w, icon_h = 45, 45
            x0, y0 = S(1689 + (45 - icon_w) / 2, 67 + (45 - icon_h) / 2)
            x1, y1 = S(1689 + (45 + icon_w) / 2, 67 + (45 + icon_h) / 2)
            imgui.get_background_draw_list().add_image(
                self._fuelred_tex,
                (x0, y0),
                (x1, y1),
            )
        rect(1734, 67, 150, 45, col(RED))
        ranking_target_red = 300 if red_score >= 100 else 100
        text(1809, 89.5, f"{red_score} / {ranking_target_red}", size=18, center=True)

        if self._ui_font is not None:
            imgui.pop_font()

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
        if self._bloom_enabled:
            self._init_bloom_targets()

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
        # Alt+Enter fullscreen toggle
        keys = self.wnd.keys
        is_alt = False
        # modifiers can be int (bitmask) or object with .alt attribute
        if hasattr(modifiers, "alt"):
            is_alt = bool(getattr(modifiers, "alt", False))
        elif isinstance(modifiers, int):
            # moderngl_window uses bitmask: 0x0008 for Alt
            is_alt = (modifiers & 0x0008) != 0
        if action == self.wnd.keys.ACTION_PRESS:
            if is_alt and key == keys.ENTER:
                # Toggle fullscreen
                self.wnd.fullscreen = not self.wnd.fullscreen
            self._keys_pressed.add(key)
        elif action == self.wnd.keys.ACTION_RELEASE:
            self._keys_pressed.discard(key)

    def on_key_event(self, key: int, action: int, modifiers) -> None:
        self.key_event(key, action, modifiers)

    def char_event(self, char: str) -> None:
        self.imgui.unicode_char_entered(char)

    def on_unicode_char_entered(self, char: str) -> None:
        self.char_event(char)
