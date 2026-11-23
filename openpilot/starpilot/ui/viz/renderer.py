"""Draws one scene view: ground pass, road geometry from the model, and spheres for the car and leads.

Per frame the CPU work is a handful of small numpy projections and about a dozen batched draw calls; the road, sky and
camera reprojection run in a single full-layer shader pass. GL resources are created on first use.

To add a visualization mode: add a VizMode in options.py and a ModeStyle here. To add an angle: see rigs.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.swaglog import cloudlog
from openpilot.starpilot.ui.raylib import rl
from openpilot.starpilot.ui.viz import shaders
from openpilot.starpilot.ui.viz.camera_feed import CameraGeometry, FrameTextures
from openpilot.starpilot.ui.viz.geometry import VirtualCamera, clip_line, ground_disc, offset_line, project_ribbon
from openpilot.starpilot.ui.viz.options import VizMode
from openpilot.starpilot.ui.viz.rigs import Rig
from openpilot.starpilot.ui.viz.scene import EGO_RADIUS, LEAD_RADIUS, RADAR_TO_CAMERA, Scene
from openpilot.system.ui.lib.shader_polygon import Gradient, draw_polygon

UNIFORM_FLOAT = rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT
UNIFORM_INT = rl.ShaderUniformDataType.SHADER_UNIFORM_INT
UNIFORM_VEC3 = rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3
UNIFORM_VEC4 = rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4

MAX_LINE_DISTANCE = 120.0  # m, like the stock renderer's draw distance cap with some extra for the top view
PATH_HALF_WIDTH = 0.9      # m, stock path width
MIN_PATH_LENGTH = 1.5      # m; when stopped the predicted path is a few centimetres long and would draw as a sliver

# Stock path colours (model_renderer.THROTTLE_COLORS / NO_THROTTLE_COLORS), bottom to top of the layer.
THROTTLE = ((13, 248, 122, 102), (114, 255, 92, 89), (114, 255, 92, 0))
NO_THROTTLE = ((242, 242, 242, 102), (242, 242, 242, 89), (242, 242, 242, 0))

EGO_COLORS = {"engaged": (40, 205, 115), "override": (160, 168, 164), "disengaged": (70, 145, 240)}
LEAD_COLOR = (232, 62, 52)
SECOND_LEAD_COLOR = (235, 150, 55)
LIGHT_DIR = np.array([-0.45, 0.62, 0.64], dtype=np.float32)  # from upper left, slightly towards the viewer


@dataclass(frozen=True)
class Palette:
  sky_top: tuple = (16, 22, 33, 255)
  sky_horizon: tuple = (62, 76, 96, 255)
  ground: tuple = (30, 33, 39, 255)
  grid: tuple = (110, 122, 140, 90)   # alpha is grid strength
  grid_size: float = 4.0              # m
  fog_distance: float = 140.0         # m


@dataclass(frozen=True)
class ModeStyle:
  shader_mode: int                 # 0 synthetic ground, 1 camera colour, 2 hybrid markings
  road_surface: tuple              # ground between the model's road edges where no camera image is shown; alpha 0 = none
  lane_rgb: tuple
  lane_alpha: float                # scales the model's lane line probability
  edge_alpha: float
  path_alpha: float
  lane_width_scale: float = 1.0


MODE_STYLES = {
  VizMode.CAMERA: ModeStyle(1, (46, 50, 58, 200), (255, 255, 255), 0.85, 0.8, 1.0),
  VizMode.SYNTHETIC: ModeStyle(0, (54, 59, 68, 255), (245, 245, 245), 1.0, 0.9, 1.5),
  # hybrid: thin blue model lines so the paint found in the camera stays visible underneath
  VizMode.HYBRID: ModeStyle(2, (54, 59, 68, 255), (120, 180, 255), 0.55, 0.6, 1.3, lane_width_scale=0.45),
}

# How much of the real camera the reprojection trusts (per stream). The wide lens is treated as a pinhole like the
# rest of openpilot, which bends towards its edges, so only the central part is used. Rows near the bottom are usually
# the hood or dashboard.
CAMERA_LIMITS = {
  # min road distance (m), fade start (m), fade end (m), lowest usable v, widest usable ray (tan^2)
  False: (4.0, 70.0, 140.0, 1.0, 0.0),
  True: (2.5, 45.0, 90.0, 0.80, 1.4 ** 2),
}


def _color(rgba, alpha_scale: float = 1.0) -> rl.Color:
  r, g, b, a = rgba
  return rl.Color(int(r), int(g), int(b), int(np.clip(a * alpha_scale, 0, 255)))


def _vec4(rgba) -> np.ndarray:
  return np.array(rgba, dtype=np.float32) / 255.0


def _hls_to_rgb(h: np.ndarray, l: np.ndarray, s: np.ndarray) -> np.ndarray:
  """Vectorised colorsys.hls_to_rgb, (N,) arrays -> (N, 3)."""
  q = np.where(l <= 0.5, l * (1.0 + s), l + s - l * s)
  p = 2.0 * l - q

  def channel(t):
    t = t % 1.0
    return np.select([t < 1 / 6, t < 0.5, t < 2 / 3], [p + (q - p) * t * 6.0, q, p + (q - p) * (2 / 3 - t) * 6.0], p)
  return np.column_stack((channel(h + 1 / 3), channel(h), channel(h - 1 / 3)))


@dataclass
class _Ribbons:
  lines: list            # (left, right, color) for road edges and lane lines
  path: np.ndarray       # driven path, clipped at the lead
  path_edges: tuple | None


class _GroundShader:
  """Uniform locations and upload buffers for one ground shader program."""
  COMMON = ("uvToRay", "camPos", "groundZ", "odometer", "gridSize", "fogDistance", "skyTop", "skyHorizon", "groundColor", "gridColor",
            "roadColor", "roadLeft", "roadRight", "roadStep")
  CAMERA = ("groundToTex", "camLimits", "camOptics", "maxTan2", "mode", "colorStrength", "paintWhite", "paintYellow", "texture1")

  def __init__(self, fragment: str, camera: bool):
    self.shader = rl.load_shader_from_memory(shaders.VERTEX_SHADER, fragment)
    # raylib falls back to its default shader when compilation fails; treat that as unusable rather than draw garbage
    self.valid = bool(self.shader.id) and self.shader.id != rl.rl_get_shader_id_default()
    names = self.COMMON + (self.CAMERA if camera else ())
    self.loc = {name: rl.get_shader_location(self.shader, name) for name in names} if self.valid else {}

  def set(self, name: str, value, kind, count: int | None = None) -> None:
    loc = self.loc.get(name, -1)
    if loc < 0:
      return
    if count is None:
      rl.set_shader_value(self.shader, loc, value, kind)
    else:
      rl.set_shader_value_v(self.shader, loc, value, kind, count)

  def close(self) -> None:
    if self.valid:
      rl.unload_shader(self.shader)


class VizRenderer:
  def __init__(self, palette: Palette | None = None):
    self.palette = palette or Palette()
    self._synthetic: _GroundShader | None = None
    self._camera: dict[bool, _GroundShader] = {}
    self._sphere = None
    self._white = None
    self._odometer = 0.0
    self._throttle_blend = FirstOrderFilter(1.0, 0.25, 0.05)
    self._road_cache_frame = -2
    self._road_cache: tuple[np.ndarray | None, np.ndarray | None] = (None, None)
    self._ribbon_key: tuple | None = None
    self._ribbon_cache = _Ribbons([], np.empty((0, 3)), None)
    self.last_camera: VirtualCamera | None = None

  # ---- resources

  def _ensure(self) -> None:
    if self._white is None:
      image = rl.gen_image_color(1, 1, rl.WHITE)
      self._white = rl.load_texture_from_image(image)
      rl.unload_image(image)
    if self._synthetic is None:
      self._synthetic = _GroundShader(shaders.SYNTHETIC_GROUND_SHADER, camera=False)
      if not self._synthetic.valid:
        raise RuntimeError("scene views: synthetic ground shader failed to compile")
    if self._sphere is None:
      self._sphere = rl.load_shader_from_memory(shaders.VERTEX_SHADER, shaders.SPHERE_SHADER)
      if not self._sphere.id or self._sphere.id == rl.rl_get_shader_id_default():
        raise RuntimeError("scene views: sphere shader failed to compile")
      loc = rl.get_shader_location(self._sphere, "lightDir")
      rl.set_shader_value(self._sphere, loc, LIGHT_DIR, UNIFORM_VEC3)

  def _camera_shader(self, egl: bool) -> _GroundShader | None:
    """None when this GPU could not compile it; camera modes then fall back to the synthetic ground."""
    if egl not in self._camera:
      # StarPilot's camera colour treatment, so reprojected video matches the stock view (imported where it is needed;
      # the stock CameraView already loaded it)
      from openpilot.starpilot.ui.camera_color import ROAD_COLOR_SHADER
      program = _GroundShader(shaders.camera_ground_shader(egl, ROAD_COLOR_SHADER), camera=True)
      if not program.valid:
        cloudlog.error("scene views: camera ground shader failed to compile, showing the synthetic road instead")
      self._camera[egl] = program
    program = self._camera[egl]
    return program if program.valid else None

  def close(self) -> None:
    for program in (self._synthetic, *self._camera.values()):
      if program is not None:
        program.close()
    self._synthetic, self._camera = None, {}
    if self._sphere is not None and self._sphere.id:
      rl.unload_shader(self._sphere)
    self._sphere = None
    if self._white is not None and self._white.id:
      rl.unload_texture(self._white)
    self._white = None

  # ---- frame

  def render(self, rect: rl.Rectangle, rig: Rig, camera: VirtualCamera, scene: Scene, mode: VizMode, *, dt: float,
             frame: FrameTextures | None = None, source: CameraGeometry | None = None, color_strength: float = 1.0,
             status: str = "disengaged") -> None:
    self._ensure()
    self.last_camera = camera
    style = MODE_STYLES[mode]
    self._odometer = (self._odometer + max(scene.v_ego, 0.0) * dt) % 1000.0
    self._throttle_blend.update(1.0 if scene.allow_throttle else 0.0)

    program = self._camera_shader(frame.egl) if style.shader_mode and frame is not None and source is not None else None
    if program is not None:
      self._draw_camera_ground(program, rect, camera, scene, style, frame, source, color_strength)
    else:
      self._draw_synthetic_ground(rect, camera, scene, style)
    if not scene.valid:
      return

    self._draw_road(rect, rig, camera, scene, style)
    self._draw_path(rect, rig, camera, scene, style)
    self._draw_bodies(rig, camera, scene, status)

  # ---- ground

  def _ground_uniforms(self, program: _GroundShader, rect: rl.Rectangle, camera: VirtualCamera, scene: Scene, style: ModeStyle) -> None:
    p = self.palette
    uv_to_ray = camera.uv_to_ray((rect.x, rect.y, rect.width, rect.height))
    program.set("uvToRay", np.ascontiguousarray(uv_to_ray.T, dtype=np.float32), UNIFORM_VEC3, 3)
    program.set("camPos", camera.position.astype(np.float32), UNIFORM_VEC3)
    program.set("groundZ", np.array([scene.height], dtype=np.float32), UNIFORM_FLOAT)
    program.set("odometer", np.array([self._odometer], dtype=np.float32), UNIFORM_FLOAT)
    program.set("gridSize", np.array([p.grid_size], dtype=np.float32), UNIFORM_FLOAT)
    program.set("fogDistance", np.array([p.fog_distance], dtype=np.float32), UNIFORM_FLOAT)
    program.set("skyTop", _vec4(p.sky_top), UNIFORM_VEC4)
    program.set("skyHorizon", _vec4(p.sky_horizon), UNIFORM_VEC4)
    program.set("groundColor", _vec4(p.ground), UNIFORM_VEC4)
    program.set("gridColor", _vec4(p.grid), UNIFORM_VEC4)
    left, right = self._road_samples(scene)
    program.set("roadColor", _vec4(style.road_surface if left is not None else (0, 0, 0, 0)), UNIFORM_VEC4)
    if left is not None:
      program.set("roadLeft", left, UNIFORM_FLOAT, shaders.ROAD_SAMPLES)
      program.set("roadRight", right, UNIFORM_FLOAT, shaders.ROAD_SAMPLES)
    program.set("roadStep", np.array([shaders.ROAD_STEP], dtype=np.float32), UNIFORM_FLOAT)

  def _road_samples(self, scene: Scene) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The model's road edges resampled at fixed distances for the ground shader (cached per model frame)."""
    if self._road_cache_frame == scene.model_frame:
      return self._road_cache
    left = right = None
    if scene.valid and len(scene.road_edges) == 2 and all(len(edge) > 1 for edge in scene.road_edges):
      xs = np.arange(shaders.ROAD_SAMPLES) * shaders.ROAD_STEP
      left = np.interp(xs, scene.road_edges[0][:, 0], scene.road_edges[0][:, 1]).astype(np.float32)
      right = np.interp(xs, scene.road_edges[1][:, 0], scene.road_edges[1][:, 1]).astype(np.float32)
    self._road_cache_frame, self._road_cache = scene.model_frame, (left, right)
    return left, right

  def _draw_synthetic_ground(self, rect: rl.Rectangle, camera: VirtualCamera, scene: Scene, style: ModeStyle) -> None:
    program = self._synthetic
    rl.begin_shader_mode(program.shader)
    self._ground_uniforms(program, rect, camera, scene, style)
    rl.draw_texture_pro(self._white, rl.Rectangle(0, 0, 1, 1), rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
    rl.end_shader_mode()

  def _draw_camera_ground(self, program: _GroundShader, rect: rl.Rectangle, camera: VirtualCamera, scene: Scene, style: ModeStyle,
                          frame: FrameTextures, source: CameraGeometry, color_strength: float) -> None:
    k = source.intrinsics
    # road plane (X, Y, 1) -> camera pixels -> texture uv
    ground = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, scene.height]])
    to_uv = np.diag([1.0 / frame.uv_width, 1.0 / frame.uv_height, 1.0])
    ground_to_tex = to_uv @ k @ source.view_from_calib @ ground
    near, fade_start, fade_end, max_v, max_tan2 = CAMERA_LIMITS[source.wide]
    optics = np.array([k[0, 2] / frame.uv_width, k[1, 2] / frame.uv_height, k[0, 0] / frame.uv_width, k[1, 1] / frame.uv_height],
                      dtype=np.float32)

    rl.begin_shader_mode(program.shader)
    self._ground_uniforms(program, rect, camera, scene, style)
    program.set("groundToTex", np.ascontiguousarray(ground_to_tex.T, dtype=np.float32), UNIFORM_VEC3, 3)
    program.set("camLimits", np.array([near, fade_start, fade_end, max_v], dtype=np.float32), UNIFORM_VEC4)
    program.set("camOptics", optics, UNIFORM_VEC4)
    program.set("maxTan2", np.array([max_tan2], dtype=np.float32), UNIFORM_FLOAT)
    program.set("mode", np.array([style.shader_mode], dtype=np.int32), UNIFORM_INT)
    program.set("colorStrength", np.array([color_strength], dtype=np.float32), UNIFORM_FLOAT)
    program.set("paintWhite", _vec4((240, 240, 240, 255)), UNIFORM_VEC4)
    program.set("paintYellow", _vec4((250, 200, 40, 255)), UNIFORM_VEC4)
    if frame.uv is not None and program.loc.get("texture1", -1) >= 0:
      rl.set_shader_value_texture(program.shader, program.loc["texture1"], frame.uv)
    src = rl.Rectangle(0, 0, float(frame.texture.width), float(frame.texture.height))
    rl.draw_texture_pro(frame.texture, src, rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
    rl.end_shader_mode()

  # ---- model geometry

  def _ribbons(self, rig: Rig, scene: Scene, style: ModeStyle) -> _Ribbons:
    """Road-frame ribbons only change with a new model/radar message, so build them once per scene revision."""
    key = (scene.revision, rig.angle, style)
    if key == self._ribbon_key:
      return self._ribbon_cache
    lines: list[tuple[np.ndarray, np.ndarray, rl.Color]] = []
    for i, edge in enumerate(scene.road_edges):
      std = scene.road_edge_stds[i] if i < len(scene.road_edge_stds) else 1.0
      alpha = float(np.clip(1.0 - std, 0.0, 1.0)) * style.edge_alpha
      if alpha > 0.01 and len(edge) > 1:
        lines.append((*offset_line(clip_line(edge, MAX_LINE_DISTANCE), rig.edge_half_width), _color((255, 40, 40, 255 * alpha))))
    for i, line in enumerate(scene.lane_lines):
      prob = scene.lane_probs[i] if i < len(scene.lane_probs) else 0.0
      alpha = float(np.clip(prob, 0.0, 1.0)) * style.lane_alpha
      if alpha > 0.01 and len(line) > 1:
        half_width = rig.lane_half_width * style.lane_width_scale
        lines.append((*offset_line(clip_line(line, MAX_LINE_DISTANCE), half_width), _color((*style.lane_rgb, 255 * alpha))))
    # The path stops at the lead's rear bumper instead of running underneath it.
    end = MAX_LINE_DISTANCE if not scene.leads else max(RADAR_TO_CAMERA + scene.leads[0].d_rel - 0.5, 0.0)
    path = clip_line(scene.path, end) if len(scene.path) > 1 else scene.path
    long_enough = len(path) > 1 and path[-1, 0] - path[0, 0] >= MIN_PATH_LENGTH
    path_edges = offset_line(path, PATH_HALF_WIDTH) if long_enough else None
    self._ribbon_key, self._ribbon_cache = key, _Ribbons(lines, path, path_edges)
    return self._ribbon_cache

  def _draw_road(self, rect: rl.Rectangle, rig: Rig, camera: VirtualCamera, scene: Scene, style: ModeStyle) -> None:
    for left, right, color in self._ribbons(rig, scene, style).lines:
      polygon = project_ribbon(camera, left, right)
      if polygon is not None:
        draw_polygon(rect, polygon, color)

  def _draw_path(self, rect: rl.Rectangle, rig: Rig, camera: VirtualCamera, scene: Scene, style: ModeStyle) -> None:
    ribbons = self._ribbons(rig, scene, style)
    if ribbons.path_edges is None:
      return
    polygon = project_ribbon(camera, *ribbons.path_edges)
    if polygon is None:
      return
    gradient = None
    if scene.experimental and len(scene.path_accel) == len(scene.path):
      gradient = self._accel_gradient(rect, camera, scene, ribbons.path, style.path_alpha)
    else:
      blend = float(np.clip(self._throttle_blend.x, 0.0, 1.0))
      colors = [_color(tuple(a + (b - a) * blend for a, b in zip(off, on, strict=True)), style.path_alpha)
                for off, on in zip(NO_THROTTLE, THROTTLE, strict=True)]
      gradient = Gradient(start=(0.0, 1.0), end=(0.0, 0.0), colors=colors, stops=[0.0, 0.5, 1.0])
    if gradient is None:
      draw_polygon(rect, polygon, _color((255, 255, 255, 30), style.path_alpha))
    else:
      draw_polygon(rect, polygon, gradient=gradient)

  @staticmethod
  def _accel_gradient(rect: rl.Rectangle, camera: VirtualCamera, scene: Scene, path: np.ndarray, alpha_scale: float) -> Gradient | None:
    """Experimental mode: hue by planned acceleration along the path (stock colouring, screen-space stops)."""
    xs = np.linspace(path[0, 0], path[-1, 0], 8)
    points = np.column_stack((xs, np.interp(xs, path[:, 0], path[:, 1]), np.interp(xs, path[:, 0], path[:, 2])))
    ys = camera.project(points)[0][:, 1]
    ok = np.isfinite(ys)
    if ok.sum() < 2:
      return None
    accel = np.interp(xs[ok], scene.path[:, 0], scene.path_accel)
    stops = np.clip(1.0 - (ys[ok] - rect.y) / rect.height, 0.0, 1.0)
    saturation = np.minimum(np.abs(accel * 1.5), 1.0)
    rgb = _hls_to_rgb(np.clip(60 + accel * 35, 0, 120) / 360.0, np.interp(saturation, [0.0, 1.0], [0.95, 0.62]), saturation)
    alpha = np.interp(stops, [0.75 / 2.0, 0.75], [0.4, 0.0]) * alpha_scale
    order = np.argsort(stops)
    colors = [rl.Color(int(rgb[i, 0] * 255), int(rgb[i, 1] * 255), int(rgb[i, 2] * 255), int(np.clip(alpha[i] * 255, 0, 255))) for i in order]
    return Gradient(start=(0.0, 1.0), end=(0.0, 0.0), colors=colors, stops=[float(stops[i]) for i in order])

  # ---- spheres

  def _draw_bodies(self, rig: Rig, camera: VirtualCamera, scene: Scene, status: str) -> None:
    bodies = []  # (center, radius, rgb, alpha)
    if rig.show_ego:
      bodies.append((scene.ego_center, EGO_RADIUS, EGO_COLORS.get(status, EGO_COLORS["disengaged"]), 255))
    for lead in scene.leads:
      closing = float(np.clip(-lead.v_rel / 8.0, 0.0, 1.0))
      near = float(np.clip(1.0 - lead.d_rel / 60.0, 0.0, 1.0))
      rgb = LEAD_COLOR if lead.primary else SECOND_LEAD_COLOR
      rgb = tuple(int(c * (0.65 + 0.35 * max(closing, near))) for c in rgb)
      bodies.append((lead.center, LEAD_RADIUS, rgb, 255 if lead.primary else 200))

    projected = []
    for center, radius, rgb, alpha in bodies:
      shadow = ground_disc((center[0], center[1], center[2] + radius), radius * 1.15)
      xy, depth = camera.project(shadow)
      if np.isfinite(xy).all():
        rl.draw_triangle_strip(np.ascontiguousarray(xy, dtype=np.float32), len(xy), rl.Color(0, 0, 0, 105))
      hit = camera.project_point(center, near=radius + 0.2)
      if hit is not None:
        projected.append((hit[2], hit[0], hit[1], max(camera.focal * radius / hit[2], 3.0), rgb, alpha))

    rl.begin_shader_mode(self._sphere)
    for _depth, x, y, r, rgb, alpha in sorted(projected, reverse=True):  # far to near
      rl.draw_texture_pro(self._white, rl.Rectangle(0, 0, 1, 1), rl.Rectangle(x - r, y - r, 2 * r, 2 * r),
                          rl.Vector2(0, 0), 0.0, rl.Color(*rgb, alpha))
    rl.end_shader_mode()
