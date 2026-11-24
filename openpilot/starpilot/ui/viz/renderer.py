"""Draws one scene view: the ground pass (road, markings, path and everything else on the road) and the vehicles.

CPU work per frame is a few dozen small uniform uploads and one draw call for the ground plus one per vehicle. The
road geometry arrays are uploaded only when a new model frame arrives, and constants only once per shader.

To add a visualization mode: add a VizMode in options.py and a ModeStyle here. To add an angle: see rigs.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.swaglog import cloudlog
from openpilot.starpilot.ui.raylib import rl
from openpilot.starpilot.ui.viz import shaders
from openpilot.starpilot.ui.viz.camera_feed import CameraGeometry, FrameTextures
from openpilot.starpilot.ui.viz.geometry import VirtualCamera
from openpilot.starpilot.ui.viz.options import SceneStyle, VehicleLook, VehicleModel, VizMode
from openpilot.starpilot.ui.viz.rigs import Rig
from openpilot.starpilot.ui.viz.scene import EGO_LENGTH, RADAR_TO_CAMERA, SAMPLE_MAX_X, SAMPLE_X, Scene

UNIFORM_FLOAT = rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT
UNIFORM_INT = rl.ShaderUniformDataType.SHADER_UNIFORM_INT
UNIFORM_VEC3 = rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3
UNIFORM_VEC4 = rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4

# Stock path colours (model_renderer.THROTTLE_COLORS / NO_THROTTLE_COLORS), near and far.
THROTTLE = ((13, 248, 122, 102), (114, 255, 92, 0))
NO_THROTTLE = ((242, 242, 242, 102), (242, 242, 242, 0))
STATUS_GLOW = {"engaged": (0.12, 0.86, 0.45, 0.38), "override": (0.75, 0.78, 0.8, 0.22)}

# Vehicle model id in the shader, footprint (length, width) and the radius of its wheels.
VEHICLE_MODELS = {VehicleModel.CAR: (0, 4.6, 1.84, 0.34), VehicleModel.TRUCK: (1, 5.6, 2.0, 0.4),
                  VehicleModel.SPHERE: (2, 1.8, 1.8, 0.9)}
VEHICLE_HEIGHT = {VehicleModel.CAR: 1.48, VehicleModel.TRUCK: 1.95, VehicleModel.SPHERE: 1.82}
GLOSS = {VehicleModel.CAR: 0.85, VehicleModel.TRUCK: 0.7, VehicleModel.SPHERE: 0.9}
BLINK_PERIOD = 1.0   # s, matches StarPilot's turn signal border (500 ms on, 500 ms off)
ODOMETER_WRAP = 2400.0  # m; a multiple of the dash (12 m) and grid (4 m) periods so the wrap is invisible


@dataclass(frozen=True)
class Palette:
  sky_top: tuple = (16, 22, 33, 255)
  sky_horizon: tuple = (62, 76, 96, 255)
  ground: tuple = (30, 33, 39, 255)
  grid: tuple = (110, 122, 140, 70)   # alpha is grid strength
  fog_distance: float = 140.0         # m


@dataclass(frozen=True)
class ModeStyle:
  shader_mode: int                 # 0 synthetic ground, 1 camera colour, 2 hybrid markings
  road_surface: tuple              # ground between the road edges where no camera image is shown; alpha 0 = none
  lane_alpha: float                # scales the model's lane line probability
  edge_alpha: float
  path_alpha: float
  lane_width_scale: float = 1.0
  lane_rgb: tuple | None = None    # None: StarPilot's lane colours (white unless customised)


MODE_STYLES = {
  VizMode.CAMERA: ModeStyle(1, (46, 50, 58, 200), 0.85, 0.8, 1.0),
  VizMode.SYNTHETIC: ModeStyle(0, (52, 57, 66, 255), 1.0, 0.9, 1.0, lane_width_scale=1.2),
  # hybrid: thin blue model lines so the paint found in the camera stays visible underneath
  VizMode.HYBRID: ModeStyle(2, (52, 57, 66, 255), 0.55, 0.6, 1.0, lane_width_scale=0.45, lane_rgb=(120, 180, 255)),
}

# How much of the real camera the reprojection trusts: min road distance (m), fade start/end (m), lowest usable v.
CAMERA_LIMITS = {False: (4.0, 70.0, 140.0, 1.0), True: (2.5, 45.0, 90.0, 0.80)}


def _vec4(rgba) -> np.ndarray:
  return np.array(rgba, dtype=np.float32) / 255.0


def _hex(color: str | None) -> tuple[int, int, int, int] | None:
  if not isinstance(color, str) or len(color) != 9:
    return None
  try:
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5, 7))
  except ValueError:
    return None


def _f32(*values) -> np.ndarray:
  return np.array(values, dtype=np.float32)


class _Program:
  """A shader with its uniform locations; remembers which cached values it already holds."""

  def __init__(self, fragment: str, names: tuple[str, ...]):
    self.shader = rl.load_shader_from_memory(shaders.VERTEX_SHADER, fragment)
    # raylib falls back to its default shader when compilation fails; treat that as unusable rather than draw garbage
    self.valid = bool(self.shader.id) and self.shader.id != rl.rl_get_shader_id_default()
    self.loc = {name: rl.get_shader_location(self.shader, name) for name in names} if self.valid else {}
    self.keys: dict[str, object] = {}

  def set(self, name: str, value, kind, count: int | None = None) -> None:
    loc = self.loc.get(name, -1)
    if loc < 0:
      return
    if count is None:
      rl.set_shader_value(self.shader, loc, value, kind)
    else:
      rl.set_shader_value_v(self.shader, loc, value, kind, count)

  def changed(self, group: str, key) -> bool:
    """True (and remembered) when ``key`` differs from what this program last received for ``group``."""
    if self.keys.get(group) == key:
      return False
    self.keys[group] = key
    return True

  def close(self) -> None:
    if self.valid:
      rl.unload_shader(self.shader)


GROUND_UNIFORMS = ("uvToRay", "camPos", "groundZ", "odometer", "time", "fogDistance", "skyTop", "skyHorizon", "groundColor",
                   "gridColor", "roadColor", "laneY", "roadY", "sampleMaxX", "laneAlpha", "laneDash", "laneInner", "laneOuter",
                   "edgeStyle", "pathRange", "pathNear", "pathFar", "pathHalfWidth", "laneChange", "stopLine", "lighting",
                   "bodyPose", "bodySize", "statusGlow")
CAMERA_UNIFORMS = ("groundToTex", "camLimits", "mode", "colorStrength", "paintWhite", "paintYellow", "texture1")
VEHICLE_UNIFORMS = ("uvToRay", "camPos", "quad", "pose", "model", "paint", "lights", "extra", "sky")


@dataclass
class Body:
  """A vehicle to draw: its footprint centre on the road, heading and look."""
  center: np.ndarray
  yaw: float
  look: VehicleLook
  lights: np.ndarray   # headlights, tail/brake, left, right
  reverse: float = 0.0
  tail_glow: float = 0.0


class VizRenderer:
  def __init__(self, palette: Palette | None = None):
    self.palette = palette or Palette()
    self._synthetic: _Program | None = None
    self._camera: dict[bool, _Program] = {}
    self._vehicle: _Program | None = None
    self._white = None
    self._odometer = 0.0
    self._time = 0.0
    self._throttle = FirstOrderFilter(1.0, 0.25, 0.05)
    self._stop_x = FirstOrderFilter(30.0, 0.25, 0.05)
    self._stop_alpha = FirstOrderFilter(0.0, 0.2, 0.05)
    self._change = FirstOrderFilter(0.0, 0.25, 0.05)
    self._change_direction = 0
    self._dash = FirstOrderFilter(np.ones(4), 0.6, 0.05)
    self._dash_revision = -1
    self.last_camera: VirtualCamera | None = None

  # ---- resources

  def _ensure(self) -> None:
    if self._white is None:
      image = rl.gen_image_color(1, 1, rl.WHITE)
      self._white = rl.load_texture_from_image(image)
      rl.unload_image(image)
    if self._synthetic is None:
      self._synthetic = _Program(shaders.SYNTHETIC_GROUND_SHADER, GROUND_UNIFORMS)
      if not self._synthetic.valid:
        raise RuntimeError("scene views: synthetic ground shader failed to compile")
    if self._vehicle is None:
      self._vehicle = _Program(shaders.VEHICLE_SHADER, VEHICLE_UNIFORMS)
      if not self._vehicle.valid:
        cloudlog.error("scene views: vehicle shader failed to compile, vehicles are hidden")

  def _camera_program(self, egl: bool) -> _Program | None:
    """None when this GPU could not compile it; camera modes then fall back to the synthetic ground."""
    if egl not in self._camera:
      # StarPilot's camera colour treatment, so reprojected video matches the stock view (imported where it is needed;
      # the stock CameraView already loaded it)
      from openpilot.starpilot.ui.camera_color import ROAD_COLOR_SHADER
      program = _Program(shaders.camera_ground_shader(egl, ROAD_COLOR_SHADER), GROUND_UNIFORMS + CAMERA_UNIFORMS)
      if not program.valid:
        cloudlog.error("scene views: camera ground shader failed to compile, showing the synthetic road instead")
      self._camera[egl] = program
    program = self._camera[egl]
    return program if program.valid else None

  def close(self) -> None:
    for program in (self._synthetic, self._vehicle, *self._camera.values()):
      if program is not None:
        program.close()
    self._synthetic, self._vehicle, self._camera = None, None, {}
    if self._white is not None and self._white.id:
      rl.unload_texture(self._white)
    self._white = None

  # ---- frame

  def render(self, rect: rl.Rectangle, rig: Rig, camera: VirtualCamera, scene: Scene, mode: VizMode, *, dt: float,
             style: SceneStyle | None = None, road_style: dict | None = None, frame: FrameTextures | None = None,
             source: CameraGeometry | None = None, color_strength: float = 1.0, status: str = "disengaged",
             blink_time: float | None = None) -> None:
    self._ensure()
    self.last_camera = camera
    style = style or SceneStyle()
    mode_style = MODE_STYLES[mode]
    self._odometer = (self._odometer + max(scene.v_ego, 0.0) * dt) % ODOMETER_WRAP
    self._time += dt
    blink_on = ((self._time if blink_time is None else blink_time) % BLINK_PERIOD) < BLINK_PERIOD / 2
    self._throttle.update(1.0 if scene.allow_throttle else 0.0)

    program = self._camera_program(frame.egl) if mode_style.shader_mode and frame is not None and source is not None else None
    bodies = self._bodies(rig, scene, style, blink_on) if scene.valid else []
    ground = program or self._synthetic
    rl.begin_shader_mode(ground.shader)
    self._ground_uniforms(ground, rect, rig, camera, scene, mode_style, road_style, bodies, status)
    if program is not None:
      self._camera_uniforms(program, scene, mode_style, frame, source, color_strength)
      src = rl.Rectangle(0, 0, float(frame.texture.width), float(frame.texture.height))
      rl.draw_texture_pro(frame.texture, src, rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
    else:
      rl.draw_texture_pro(self._white, rl.Rectangle(0, 0, 1, 1), rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
    rl.end_shader_mode()
    if not scene.valid:
      return
    if rig.stop_curtain:
      self._draw_stop_curtain(rect, camera, scene)
    self._draw_vehicles(rect, camera, scene, bodies)

  # ---- ground

  def _ground_uniforms(self, program: _Program, rect: rl.Rectangle, rig: Rig, camera: VirtualCamera, scene: Scene,
                       mode_style: ModeStyle, road_style: dict | None, bodies: list[Body], status: str) -> None:
    p = self.palette
    if program.changed("constants", p):
      program.set("fogDistance", _f32(p.fog_distance), UNIFORM_FLOAT)
      program.set("skyTop", _vec4(p.sky_top), UNIFORM_VEC4)
      program.set("skyHorizon", _vec4(p.sky_horizon), UNIFORM_VEC4)
      program.set("groundColor", _vec4(p.ground), UNIFORM_VEC4)
      program.set("gridColor", _vec4(p.grid), UNIFORM_VEC4)
      program.set("sampleMaxX", _f32(SAMPLE_MAX_X), UNIFORM_FLOAT)

    # camera
    uv_to_ray = camera.uv_to_ray((rect.x, rect.y, rect.width, rect.height))
    program.set("uvToRay", np.ascontiguousarray(uv_to_ray.T, dtype=np.float32), UNIFORM_VEC3, 3)
    program.set("camPos", camera.position.astype(np.float32), UNIFORM_VEC3)
    program.set("groundZ", _f32(scene.height), UNIFORM_FLOAT)
    program.set("odometer", _f32(self._odometer), UNIFORM_FLOAT)
    program.set("time", _f32(self._time % 1000.0), UNIFORM_FLOAT)

    # geometry: once per model frame
    has_road = scene.valid and scene.road_samples is not None and len(scene.road_edges) == 2
    lane_rgb = mode_style.lane_rgb
    inner = _hex((road_style or {}).get("pathEdge")) if lane_rgb is None else None
    outer = _hex((road_style or {}).get("laneLines")) if lane_rgb is None else None
    if program.changed("geometry", (scene.revision, mode_style, rig.angle, inner, outer)):
      if scene.valid and scene.lane_samples is not None:
        program.set("laneY", scene.lane_samples, UNIFORM_VEC4, shaders.SAMPLES)
        program.set("roadY", scene.road_samples, UNIFORM_VEC4, shaders.SAMPLES)
        probs = np.zeros(4, dtype=np.float32)
        probs[:len(scene.lane_probs)] = np.clip(scene.lane_probs[:4], 0.0, 1.0)
        stds = np.ones(2)
        stds[:len(scene.road_edge_stds)] = scene.road_edge_stds[:2]
        edges = np.clip(1.0 - stds, 0.0, 1.0) * mode_style.edge_alpha
      else:
        probs, edges = np.zeros(4, dtype=np.float32), np.zeros(2)
      cap = 0.7 if inner is not None or outer is not None else 1.0  # StarPilot's custom lane colours cap at 70 %
      program.set("laneAlpha", np.minimum(probs, cap).astype(np.float32) * mode_style.lane_alpha, UNIFORM_VEC4)
      program.set("edgeStyle", _f32(edges[0], edges[1], rig.edge_half_width, rig.lane_half_width * mode_style.lane_width_scale),
                  UNIFORM_VEC4)
      white = (*(lane_rgb or (245, 245, 245)), 255)
      program.set("laneInner", _vec4(inner or white), UNIFORM_VEC4)
      program.set("laneOuter", _vec4(outer or white), UNIFORM_VEC4)
      program.set("roadColor", _vec4(mode_style.road_surface if has_road else (0, 0, 0, 0)), UNIFORM_VEC4)
      program.set("pathHalfWidth", _f32(rig.path_half_width), UNIFORM_FLOAT)
    program.set("laneDash", self._lane_dashes(scene).astype(np.float32), UNIFORM_VEC4)

    # path
    path_mode, near, far = self._path_colors(road_style, mode_style)
    start = 0.0 if not rig.show_ego else scene.ego_ground[0] + EGO_LENGTH / 2 - 0.3
    end = scene.path_end if scene.valid and len(scene.path) > 1 and scene.path_end - start >= 1.5 else 0.0
    program.set("pathRange", _f32(start, end, mode_style.path_alpha if end > start else 0.0, path_mode), UNIFORM_VEC4)
    program.set("pathNear", near, UNIFORM_VEC4)
    program.set("pathFar", far, UNIFORM_VEC4)

    # lane change, stop line, lights
    program.set("laneChange", self._lane_change(scene), UNIFORM_VEC4)
    program.set("stopLine", self._stop_line(scene), UNIFORM_VEC4)
    darkness = scene.darkness
    program.set("lighting", _f32(darkness, float(np.clip((darkness - 0.3) / 0.4, 0.0, 1.0)), RADAR_TO_CAMERA, 0.0), UNIFORM_VEC4)

    poses = np.zeros((shaders.MAX_BODIES, 4), dtype=np.float32)
    sizes = np.zeros((shaders.MAX_BODIES, 4), dtype=np.float32)
    for i, body in enumerate(bodies[:shaders.MAX_BODIES]):
      _, length, width, _ = VEHICLE_MODELS[body.look.model]
      poses[i] = (body.center[0], body.center[1], math.cos(body.yaw), math.sin(body.yaw))
      round_body = body.look.model == VehicleModel.SPHERE
      sizes[i] = (length / 2 * (0.6 if round_body else 1.0), width / 2 * (0.6 if round_body else 1.0), 0.55, body.tail_glow)
    program.set("bodyPose", poses, UNIFORM_VEC4, shaders.MAX_BODIES)
    program.set("bodySize", sizes, UNIFORM_VEC4, shaders.MAX_BODIES)
    glow = STATUS_GLOW.get(status) if rig.show_ego and bodies else None
    program.set("statusGlow", _f32(*(glow or (0.0, 0.0, 0.0, 0.0))), UNIFORM_VEC4)

  def _camera_uniforms(self, program: _Program, scene: Scene, mode_style: ModeStyle, frame: FrameTextures, source: CameraGeometry,
                       color_strength: float) -> None:
    k = source.intrinsics
    # road plane (X, Y, 1) -> camera pixels -> texture uv
    ground = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, scene.height]])
    to_uv = np.diag([1.0 / frame.uv_width, 1.0 / frame.uv_height, 1.0])
    ground_to_tex = to_uv @ k @ source.view_from_calib @ ground
    program.set("groundToTex", np.ascontiguousarray(ground_to_tex.T, dtype=np.float32), UNIFORM_VEC3, 3)
    program.set("camLimits", _f32(*CAMERA_LIMITS[source.wide]), UNIFORM_VEC4)
    program.set("mode", np.array([mode_style.shader_mode], dtype=np.int32), UNIFORM_INT)
    program.set("colorStrength", _f32(color_strength), UNIFORM_FLOAT)
    if program.changed("paint", True):
      program.set("paintWhite", _vec4((240, 240, 240, 255)), UNIFORM_VEC4)
      program.set("paintYellow", _vec4((250, 200, 40, 255)), UNIFORM_VEC4)
    if frame.uv is not None and program.loc.get("texture1", -1) >= 0:
      rl.set_shader_value_texture(program.shader, program.loc["texture1"], frame.uv)

  def _lane_dashes(self, scene: Scene) -> np.ndarray:
    """Lines right beside a road edge are solid edge lines; lines between lanes are dashed (smoothed over time)."""
    if scene.revision != self._dash_revision and scene.valid and scene.lane_samples is not None and len(scene.road_edges) == 2:
      self._dash_revision = scene.revision
      window = slice(6, 14)  # about 5 to 45 m ahead
      lanes = scene.lane_samples[window]
      edges = scene.road_samples[window, :2]
      gap = np.minimum(np.abs(lanes - edges[:, :1]), np.abs(lanes - edges[:, 1:2])).mean(axis=0)
      self._dash.update((gap > 1.1).astype(np.float64))
    return np.round(self._dash.x * 4.0) / 4.0

  def _path_colors(self, road_style: dict | None, mode_style: ModeStyle) -> tuple[float, np.ndarray, np.ndarray]:
    """Shader colour mode and near/far colours following StarPilot's path style (stock throttle colours without one)."""
    if road_style is not None:
      from openpilot.starpilot.ui.road_colors import path_mode
      chosen = path_mode(road_style)
      if chosen == "rainbow":
        return 3.0, _vec4((255, 255, 255, 255)), _vec4((255, 255, 255, 0))
      if chosen == "acceleration":
        return 2.0, _vec4((255, 255, 255, 255)), _vec4((255, 255, 255, 0))
      color = _hex(road_style.get("path")) or (48, 255, 156, 255)
      return 1.0, _vec4((*color[:3], color[3] * 0.75)), _vec4((*color[:3], 0))
    blend = float(np.clip(self._throttle.x, 0.0, 1.0))
    near = [a + (b - a) * blend for a, b in zip(NO_THROTTLE[0], THROTTLE[0], strict=True)]
    far = [a + (b - a) * blend for a, b in zip(NO_THROTTLE[1], THROTTLE[1], strict=True)]
    near[3] = 190  # the stock 40 % alpha disappears on a dark synthetic road
    return 0.0, _vec4(near), _vec4(far)

  def _lane_change(self, scene: Scene) -> np.ndarray:
    target = {1: 0.55 + 0.25 * math.sin(self._time * 5.0), 2: 1.0, 3: 0.35}.get(scene.lane_change, 0.0)
    if scene.lane_change_direction == 0:
      target = 0.0
    else:
      self._change_direction = scene.lane_change_direction
    strength = self._change.update(target)
    signals = scene.signals
    blocked = signals.left_blindspot if self._change_direction < 0 else signals.right_blindspot
    return _f32(self._change_direction, strength if strength > 0.01 else 0.0, 1.0 if blocked else 0.0, 0.0)

  def _stop_line(self, scene: Scene) -> np.ndarray:
    if scene.stop_x is not None:
      if self._stop_alpha.x < 0.02:
        self._stop_x.x = scene.stop_x  # appear in place rather than slide in
      self._stop_x.update(scene.stop_x)
    alpha = self._stop_alpha.update(1.0 if scene.stop_x is not None else 0.0)
    return _f32(self._stop_x.x, alpha if alpha > 0.01 else 0.0, 0.0, 0.0)

  # ---- vehicles

  def _bodies(self, rig: Rig, scene: Scene, style: SceneStyle, blink_on: bool) -> list[Body]:
    """Our car first (when the angle shows it), then the leads."""
    signals = scene.signals
    night = float(np.clip((scene.darkness - 0.3) / 0.4, 0.0, 1.0))
    tail = 0.35 * night
    bodies = []
    if rig.show_ego:
      _, length, _, _ = VEHICLE_MODELS[style.ego.model]
      center = np.array([RADAR_TO_CAMERA - length / 2, 0.0, scene.height])
      lights = _f32(night, 1.0 if signals.braking else tail, float(signals.left_blinker and blink_on),
                    float(signals.right_blinker and blink_on))
      bodies.append(Body(center, 0.0, style.ego, lights, reverse=float(signals.reverse),
                         tail_glow=0.6 * night + (0.5 if signals.braking else 0.0) * max(night, 0.25)))
    for lead in scene.leads:
      _, length, _, _ = VEHICLE_MODELS[style.lead.model]
      forward = np.array([math.cos(lead.heading), math.sin(lead.heading), 0.0])
      center = lead.rear + forward * (length / 2)
      lights = _f32(night, 1.0 if lead.braking else tail, 0.0, 0.0)
      bodies.append(Body(center, lead.heading, style.lead, lights,
                         tail_glow=0.6 * night + (0.5 if lead.braking else 0.0) * max(night, 0.25)))
    return bodies

  def _draw_vehicles(self, rect: rl.Rectangle, camera: VirtualCamera, scene: Scene, bodies: list[Body]) -> None:
    program = self._vehicle
    if program is None or not program.valid or not bodies:
      return
    r = (rect.x, rect.y, rect.width, rect.height)
    sky = _vec4(self.palette.sky_horizon)
    pixel = 1.0 / max(camera.focal, 1.0)
    items = []
    for body in bodies:
      box = self._screen_box(camera, body, r)
      if box is not None:
        depth = float(camera.homogeneous(body.center[None, :])[0, 2])
        items.append((depth, body, box))
    if not items:
      return
    rl.begin_shader_mode(program.shader)
    program.set("uvToRay", np.ascontiguousarray(camera.uv_to_ray(r).T, dtype=np.float32), UNIFORM_VEC3, 3)
    program.set("camPos", camera.position.astype(np.float32), UNIFORM_VEC3)
    program.set("sky", sky, UNIFORM_VEC4)
    wheel_turn = 0.0
    for _depth, body, (x0, y0, x1, y1) in sorted(items, key=lambda item: -item[0]):  # far to near
      model, _, _, radius = VEHICLE_MODELS[body.look.model]
      wheel_turn = (self._odometer / radius) % (2 * math.pi)
      program.set("quad", _f32((x0 - r[0]) / r[2], (y0 - r[1]) / r[3], (x1 - x0) / r[2], (y1 - y0) / r[3]), UNIFORM_VEC4)
      program.set("pose", _f32(body.center[0], body.center[1], body.center[2], body.yaw), UNIFORM_VEC4)
      program.set("model", np.array([model], dtype=np.int32), UNIFORM_INT)
      program.set("paint", _f32(*(c / 255.0 for c in body.look.rgb), GLOSS[body.look.model]), UNIFORM_VEC4)
      program.set("lights", body.lights, UNIFORM_VEC4)
      program.set("extra", _f32(body.reverse, scene.darkness, pixel, wheel_turn), UNIFORM_VEC4)
      rl.draw_texture_pro(self._white, rl.Rectangle(0, 0, 1, 1), rl.Rectangle(x0, y0, x1 - x0, y1 - y0), rl.Vector2(0, 0),
                          0.0, rl.WHITE)
      rl.rl_draw_render_batch_active()  # each vehicle has its own uniforms
    rl.end_shader_mode()

  @staticmethod
  def _screen_box(camera: VirtualCamera, body: Body, rect) -> tuple[float, float, float, float] | None:
    """Screen rectangle covering the vehicle and the glow of its lamps, clipped to the layer; None when off screen."""
    _, length, width, _ = VEHICLE_MODELS[body.look.model]
    height = VEHICLE_HEIGHT[body.look.model]
    hl, hw = length / 2 + 0.5, width / 2 + 0.5  # room for lamp glow
    c, s = math.cos(body.yaw), math.sin(body.yaw)
    corners = np.array([[sx * hl, sy * hw] for sx in (-1, 1) for sy in (-1, 1)])
    xy = body.center[:2] + np.column_stack((corners[:, 0] * c - corners[:, 1] * s, corners[:, 0] * s + corners[:, 1] * c))
    points = np.vstack((np.column_stack((xy, np.full(4, body.center[2] + 0.3))),
                        np.column_stack((xy, np.full(4, body.center[2] - height - 0.4)))))
    screen, depth = camera.project(points)
    x, y, w, h = rect
    if (depth < 0.3).all():
      return None
    if (depth < 0.3).any():
      return x, y, x + w, y + h  # partly behind the camera: let the shader find it anywhere
    pad = 12.0  # px, minimum glow around small, distant vehicles
    x0, y0 = screen.min(axis=0) - pad
    x1, y1 = screen.max(axis=0) + pad
    x0, y0, x1, y1 = max(x0, x), max(y0, y), min(x1, x + w), min(y1, y + h)
    if x1 - x0 < 1 or y1 - y0 < 1:
      return None
    return float(x0), float(y0), float(x1), float(y1)

  # ---- stop line glow wall

  def _draw_stop_curtain(self, rect: rl.Rectangle, camera: VirtualCamera, scene: Scene) -> None:
    """A low glowing wall standing on the stop line, so it reads from behind as well as from above. It fades out as
    the stop gets close (stopped at it, the line on the road is enough)."""
    x = float(self._stop_x.x)
    alpha = float(self._stop_alpha.x) * float(np.clip((x - RADAR_TO_CAMERA - 5.0) / 10.0, 0.0, 1.0))
    if alpha <= 0.02 or scene.lane_samples is None:
      return
    from openpilot.system.ui.lib.shader_polygon import Gradient, draw_polygon
    lanes = [float(np.interp(x, SAMPLE_X, scene.lane_samples[:, i])) for i in (1, 2)]
    lo, hi = min(lanes), max(lanes)
    if not 2.0 <= hi - lo <= 6.0:
      center = float(np.interp(x, SAMPLE_X, scene.road_samples[:, 2]))
      lo, hi = center - 1.8, center + 1.8
    ground, top = scene.height, scene.height - 0.7
    quad = np.array([[x, lo - 0.15, ground], [x, hi + 0.15, ground], [x, hi + 0.15, top], [x, lo - 0.15, top]])
    screen, depth = camera.project(quad)
    if (depth < 0.3).any() or not np.isfinite(screen).all():
      return
    bottom = screen[:2].mean(axis=0)
    upper = screen[2:].mean(axis=0)
    if np.linalg.norm(upper - bottom) < 3.0:
      return  # edge-on
    polygon = np.ascontiguousarray(screen[[0, 3, 2, 1]], dtype=np.float32)  # left side up, then the right side down
    start = ((bottom[0] - rect.x) / rect.width, (bottom[1] - rect.y) / rect.height)
    end = ((upper[0] - rect.x) / rect.width, (upper[1] - rect.y) / rect.height)
    pulse = 0.85 + 0.15 * math.sin(self._time * 4.5)
    colors = [rl.Color(255, 40, 30, int(150 * alpha * pulse)), rl.Color(255, 40, 30, int(45 * alpha * pulse)), rl.Color(255, 40, 30, 0)]
    draw_polygon(rect, polygon, gradient=Gradient(start=start, end=end, colors=colors, stops=[0.0, 0.35, 1.0]))

