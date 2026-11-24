"""Draws one scene view: the ground pass (road, markings, path and everything else on the road) and the vehicles.

CPU work per frame: the ground's inputs are packed into one vec4 array uploaded in a single call (shaders.F_*), the
road samples go up once per model frame and the colours only when the style changes; then one draw call for the
ground and one for all vehicles together (each quad's vertex colour selects its slot in shaders.V_*). The blocks are
preallocated and refilled in place.

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
from openpilot.starpilot.ui.viz.geometry import NEAR_PLANE, VirtualCamera
from openpilot.starpilot.ui.viz.options import SceneStyle, VehicleLook, VehicleModel, VizMode
from openpilot.starpilot.ui.viz.rigs import Rig
from openpilot.starpilot.ui.viz.scene import EGO_LENGTH, RADAR_TO_CAMERA, SAMPLE_MAX_X, SAMPLE_X, Scene

UNIFORM_VEC4 = rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4

# Stock path colours (model_renderer.THROTTLE_COLORS / NO_THROTTLE_COLORS), near and far.
THROTTLE = ((13, 248, 122, 102), (114, 255, 92, 0))
NO_THROTTLE = ((242, 242, 242, 102), (242, 242, 242, 0))
STATUS_GLOW = {"engaged": (0.12, 0.86, 0.45, 0.38), "override": (0.75, 0.78, 0.8, 0.22)}

# Shader model id, size (length, width, height in m), wheel radius and paint gloss of each vehicle model.
VEHICLE_MODELS = {VehicleModel.CAR: (0, (4.6, 1.84, 1.48), 0.34, 0.85), VehicleModel.TRUCK: (1, (5.6, 2.0, 1.95), 0.4, 0.7),
                  VehicleModel.SPHERE: (2, (1.8, 1.8, 1.82), 0.9, 0.9)}
STOP_SIGN_MODEL, STOP_SIGN_SIZE = 3, (0.24, 0.88, 2.75)
STOP_MARKER_MODEL, STOP_MARKER_SIZE = 4, (1.9, 1.9, 0.15)
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


def _unit(rgba) -> tuple[float, float, float, float]:
  return rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0, rgba[3] / 255.0


def _hex(color: str | None) -> tuple[int, int, int, int] | None:
  if not isinstance(color, str) or len(color) != 9:
    return None
  try:
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5, 7))
  except ValueError:
    return None


class _Program:
  """A shader with its uniform locations; remembers which cached values it already holds."""

  def __init__(self, fragment: str, names: tuple[str, ...]):
    self.shader = rl.load_shader_from_memory(shaders.VERTEX_SHADER, fragment)
    # raylib falls back to its default shader when compilation fails; treat that as unusable rather than draw garbage
    self.valid = bool(self.shader.id) and self.shader.id != rl.rl_get_shader_id_default()
    self.loc = {name: rl.get_shader_location(self.shader, name) for name in names} if self.valid else {}
    self.keys: dict[str, object] = {}

  def set_vec4s(self, name: str, values: np.ndarray) -> None:
    """Upload a (N, 4) float32 array to a vec4 array uniform in one call."""
    loc = self.loc.get(name, -1)
    if loc >= 0:
      rl.set_shader_value_v(self.shader, loc, values, UNIFORM_VEC4, len(values))

  def changed(self, group: str, key) -> bool:
    """True (and remembered) when ``key`` differs from what this program last received for ``group``."""
    if self.keys.get(group) == key:
      return False
    self.keys[group] = key
    return True

  def close(self) -> None:
    if self.valid:
      rl.unload_shader(self.shader)


GROUND_UNIFORMS = ("frameData", "look", "laneY", "roadY")
CAMERA_UNIFORMS = ("texture1",)
VEHICLE_UNIFORMS = ("vehicleData",)

# Corners of a body's bounding box (bottom four, then top four) for its screen rectangle
_CORNER_X = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
_CORNER_Y = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_CORNER_TOP = np.array([False, False, False, False, True, True, True, True])


@dataclass
class Body:
  """Something the vehicle shader draws: a vehicle (with a shadow and lights) or a roadside prop."""
  center: tuple         # footprint centre on the road, calib frame (foot of the post for a sign)
  yaw: float
  model: int            # shader model id
  size: tuple           # length, width, height (m)
  rgb: tuple
  gloss: float
  lights: tuple         # headlights, tail/brake, left, right (a sign's opacity in x)
  wheel_radius: float = 0.34
  reverse: float = 0.0
  tail_glow: float = 0.0
  shadow: bool = True
  round: bool = False   # round footprint (sphere): smaller, softer shadow

  @classmethod
  def vehicle(cls, center: tuple, yaw: float, look: VehicleLook, lights: tuple, **kwargs) -> Body:
    model, size, wheel_radius, gloss = VEHICLE_MODELS[look.model]
    return cls(center, yaw, model, size, look.rgb, gloss, lights, wheel_radius=wheel_radius,
               round=look.model == VehicleModel.SPHERE, **kwargs)

  @property
  def lit(self) -> bool:
    return self.model < STOP_SIGN_MODEL and (max(self.lights) > 0.02 or self.reverse > 0.02)


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
    self._lane_key = None
    self._path_key = None
    self._path = (0.0, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
    # Uniform blocks, reused every frame (layouts in shaders.F_* / L_* / V_*)
    self._frame = np.zeros((shaders.FRAME_SLOTS, 4), dtype=np.float32)
    self._look = np.zeros((shaders.LOOK_SLOTS, 4), dtype=np.float32)
    self._vehicle_data = np.zeros((shaders.VEHICLE_SLOTS, 4), dtype=np.float32)
    self._vehicle_data[shaders.V_SKY] = _unit(self.palette.sky_horizon)
    self.last_camera: VirtualCamera | None = None

  # ---- resources

  def _ensure(self) -> None:
    while not self._compile_next():
      pass
    if not self._synthetic.valid:
      raise RuntimeError("scene views: synthetic ground shader failed to compile")

  def _compile_next(self) -> bool:
    """Create the next missing resource (one shader compile at most); True when the always-needed ones exist."""
    if self._white is None:
      image = rl.gen_image_color(1, 1, rl.WHITE)
      self._white = rl.load_texture_from_image(image)
      rl.unload_image(image)
    if self._synthetic is None:
      self._synthetic = _Program(shaders.SYNTHETIC_GROUND_SHADER, GROUND_UNIFORMS)
      return False
    if self._vehicle is None:
      self._vehicle = _Program(shaders.VEHICLE_SHADER, VEHICLE_UNIFORMS)
      if not self._vehicle.valid:
        cloudlog.error("scene views: vehicle shader failed to compile, vehicles are hidden")
      return False
    return True

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

  def warm_up(self, egl: bool | None) -> bool:
    """Compile one missing program per call (they take a while on first use); True once all are ready. ``egl`` None
    skips the camera program (no camera frame format known yet)."""
    if not self._compile_next():
      return False
    if egl is not None and egl not in self._camera:
      self._camera_program(egl)
      return False
    return True

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
    r = (rect.x, rect.y, rect.width, rect.height)
    uv_to_ray = camera.uv_to_ray(r)

    program = self._camera_program(frame.egl) if mode_style.shader_mode and frame is not None and source is not None else None
    bodies = self._bodies(rig, scene, style, blink_on) if scene.valid else []
    props = self._props(scene, rig) if scene.valid else []
    ground = program or self._synthetic
    rl.begin_shader_mode(ground.shader)
    self._ground_uniforms(ground, rig, camera, uv_to_ray, scene, mode_style, road_style, bodies, status)
    if program is not None:
      self._camera_uniforms(program, scene, mode_style, frame, source, color_strength)
    ground.set_vec4s("frameData", self._frame)
    if program is not None:
      src = rl.Rectangle(0, 0, float(frame.texture.width), float(frame.texture.height))
      rl.draw_texture_pro(frame.texture, src, rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
    else:
      rl.draw_texture_pro(self._white, rl.Rectangle(0, 0, 1, 1), rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
    rl.end_shader_mode()
    if not scene.valid:
      return
    if rig.stop_curtain:
      self._draw_stop_curtain(rect, camera, scene)
    self._draw_vehicles(r, camera, uv_to_ray, scene, props + bodies)

  # ---- ground

  def _ground_uniforms(self, program: _Program, rig: Rig, camera: VirtualCamera, uv_to_ray: np.ndarray, scene: Scene,
                       mode_style: ModeStyle, road_style: dict | None, bodies: list[Body], status: str) -> None:
    f = self._frame
    has_road = scene.valid and scene.road_samples is not None and len(scene.road_edges) == 2
    lane_rgb = mode_style.lane_rgb
    inner = _hex((road_style or {}).get("pathEdge")) if lane_rgb is None else None
    outer = _hex((road_style or {}).get("laneLines")) if lane_rgb is None else None

    # colours and constants: only when the style changes
    if program.changed("look", (self.palette, mode_style, rig.path_half_width, inner, outer, has_road)):
      p, look = self.palette, self._look
      white = (*(lane_rgb or (245, 245, 245)), 255)
      look[shaders.L_SKY_TOP] = _unit(p.sky_top)
      look[shaders.L_SKY_HORIZON] = _unit(p.sky_horizon)
      look[shaders.L_GROUND] = _unit(p.ground)
      look[shaders.L_GRID] = _unit(p.grid)
      look[shaders.L_ROAD] = _unit(mode_style.road_surface if has_road else (0, 0, 0, 0))
      look[shaders.L_LANE_INNER] = _unit(inner or white)
      look[shaders.L_LANE_OUTER] = _unit(outer or white)
      look[shaders.L_MISC] = (rig.path_half_width, SAMPLE_MAX_X, p.fog_distance, 0.0)
      look[shaders.L_PAINT_WHITE] = _unit((240, 240, 240, 255))
      look[shaders.L_PAINT_YELLOW] = _unit((250, 200, 40, 255))
      program.set_vec4s("look", look)

    # road samples: once per model frame
    if scene.valid and scene.lane_samples is not None and program.changed("samples", scene.revision):
      program.set_vec4s("laneY", scene.lane_samples)
      program.set_vec4s("roadY", scene.road_samples)
    lane_key = (scene.revision, mode_style, rig.angle, inner is not None or outer is not None)
    if lane_key != self._lane_key:
      self._lane_key = lane_key
      if scene.valid and scene.lane_samples is not None:
        probs = np.zeros(4)
        probs[:len(scene.lane_probs)] = np.clip(scene.lane_probs[:4], 0.0, 1.0)
        stds = np.ones(2)
        stds[:len(scene.road_edge_stds)] = scene.road_edge_stds[:2]
        edges = np.clip(1.0 - stds, 0.0, 1.0) * mode_style.edge_alpha
      else:
        probs, edges = np.zeros(4), np.zeros(2)
      cap = 0.7 if lane_key[3] else 1.0  # StarPilot's custom lane colours cap at 70 %
      f[shaders.F_LANE_ALPHA] = np.minimum(probs, cap) * mode_style.lane_alpha
      f[shaders.F_EDGE] = (edges[0], edges[1], rig.edge_half_width, rig.lane_half_width * mode_style.lane_width_scale)

    # camera
    f[shaders.F_RAY:shaders.F_RAY + 3, :3] = uv_to_ray.T
    f[shaders.F_RAY:shaders.F_RAY + 3, 3] = camera.position
    f[shaders.F_MISC] = (scene.height, self._odometer, self._time % 1000.0, 0.0)
    f[shaders.F_DASH] = self._lane_dashes(scene)

    # path
    path_mode, near, far = self._path_colors(road_style, mode_style)
    start = 0.0 if not rig.show_ego else scene.ego_ground[0] + EGO_LENGTH / 2 - 0.3
    end = scene.path_end if scene.valid and len(scene.path) > 1 and scene.path_end - start >= 1.5 else 0.0
    f[shaders.F_PATH] = (start, end, mode_style.path_alpha if end > start else 0.0, path_mode)
    f[shaders.F_PATH_NEAR] = near
    f[shaders.F_PATH_FAR] = far

    # lane change, stop line, lights
    f[shaders.F_CHANGE] = self._lane_change(scene)
    f[shaders.F_STOP] = self._stop_line(scene)
    darkness = scene.darkness
    f[shaders.F_LIGHT] = (darkness, min(max((darkness - 0.3) / 0.4, 0.0), 1.0), RADAR_TO_CAMERA, 0.0)

    poses = f[shaders.F_POSE:shaders.F_POSE + shaders.MAX_BODIES]
    sizes = f[shaders.F_SIZE:shaders.F_SIZE + shaders.MAX_BODIES]
    sizes[:] = 0.0  # slot 0 empty ends the shader's loop
    for i, body in enumerate([body for body in bodies if body.shadow][:shaders.MAX_BODIES]):
      length, width, _ = body.size
      poses[i] = (body.center[0], body.center[1], math.cos(body.yaw), math.sin(body.yaw))
      shrink = 0.6 if body.round else 1.0
      sizes[i] = (length / 2 * shrink, width / 2 * shrink, 0.55, body.tail_glow)
    glow = STATUS_GLOW.get(status) if rig.show_ego and bodies else None
    f[shaders.F_GLOW] = glow or (0.0, 0.0, 0.0, 0.0)

  def _camera_uniforms(self, program: _Program, scene: Scene, mode_style: ModeStyle, frame: FrameTextures, source: CameraGeometry,
                       color_strength: float) -> None:
    f = self._frame
    k = source.intrinsics
    # road plane (X, Y, 1) -> camera pixels -> texture uv
    ground = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, scene.height]])
    to_uv = np.diag([1.0 / frame.uv_width, 1.0 / frame.uv_height, 1.0])
    f[shaders.F_TEX:shaders.F_TEX + 3, :3] = (to_uv @ k @ source.view_from_calib @ ground).T
    f[shaders.F_CAM_LIMITS] = CAMERA_LIMITS[source.wide]
    f[shaders.F_CAM_MISC] = (mode_style.shader_mode, color_strength, 0.0, 0.0)
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

  def _path_colors(self, road_style: dict | None, mode_style: ModeStyle) -> tuple[float, tuple, tuple]:
    """Shader colour mode and near/far colours following StarPilot's path style (stock throttle colours without one)."""
    if road_style is not None:
      key = (road_style.get("pathMode"), road_style.get("path"))
      if key != self._path_key:
        from openpilot.starpilot.ui.road_colors import path_mode
        self._path_key = key
        chosen = path_mode(road_style)
        if chosen == "rainbow":
          self._path = 3.0, (1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 0.0)
        elif chosen == "acceleration":
          self._path = 2.0, (1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 0.0)
        else:
          color = _hex(road_style.get("path")) or (48, 255, 156, 255)
          self._path = 1.0, _unit((*color[:3], color[3] * 0.75)), _unit((*color[:3], 0))
      return self._path
    blend = min(max(self._throttle.x, 0.0), 1.0)
    near = [a + (b - a) * blend for a, b in zip(NO_THROTTLE[0], THROTTLE[0], strict=True)]
    far = [a + (b - a) * blend for a, b in zip(NO_THROTTLE[1], THROTTLE[1], strict=True)]
    near[3] = 190  # the stock 40 % alpha disappears on a dark synthetic road
    return 0.0, _unit(near), _unit(far)

  def _lane_change(self, scene: Scene) -> tuple[float, float, float, float]:
    target = {1: 0.55 + 0.25 * math.sin(self._time * 5.0), 2: 1.0, 3: 0.35}.get(scene.lane_change, 0.0)
    if scene.lane_change_direction == 0:
      target = 0.0
    else:
      self._change_direction = scene.lane_change_direction
    strength = self._change.update(target)
    signals = scene.signals
    blocked = signals.left_blindspot if self._change_direction < 0 else signals.right_blindspot
    return float(self._change_direction), strength if strength > 0.01 else 0.0, 1.0 if blocked else 0.0, 0.0

  def _stop_line(self, scene: Scene) -> tuple[float, float, float, float]:
    if scene.stop_x is not None:
      if self._stop_alpha.x < 0.02:
        self._stop_x.x = scene.stop_x  # appear in place rather than slide in
      self._stop_x.update(scene.stop_x)
    alpha = self._stop_alpha.update(1.0 if scene.stop_x is not None else 0.0)
    return float(self._stop_x.x), alpha if alpha > 0.01 else 0.0, 0.0, 0.0

  # ---- vehicles

  def _bodies(self, rig: Rig, scene: Scene, style: SceneStyle, blink_on: bool) -> list[Body]:
    """Our car first (when the angle shows it), then the leads."""
    signals = scene.signals
    night = min(max((scene.darkness - 0.3) / 0.4, 0.0), 1.0)
    tail = 0.35 * night
    bodies = []
    if rig.show_ego:
      length = VEHICLE_MODELS[style.ego.model][1][0]
      lights = (night, 1.0 if signals.braking else tail, float(signals.left_blinker and blink_on),
                float(signals.right_blinker and blink_on))
      bodies.append(Body.vehicle((RADAR_TO_CAMERA - length / 2, 0.0, scene.height), 0.0, style.ego, lights,
                                 reverse=float(signals.reverse),
                                 tail_glow=0.6 * night + (0.5 if signals.braking else 0.0) * max(night, 0.25)))
    half = VEHICLE_MODELS[style.lead.model][1][0] / 2
    for lead in scene.leads:
      rear = lead.rear
      center = (float(rear[0]) + math.cos(lead.heading) * half, float(rear[1]) + math.sin(lead.heading) * half, float(rear[2]))
      lights = (night, 1.0 if lead.braking else tail, 0.0, 0.0)
      bodies.append(Body.vehicle(center, lead.heading, style.lead, lights,
                                 tail_glow=0.6 * night + (0.5 if lead.braking else 0.0) * max(night, 0.25)))
    return bodies

  @staticmethod
  def _props(scene: Scene, rig: Rig) -> list[Body]:
    """Roadside objects: the stop sign StarPilot confirmed, facing us (lying flat as a marker in the top view)."""
    sign = scene.stop_sign
    if sign is None or sign.opacity <= 0.01:
      return []
    model, size = (STOP_SIGN_MODEL, STOP_SIGN_SIZE) if rig.upright_signs else (STOP_MARKER_MODEL, STOP_MARKER_SIZE)
    position = sign.position
    return [Body((float(position[0]), float(position[1]), float(position[2])), 0.0, model, size, (0, 0, 0), 0.25,
                 (sign.opacity, 0.0, 0.0, 0.0), shadow=False)]

  def _draw_vehicles(self, r: tuple, camera: VirtualCamera, uv_to_ray: np.ndarray, scene: Scene, bodies: list[Body]) -> None:
    """Every body in one draw call: each quad's vertex colour tells the shader which slot of vehicleData is its own."""
    program = self._vehicle
    if program is None or not program.valid or not bodies:
      return
    items = [(depth, body, box) for body, (depth, box) in zip(bodies, self._screen_boxes(camera, bodies, r), strict=True)
             if box is not None]
    if not items:
      return
    items.sort(key=lambda item: -item[0])  # far to near
    items = items[-shaders.MAX_DRAWN:]
    data = self._vehicle_data
    data[shaders.V_RAY:shaders.V_RAY + 3, :3] = uv_to_ray.T
    data[shaders.V_RAY:shaders.V_RAY + 3, 3] = camera.position
    data[shaders.V_MISC, :3] = (scene.darkness, 1.0 / max(camera.focal, 1.0), 0.0)  # z: the normal loop's start
    x, y, w, h = r
    for slot, (_depth, body, (x0, y0, x1, y1)) in enumerate(items):
      base = shaders.V_BODY + shaders.BODY_SLOTS * slot
      rgb = body.rgb
      data[base:base + shaders.BODY_SLOTS] = (
        ((x0 - x) / w, (y0 - y) / h, (x1 - x0) / w, (y1 - y0) / h),
        (body.center[0], body.center[1], body.center[2], body.yaw),
        (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, body.gloss),
        body.lights,
        (body.reverse, body.model, 0.0, (self._odometer / body.wheel_radius) % (2 * math.pi)))
    rl.begin_shader_mode(program.shader)
    program.set_vec4s("vehicleData", data)
    source, origin = rl.Rectangle(0, 0, 1, 1), rl.Vector2(0, 0)
    for slot, (_depth, _body, (x0, y0, x1, y1)) in enumerate(items):
      rl.draw_texture_pro(self._white, source, rl.Rectangle(x0, y0, x1 - x0, y1 - y0), origin, 0.0, rl.Color(slot, 0, 0, 255))
    rl.end_shader_mode()

  @staticmethod
  def _screen_boxes(camera: VirtualCamera, bodies: list[Body], rect) -> list[tuple[float, tuple | None]]:
    """Per body: its depth and the screen rectangle covering it and the glow of its lamps, clipped to the layer (None
    when off screen). All bodies are projected together."""
    rows = []
    for body in bodies:
      length, width, height = body.size
      glow = 0.5 if body.lit else 0.05  # room for lamp glow only while a lamp is on
      cx, cy, cz = body.center
      rows.append((cx, cy, cz, body.yaw, length / 2 + glow, width / 2 + glow, cz - height - glow))
    p = np.array(rows)
    c, s = np.cos(p[:, 3:4]), np.sin(p[:, 3:4])
    lx, ly = _CORNER_X * p[:, 4:5], _CORNER_Y * p[:, 5:6]
    points = np.empty((len(rows), 8, 3))
    points[..., 0] = p[:, 0:1] + lx * c - ly * s
    points[..., 1] = p[:, 1:2] + lx * s + ly * c
    points[..., 2] = np.where(_CORNER_TOP, p[:, 6:7], p[:, 2:3] + 0.05)
    projected = camera.homogeneous(points.reshape(-1, 3)).reshape(len(rows), 8, 3)
    depth = projected[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
      xy = projected[..., :2] / depth[..., None]
    lo, hi = xy.min(axis=1).tolist(), xy.max(axis=1).tolist()
    behind = (depth < NEAR_PLANE).sum(axis=1).tolist()
    centers = camera.homogeneous(p[:, :3])[:, 2].tolist()
    x, y, w, h = rect
    out = []
    for body, center, count, (x0, y0), (x1, y1) in zip(bodies, centers, behind, lo, hi, strict=True):
      if count == 8:
        out.append((center, None))
        continue
      if count:
        out.append((center, (x, y, x + w, y + h)))  # partly behind the camera: let the shader find it anywhere
        continue
      pad = 8.0 if body.lit else 2.0  # px: edge antialiasing, plus a minimum glow around small, distant vehicles
      x0, y0, x1, y1 = max(x0 - pad, x), max(y0 - pad, y), min(x1 + pad, x + w), min(y1 + pad, y + h)
      out.append((center, (x0, y0, x1, y1) if x1 - x0 >= 1 and y1 - y0 >= 1 else None))
    return out

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

