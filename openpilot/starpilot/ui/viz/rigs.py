"""Camera angles. A rig turns the current scene and the onroad layer rectangle into a VirtualCamera.

To add an angle: subclass Rig, add a ViewAngle in options.py and register an instance in RIGS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from openpilot.starpilot.ui.viz.geometry import VirtualCamera, look_at, pinhole, top_down_rotation
from openpilot.starpilot.ui.viz.options import ViewAngle
from openpilot.starpilot.ui.viz.scene import Scene

INF_POINT = np.array([1000.0, 0.0, 0.0])


def stock_front_camera(intrinsic: np.ndarray, view_from_calib: np.ndarray, rect: tuple[float, float, float, float],
                       wide: bool) -> VirtualCamera:
  """The road camera exactly as AugmentedRoadView frames it (same zoom and vanishing-point centring)."""
  x, y, w, h = rect
  zoom = 2.0 if wide else 1.1
  calib_transform = intrinsic @ view_from_calib
  kep = calib_transform @ INF_POINT
  cx, cy = intrinsic[0, 2], intrinsic[1, 2]
  zoom = max(zoom, w / (2 * cx), h / (2 * cy))
  margin = 5
  max_x_offset = max(0.0, cx * zoom - w / 2 - margin)
  max_y_offset = max(0.0, cy * zoom - h / 2 - margin)
  x_offset = y_offset = 0.0
  if abs(kep[2]) > 1e-6:
    x_offset = float(np.clip((kep[0] / kep[2] - cx) * zoom, -max_x_offset, max_x_offset))
    y_offset = float(np.clip((kep[1] / kep[2] - cy) * zoom, -max_y_offset, max_y_offset))
  video_transform = np.array([
    [zoom, 0.0, (w / 2 + x - x_offset) - (cx * zoom)],
    [0.0, zoom, (h / 2 + y - y_offset) - (cy * zoom)],
    [0.0, 0.0, 1.0],
  ])
  return VirtualCamera(np.zeros(3), view_from_calib, video_transform @ intrinsic)


def _smooth(current: float | None, target: float, dt: float, time_constant: float) -> float:
  if current is None:
    return target
  return current + (target - current) * (1.0 - math.exp(-max(dt, 0.0) / time_constant))


@dataclass
class RigContext:
  rect: tuple[float, float, float, float]
  scene: Scene
  front: VirtualCamera  # the real road camera as the stock front view shows it
  dt: float             # seconds since the previous frame, for smoothing


class Rig:
  angle: ViewAngle
  show_ego = True           # draw our own car
  lane_half_width = 0.08    # m; the stock 2.5 cm lines vanish when seen from above
  edge_half_width = 0.07
  path_half_width = 0.9     # m, stock path width
  # Camera stream for camera/hybrid modes: "narrow" (the main road camera), "wide", or None (StarPilot's own choice).
  # Chase and top reproject only the main road camera: it is sharper, better in low light and sees no interior.
  reprojection_stream: str | None = "narrow"
  stop_curtain = True       # draw the stop line's glow wall; it is edge-on from straight above
  upright_signs = True      # stand signs on posts; from straight above they lie flat like map markers

  def reset(self) -> None:
    pass

  def camera(self, ctx: RigContext) -> VirtualCamera:
    raise NotImplementedError


class FrontRig(Rig):
  angle = ViewAngle.FRONT
  show_ego = False
  lane_half_width = 0.06
  edge_half_width = 0.07
  reprojection_stream = None

  def camera(self, ctx: RigContext) -> VirtualCamera:
    return ctx.front


class ChaseRig(Rig):
  """Above and behind the car, locked to its heading like a racing game camera; eases back a little with speed."""
  angle = ViewAngle.CHASE
  back = 7.6          # m behind the car's centre at a standstill
  back_per_speed = 0.045  # extra m per m/s
  height = 3.5        # m above the road
  look_ahead = 6.0    # m ahead of the car, on the road
  fov = 48.0          # vertical, degrees
  speed_time = 1.2    # s, smoothing for the speed-dependent distance

  def __init__(self):
    self._back: float | None = None

  def reset(self) -> None:
    self._back = None

  def camera(self, ctx: RigContext) -> VirtualCamera:
    ego = ctx.scene.ego_ground
    target_back = self.back + self.back_per_speed * min(max(ctx.scene.v_ego, 0.0), 35.0)
    self._back = _smooth(self._back, target_back, ctx.dt, self.speed_time)
    eye = np.array([ego[0] - self._back, 0.0, ego[2] - self.height - 0.04 * (self._back - self.back)])
    target = np.array([ego[0] + self.look_ahead, 0.0, ego[2]])
    return VirtualCamera(eye, look_at(eye, target), pinhole(ctx.rect, self.fov, center=(0.5, 0.5)))


class TopDownRig(Rig):
  """Straight down, car near the bottom; zooms out with speed so the lead stays in view."""
  angle = ViewAngle.TOP
  altitude = 60.0      # m; high and narrow so it reads as a map rather than a wide-angle shot
  behind = 6.0         # m of road shown behind the car's centre
  min_ahead, max_ahead = 18.0, 55.0
  zoom_time = 1.5      # s, smoothing time constant for zoom changes
  lane_half_width = 0.1
  edge_half_width = 0.09
  stop_curtain = False
  upright_signs = False

  def __init__(self):
    self._ahead: float | None = None

  def reset(self) -> None:
    self._ahead = None

  def ahead(self, scene: Scene, dt: float) -> float:
    target = 12.0 + 0.9 * max(scene.v_ego, 0.0)
    if scene.leads:
      target = max(target, scene.leads[0].rear[0] + 8.0 - scene.ego_ground[0])
    if scene.stop_sign is not None and scene.stop_sign.position[0] > 0.0:
      target = max(target, scene.stop_sign.position[0] + 4.0 - scene.ego_ground[0])
    target = float(np.clip(target, self.min_ahead, self.max_ahead))
    self._ahead = _smooth(self._ahead, target, dt, self.zoom_time)
    return self._ahead

  def camera(self, ctx: RigContext) -> VirtualCamera:
    x, y, w, h = ctx.rect
    ego = ctx.scene.ego_ground
    ahead = self.ahead(ctx.scene, ctx.dt)
    span = ahead + self.behind
    f = h * self.altitude / span
    eye = np.array([ego[0] + (ahead - self.behind) / 2.0, 0.0, ego[2] - self.altitude])
    intrinsics = np.array([[f, 0.0, x + w / 2.0], [0.0, f, y + h / 2.0], [0.0, 0.0, 1.0]])
    return VirtualCamera(eye, top_down_rotation(), intrinsics)


RIGS: dict[ViewAngle, Rig] = {rig.angle: rig for rig in (FrontRig(), ChaseRig(), TopDownRig())}
