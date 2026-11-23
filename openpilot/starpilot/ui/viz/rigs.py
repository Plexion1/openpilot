"""Camera angles. A rig turns the current scene and the onroad layer rectangle into a VirtualCamera.

To add an angle: subclass Rig, add a ViewAngle in options.py and register an instance in RIGS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from openpilot.starpilot.ui.viz.geometry import VirtualCamera, look_at, pinhole, top_down_rotation
from openpilot.starpilot.ui.viz.options import ViewAngle
from openpilot.starpilot.ui.viz.scene import EGO_RADIUS, Scene

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


@dataclass
class RigContext:
  rect: tuple[float, float, float, float]
  scene: Scene
  front: VirtualCamera  # the real road camera as the stock front view shows it
  dt: float             # seconds since the previous frame, for smoothing


class Rig:
  angle: ViewAngle
  show_ego = True           # draw the sphere standing in for the car
  lane_half_width = 0.08    # m; the stock 2.5 cm lines vanish when seen from above
  edge_half_width = 0.10
  reprojection_stream = "wide"  # camera stream for camera/hybrid modes: "wide", "narrow" or None (keep the stock choice)

  def reset(self) -> None:
    pass

  def camera(self, ctx: RigContext) -> VirtualCamera:
    raise NotImplementedError


class FrontRig(Rig):
  angle = ViewAngle.FRONT
  show_ego = False
  lane_half_width = 0.05
  edge_half_width = 0.06
  reprojection_stream = None

  def camera(self, ctx: RigContext) -> VirtualCamera:
    return ctx.front


class ChaseRig(Rig):
  """Above and behind the car, locked to its heading like a racing game camera."""
  angle = ViewAngle.CHASE
  back = 7.5         # m behind the car sphere
  height = 3.6       # m above the road
  look_ahead = 14.0  # m ahead of the car, on the road
  fov = 44.0         # vertical, degrees

  def camera(self, ctx: RigContext) -> VirtualCamera:
    ego = ctx.scene.ego_center
    ground = ego[2] + EGO_RADIUS
    eye = np.array([ego[0] - self.back, 0.0, ground - self.height])
    target = np.array([ego[0] + self.look_ahead, 0.0, ground])
    return VirtualCamera(eye, look_at(eye, target), pinhole(ctx.rect, self.fov, center=(0.5, 0.5)))


class TopDownRig(Rig):
  """Straight down, car near the bottom; zooms out with speed so the lead stays in view."""
  angle = ViewAngle.TOP
  altitude = 60.0      # m; high and narrow so it reads as a map rather than a wide-angle shot
  behind = 6.0         # m of road shown behind the car
  min_ahead, max_ahead = 20.0, 55.0
  zoom_time = 1.5      # s, smoothing time constant for zoom changes
  lane_half_width = 0.09
  edge_half_width = 0.11

  def __init__(self):
    self._ahead: float | None = None

  def reset(self) -> None:
    self._ahead = None

  def ahead(self, scene: Scene, dt: float) -> float:
    target = 12.0 + 1.2 * max(scene.v_ego, 0.0)
    if scene.leads:
      target = max(target, scene.leads[0].center[0] + 6.0)
    target = float(np.clip(target, self.min_ahead, self.max_ahead))
    if self._ahead is None:
      self._ahead = target
    else:
      self._ahead += (target - self._ahead) * (1.0 - math.exp(-max(dt, 0.0) / self.zoom_time))
    return self._ahead

  def camera(self, ctx: RigContext) -> VirtualCamera:
    x, y, w, h = ctx.rect
    ego = ctx.scene.ego_center
    ground = ego[2] + EGO_RADIUS
    ahead = self.ahead(ctx.scene, ctx.dt)
    span = ahead + self.behind
    f = h * self.altitude / span
    eye = np.array([ego[0] + (ahead - self.behind) / 2.0, 0.0, ground - self.altitude])
    intrinsics = np.array([[f, 0.0, x + w / 2.0], [0.0, f, y + h / 2.0], [0.0, 0.0, 1.0]])
    return VirtualCamera(eye, top_down_rotation(), intrinsics)


RIGS: dict[ViewAngle, Rig] = {rig.angle: rig for rig in (FrontRig(), ChaseRig(), TopDownRig())}
