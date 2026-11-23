"""Projection math for the scene views. Pure numpy, no raylib, so it can be unit tested anywhere.

Frames used throughout the package:

* calib  - origin at the road camera, x forward, y right, z down (the modelV2 frame).
           The road surface sits at z = +height (camera height above the road).
* view   - camera axes: x right, y down, z forward (openpilot's view frame).
* screen - pixels of the onroad camera layer, origin top-left (the layer's local coordinates).
"""
from __future__ import annotations

import math

import numpy as np

UP = np.array([0.0, 0.0, -1.0])  # calib frame is z-down
NEAR_PLANE = 0.3  # m, geometry closer to a virtual camera than this is clipped


class VirtualCamera:
  """A pinhole camera placed in the calib frame.

  ``rotation`` is view_from_calib: its rows are the camera's right, down and forward axes expressed in the calib frame.
  ``intrinsics`` maps view coordinates to layer pixels and must have [0, 0, 1] as its last row, so the homogeneous
  w of a projected point is its depth along the camera axis.
  """
  __slots__ = ("position", "rotation", "intrinsics", "_proj", "_offset")

  def __init__(self, position, rotation, intrinsics):
    self.position = np.asarray(position, dtype=np.float64).reshape(3)
    self.rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    self.intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    self._proj = self.intrinsics @ self.rotation
    self._offset = self._proj @ self.position

  @property
  def focal(self) -> float:
    return float(self.intrinsics[1, 1])

  def homogeneous(self, points: np.ndarray) -> np.ndarray:
    """(N, 3) calib points -> (N, 3) homogeneous screen points; column 2 is depth."""
    return points @ self._proj.T - self._offset

  def project(self, points: np.ndarray, near: float = NEAR_PLANE) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) calib points -> ((N, 2) screen points, (N,) depth). Points closer than ``near`` come back as NaN."""
    h = self.homogeneous(np.asarray(points, dtype=np.float64))
    depth = h[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
      xy = h[:, :2] / depth[:, None]
    xy[depth < near] = np.nan
    return xy, depth

  def project_point(self, point, near: float = NEAR_PLANE) -> tuple[float, float, float] | None:
    """One calib point -> (x, y, depth) or None when it is behind the near plane."""
    h = self._proj @ np.asarray(point, dtype=np.float64) - self._offset
    if h[2] < near:
      return None
    return float(h[0] / h[2]), float(h[1] / h[2]), float(h[2])

  def uv_to_ray(self, rect: tuple[float, float, float, float]) -> np.ndarray:
    """3x3 matrix taking (u, v, 1) over ``rect`` (u, v in 0..1) to an unnormalised calib-frame ray direction."""
    x, y, w, h = rect
    screen_from_uv = np.array([[w, 0.0, x], [0.0, h, y], [0.0, 0.0, 1.0]])
    return self.rotation.T @ np.linalg.inv(self.intrinsics) @ screen_from_uv


def look_at(eye, target, up=UP) -> np.ndarray:
  """view_from_calib rotation for a camera at ``eye`` looking at ``target``."""
  forward = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
  forward /= np.linalg.norm(forward)
  right = np.cross(forward, up)
  norm = np.linalg.norm(right)
  if norm < 1e-6:  # looking straight up or down: keep +x (car forward) as screen up
    right = np.array([0.0, 1.0, 0.0])
  else:
    right /= norm
  down = np.cross(forward, right)
  return np.stack((right, down, forward))


def top_down_rotation() -> np.ndarray:
  """Straight down, car forward towards the top of the screen."""
  return np.array([[0.0, 1.0, 0.0],    # screen right = car right
                   [-1.0, 0.0, 0.0],   # screen down = car backward
                   [0.0, 0.0, 1.0]])   # looking along +z (down)


def pinhole(rect: tuple[float, float, float, float], vertical_fov_deg: float, center=(0.5, 0.5)) -> np.ndarray:
  """Intrinsics filling ``rect`` with the given vertical field of view; ``center`` is the principal point in rect units."""
  x, y, w, h = rect
  f = (h / 2.0) / math.tan(math.radians(vertical_fov_deg) / 2.0)
  return np.array([[f, 0.0, x + w * center[0]], [0.0, f, y + h * center[1]], [0.0, 0.0, 1.0]])


def offset_line(points: np.ndarray, half_width: float) -> tuple[np.ndarray, np.ndarray]:
  """Left/right edges of a ground ribbon around a near-to-far polyline, offset perpendicular to its heading."""
  n = len(points)
  if n < 2:
    return points.copy(), points.copy()
  tangent = np.empty((n, 2))
  tangent[1:-1] = points[2:, :2] - points[:-2, :2]
  tangent[0] = points[1, :2] - points[0, :2]
  tangent[-1] = points[-1, :2] - points[-2, :2]
  norm = np.linalg.norm(tangent, axis=1, keepdims=True)
  tangent = np.divide(tangent, norm, out=np.tile([1.0, 0.0], (n, 1)), where=norm > 1e-6)
  normal = np.zeros((n, 3))
  normal[:, 0] = -tangent[:, 1]  # rotate the heading +90 degrees towards the car's right (+y)
  normal[:, 1] = tangent[:, 0]
  left = points - normal * half_width
  right = points + normal * half_width
  return left, right


def project_ribbon(camera: VirtualCamera, left: np.ndarray, right: np.ndarray, near: float = NEAR_PLANE) -> np.ndarray | None:
  """Project a ribbon given as two near-to-far chains into a drawable polygon [L0..Lk, Rk..R0].

  Cross-sections behind the near plane are dropped and the first visible one is moved onto the near plane, so a
  ribbon that starts behind the camera still reaches the screen edge. Returns float32 (M, 2) or None.
  """
  n = len(left)
  if n < 2:
    return None
  h = camera.homogeneous(np.concatenate((left, right)))
  depth_l, depth_r = h[:n, 2], h[n:, 2]
  visible = (depth_l >= near) & (depth_r >= near)
  if not visible.any():
    return None
  first = int(np.argmax(visible))
  # Keep the leading visible run; later dips behind the camera would fold the strip.
  run = visible[first:]
  last = first + (int(np.argmin(run)) if not run.all() else len(run))
  hl, hr = h[:n][first:last], h[n:][first:last]
  if first > 0:
    hl = np.vstack((_near_crossing(h[first - 1], h[first], near), hl))
    hr = np.vstack((_near_crossing(h[n + first - 1], h[n + first], near), hr))
  if len(hl) < 2:
    return None
  sl = hl[:, :2] / hl[:, 2:3]
  sr = hr[:, :2] / hr[:, 2:3]
  return np.ascontiguousarray(np.vstack((sl, sr[::-1])), dtype=np.float32)


def _near_crossing(behind: np.ndarray, ahead: np.ndarray, near: float) -> np.ndarray:
  if behind[2] >= near:
    return behind
  t = (near - behind[2]) / max(ahead[2] - behind[2], 1e-9)
  return behind + (ahead - behind) * t


def ground_disc(center, radius: float, segments: int = 20) -> np.ndarray:
  """Points of a horizontal circle on the ground (z constant) in triangle-strip order for a convex fill."""
  angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
  ring = np.empty((segments, 3))
  ring[:, 0] = center[0] + radius * np.cos(angles)
  ring[:, 1] = center[1] + radius * np.sin(angles)
  ring[:, 2] = center[2]
  # zig-zag: 0, 1, n-1, 2, n-2, ... fills a convex polygon as one strip
  order = [0]
  lo, hi = 1, segments - 1
  while lo <= hi:
    order.append(lo)
    if lo != hi:
      order.append(hi)
    lo += 1
    hi -= 1
  return ring[order]


def interp_line(points: np.ndarray, x: float) -> np.ndarray | None:
  """Point on a near-to-far polyline at longitudinal distance ``x`` (clamped to its ends)."""
  if len(points) == 0:
    return None
  xs = points[:, 0]
  return np.array([x, np.interp(x, xs, points[:, 1]), np.interp(x, xs, points[:, 2])])


def clip_line(points: np.ndarray, max_x: float) -> np.ndarray:
  """Trim a near-to-far polyline at ``max_x``, interpolating the end point."""
  if len(points) < 2 or points[-1, 0] <= max_x:
    return points
  keep = points[:, 0] < max_x
  end = interp_line(points, max_x)
  return np.vstack((points[keep], end[None, :]))
