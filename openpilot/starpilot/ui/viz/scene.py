"""Snapshot of what the scene views draw, built from SubMaster services.

Nothing here touches raylib or messaging directly: ``SceneTracker.update`` reads a SubMaster-like object, so drive
replays, previews and tests can feed it the same way the live UI does.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HEIGHT_DEFAULT = 1.22    # m, camera height before calibration (calibrationd.HEIGHT_INIT)
RADAR_TO_CAMERA = 1.52   # m, radarState distances are measured from about this far ahead of the camera
EGO_RADIUS = 0.9         # m, sphere standing in for the car
LEAD_RADIUS = 0.9        # m
SAME_LEAD_DISTANCE = 4.0  # m, leadTwo closer than this to leadOne is the same vehicle
BEHIND_EXTENSION = 15.0   # m, lane lines and road edges continue this far behind the camera


def _empty() -> np.ndarray:
  return np.empty((0, 3), dtype=np.float64)


@dataclass(frozen=True)
class Lead:
  center: np.ndarray        # sphere centre in the calib frame
  d_rel: float              # m from the front bumper
  v_rel: float              # m/s, negative when closing
  radar: bool
  primary: bool             # leadOne


@dataclass
class Scene:
  valid: bool = False
  revision: int = 0         # bumps whenever the geometry below is rebuilt; renderers key their caches on it
  height: float = HEIGHT_DEFAULT
  path: np.ndarray = field(default_factory=_empty)              # driven path on the road surface
  lane_lines: list[np.ndarray] = field(default_factory=list)    # far left, left, right, far right
  lane_probs: np.ndarray = field(default_factory=lambda: np.zeros(4))
  road_edges: list[np.ndarray] = field(default_factory=list)
  road_edge_stds: np.ndarray = field(default_factory=lambda: np.ones(2))
  path_accel: np.ndarray = field(default_factory=lambda: np.zeros(0))
  leads: list[Lead] = field(default_factory=list)
  ego_center: np.ndarray = field(default_factory=lambda: np.array([RADAR_TO_CAMERA - EGO_RADIUS, 0.0, HEIGHT_DEFAULT - EGO_RADIUS]))
  v_ego: float = 0.0
  experimental: bool = False
  allow_throttle: bool = True
  model_frame: int = -1

  def ground_z(self, x: float) -> float:
    """Road surface height (calib z) under the path at longitudinal distance x."""
    if len(self.path) == 0:
      return self.height
    return float(np.interp(x, self.path[:, 0], self.path[:, 2]))


def _xyz(line) -> np.ndarray:
  return np.column_stack((np.asarray(line.x, dtype=np.float64), np.asarray(line.y, dtype=np.float64),
                          np.asarray(line.z, dtype=np.float64)))


def _extend_back(points: np.ndarray, distance: float = BEHIND_EXTENSION) -> np.ndarray:
  """The model starts its lines at the camera; continue them straight back so views from behind the car stay closed."""
  if len(points) == 0:
    return points
  start = points[0]
  return np.vstack(([start[0] - distance, start[1], start[2]], points))


class SceneTracker:
  """Rebuilds the Scene only when the model, radar or calibration message changes."""

  def __init__(self):
    self.scene = Scene()
    self._keys: tuple = ()
    self._revision = 0

  def update(self, sm, started_frame: int = 0) -> Scene:
    # Reading sm.updated keeps replay's consumed-update bookkeeping identical to the stock model renderer.
    for name in ("modelV2", "radarState", "extrinsicsCalibration", "carState"):
      _ = sm.updated[name]
    if sm.recv_frame["modelV2"] < started_frame or not sm.seen["modelV2"]:
      self._revision += 1
      self.scene = Scene(revision=self._revision)
      self._keys = ()
      return self.scene

    # Geometry changes at model rate (20 Hz); the scalars below are cheap and refreshed every call.
    keys = (sm.recv_frame["modelV2"], sm.recv_frame["radarState"], sm.recv_frame["extrinsicsCalibration"])
    if keys != self._keys or not self.scene.valid:
      self._keys = keys
      self._revision += 1
      self.scene = self._geometry(sm)
      self.scene.revision = self._revision

    scene = self.scene
    scene.v_ego = float(sm["carState"].vEgo) if sm.seen["carState"] else 0.0
    scene.experimental = bool(sm["selfdriveState"].experimentalMode) if sm.seen["selfdriveState"] else False
    scene.allow_throttle = bool(sm["longitudinalPlan"].allowThrottle) if sm.seen["longitudinalPlan"] else True
    return scene

  def _geometry(self, sm) -> Scene:
    calib = sm["extrinsicsCalibration"]
    height = float(calib.height[0]) if len(calib.height) else HEIGHT_DEFAULT
    model = sm["modelV2"]
    scene = Scene(valid=True, height=height)

    path = _xyz(model.position)
    path[:, 2] += height  # modelV2.position follows the camera; drop it onto the road
    scene.path = path
    scene.lane_lines = [_extend_back(_xyz(line)) for line in list(model.laneLines)[:4]]
    scene.lane_probs = np.asarray(model.laneLineProbs, dtype=np.float64)[:4]
    scene.road_edges = [_extend_back(_xyz(edge)) for edge in list(model.roadEdges)[:2]]
    scene.road_edge_stds = np.asarray(model.roadEdgeStds, dtype=np.float64)[:2]
    scene.path_accel = np.asarray(model.acceleration.x, dtype=np.float64)
    scene.model_frame = int(sm.recv_frame["modelV2"])
    scene.ego_center = np.array([RADAR_TO_CAMERA - EGO_RADIUS, 0.0, scene.ground_z(0.0) - EGO_RADIUS])
    if sm.seen["radarState"] and sm.valid["radarState"]:
      scene.leads = self._leads(sm["radarState"], scene)
    return scene

  @staticmethod
  def _leads(radar_state, scene: Scene) -> list[Lead]:
    leads: list[Lead] = []
    for primary, data in ((True, radar_state.leadOne), (False, radar_state.leadTwo)):
      if not data.present:
        continue
      d_rel, y_rel = float(data.dRel), float(data.yRel)
      if not primary and leads and abs(d_rel - leads[0].d_rel) < SAME_LEAD_DISTANCE:
        continue
      x = RADAR_TO_CAMERA + d_rel + LEAD_RADIUS  # back of the sphere at the lead's rear bumper
      center = np.array([x, -y_rel, scene.ground_z(x) - LEAD_RADIUS])  # radar y is left-positive
      leads.append(Lead(center=center, d_rel=d_rel, v_rel=float(data.vRel), radar=bool(data.radar), primary=primary))
    return leads
