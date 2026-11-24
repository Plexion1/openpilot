"""Snapshot of what the scene views draw, built from SubMaster services.

Nothing here touches raylib or messaging directly: ``SceneTracker.update`` reads a SubMaster-like object, so drive
replays, previews and tests can feed it the same way the live UI does.

Geometry (lane lines, road edges, path) is rebuilt only when a new model, radar or calibration message arrives and is
resampled once into two small fixed-size arrays that the ground shader reads. The per-frame signals (blinkers,
brakes, darkness) are cheap scalars refreshed on every call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

HEIGHT_DEFAULT = 1.22    # m, camera height before calibration (calibrationd.HEIGHT_INIT)
RADAR_TO_CAMERA = 1.52   # m, radarState distances are measured from about this far ahead of the camera (front bumper)
SAME_LEAD_DISTANCE = 4.0  # m, leadTwo closer than this to leadOne is the same vehicle
EGO_LENGTH = 4.6         # m, nominal car length used to frame the camera rigs

# Road samples handed to the GPU, spaced like the model's own points (dense near the car): x_i = MAX * (i / (N - 1))^2
SAMPLES = 24
SAMPLE_MAX_X = 150.0
SAMPLE_X = SAMPLE_MAX_X * (np.arange(SAMPLES) / (SAMPLES - 1)) ** 2

# modelV2.meta laneChangeState / laneChangeDirection names
LANE_CHANGE_STATES = {"off": 0, "preLaneChange": 1, "laneChangeStarting": 2, "laneChangeFinishing": 3}
LANE_CHANGE_DIRECTIONS = {"none": 0, "left": -1, "right": 1}

STOP_SPEED = 0.5          # m/s; where the model's predicted speed falls below this is where it wants to stop
STOP_MIN_AHEAD = 1.5      # m in front of the bumper, so the line stays visible while holding at it

# Darkness from the road camera exposure, like ui_state.light_sensor (100 - exposure); hysteresis keeps it steady
DARK_EXPOSURE, LIGHT_EXPOSURE = 70.0, 50.0
DARKNESS_TIME = 2.0       # s, fade between the day and night looks

# The model has no brake light output for other cars: infer them from firm deceleration or being stopped
LEAD_BRAKE_ON, LEAD_BRAKE_OFF = -0.7, -0.25   # m/s^2, radarState aLeadK
LEAD_STOPPED = 0.4        # m/s
LEAD_BRAKE_HOLD = 0.6     # s; vision accelerations are noisy, so the lights stay on at least this long
EGO_BRAKE_DECEL = -1.0    # m/s^2; our brake lights also come on under firm (ACC) deceleration

# Stop signs come from StarPilot's own confirmed STOP sighting (the same one that shows its STOP icon)
STOP_SIGN_KEEP = 6.0      # s; a placed sign stays this long after the last sighting (it leaves the camera's view up close)
STOP_SIGN_MOVE = 8.0      # m; a sighting this far from the placed sign is a different sign


def _empty() -> np.ndarray:
  return np.empty((0, 3), dtype=np.float64)


@dataclass(frozen=True)
class Lead:
  rear: np.ndarray          # centre of the rear bumper on the road, calib frame
  heading: float            # rad, yaw of the road at the lead (positive turns right)
  d_rel: float              # m from our front bumper
  v_rel: float              # m/s, negative when closing
  v_lead: float             # m/s
  a_lead: float             # m/s^2
  braking: bool             # inferred brake lights
  radar: bool
  primary: bool             # leadOne


@dataclass(frozen=True)
class StopSign:
  position: np.ndarray      # foot of the post on the road, calib frame
  opacity: float


@dataclass
class Signals:
  left_blinker: bool = False
  right_blinker: bool = False
  left_blindspot: bool = False
  right_blindspot: bool = False
  braking: bool = False
  reverse: bool = False
  standstill: bool = False


@dataclass
class Scene:
  valid: bool = False
  revision: int = 0         # bumps whenever the geometry is rebuilt; renderers key their caches on it
  height: float = HEIGHT_DEFAULT
  path: np.ndarray = field(default_factory=_empty)              # driven path on the road surface
  lane_lines: list[np.ndarray] = field(default_factory=list)    # far left, left, right, far right
  lane_probs: np.ndarray = field(default_factory=lambda: np.zeros(4))
  road_edges: list[np.ndarray] = field(default_factory=list)
  road_edge_stds: np.ndarray = field(default_factory=lambda: np.ones(2))
  path_accel: np.ndarray = field(default_factory=lambda: np.zeros(0))
  # (SAMPLES, 4) float32 at SAMPLE_X: the 4 lane lines' y, and [left edge y, right edge y, path y, path acceleration]
  lane_samples: np.ndarray | None = None
  road_samples: np.ndarray | None = None
  path_end: float = 0.0     # calib x where the drawn path stops: its predicted end, the lead or the stop point
  leads: list[Lead] = field(default_factory=list)
  should_stop: bool = False
  stop_x: float | None = None  # calib x where the model wants to stop
  lane_change: int = 0         # LANE_CHANGE_STATES value
  lane_change_direction: int = 0
  model_frame: int = -1
  # refreshed every frame
  v_ego: float = 0.0
  a_ego: float = 0.0
  experimental: bool = False
  allow_throttle: bool = True
  signals: Signals = field(default_factory=Signals)
  darkness: float = 0.0     # 0 day .. 1 night
  stop_sign: StopSign | None = None

  @property
  def ego_ground(self) -> np.ndarray:
    """Centre of our car's footprint on the road; the camera sits about a third of the way back from the bumper."""
    return np.array([RADAR_TO_CAMERA - EGO_LENGTH / 2, 0.0, self.height])


def _xy(line, height: float) -> np.ndarray:
  """(N, 3) road points from a model XYZTData on the flat road under the camera (z = height)."""
  x, y = list(line.x), list(line.y)  # list() is the fast way out of a capnp list
  n = min(len(x), len(y))
  points = np.empty((n, 3))
  points[:, 0] = x[:n]
  points[:, 1] = y[:n]
  points[:, 2] = height
  return points


def _sample(points: np.ndarray) -> np.ndarray:
  if len(points) < 2:
    return np.zeros(SAMPLES)
  return np.interp(SAMPLE_X, points[:, 0], points[:, 1])


def _heading(points: np.ndarray, x: float) -> float:
  if len(points) < 2:
    return 0.0
  y0, y1 = np.interp([x - 2.0, x + 2.0], points[:, 0], points[:, 1])
  return math.atan2(y1 - y0, 4.0)


def _name(value) -> str:
  try:
    return str(value)
  except Exception:
    return ""


def _seen(sm, name: str) -> bool:
  try:
    return bool(sm.seen[name])
  except KeyError:
    return False


class _Latch:
  """A flag with separate on/off conditions that stays on for a minimum time."""

  def __init__(self, hold: float):
    self.hold = hold
    self.on = False
    self._since = 0.0

  def update(self, now: float, turn_on: bool, turn_off: bool) -> bool:
    if not self.on and turn_on:
      self.on, self._since = True, now
    elif self.on and turn_off and now - self._since >= self.hold:
      self.on = False
    return self.on


class SceneTracker:
  """Rebuilds the Scene geometry only when the model, radar or calibration message changes."""

  def __init__(self):
    self.scene = Scene()
    self._keys: tuple = ()
    self._revision = 0
    self._lead_brakes = {True: _Latch(LEAD_BRAKE_HOLD), False: _Latch(LEAD_BRAKE_HOLD)}
    self._ego_decel = _Latch(0.3)
    self._dark_target = 0.0
    self._darkness: float | None = None
    self._now = 0.0

  def update(self, sm, started_frame: int = 0, dt: float = 0.05) -> Scene:
    self._now += max(dt, 0.0)
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
    self._scalars(sm, self.scene, dt)
    return self.scene

  # ---- every frame

  def _scalars(self, sm, scene: Scene, dt: float) -> None:
    car = sm["carState"] if sm.seen["carState"] else None
    scene.v_ego = float(car.vEgo) if car is not None else 0.0
    scene.a_ego = float(car.aEgo) if car is not None else 0.0
    scene.experimental = bool(sm["selfdriveState"].experimentalMode) if sm.seen["selfdriveState"] else False
    scene.allow_throttle = bool(sm["longitudinalPlan"].allowThrottle) if sm.seen["longitudinalPlan"] else True

    signals = Signals()
    if car is not None:
      signals.left_blinker, signals.right_blinker = bool(car.leftBlinker), bool(car.rightBlinker)
      signals.left_blindspot, signals.right_blindspot = bool(car.leftBlindspot), bool(car.rightBlindspot)
      signals.standstill = bool(car.standstill)
      signals.reverse = _name(car.gearShifter) == "reverse"
      pedal = bool(car.brakePressed) or bool(getattr(car, "regenBraking", False))
      lights = _seen(sm, "starpilotCarState") and bool(sm["starpilotCarState"].brakeLights)
      decel = self._ego_decel.update(self._now, scene.a_ego < EGO_BRAKE_DECEL, scene.a_ego > EGO_BRAKE_DECEL * 0.6)
      signals.braking = pedal or lights or decel or (signals.standstill and scene.v_ego < 0.1)
    scene.signals = signals

    for name in ("wideRoadCameraState", "narrowRoadCameraState"):
      if _seen(sm, name):
        exposure = float(sm[name].exposureValPercent)
        if exposure >= DARK_EXPOSURE:
          self._dark_target = 1.0
        elif exposure <= LIGHT_EXPOSURE:
          self._dark_target = 0.0
        break
    if self._darkness is None:
      self._darkness = self._dark_target
    self._darkness += (self._dark_target - self._darkness) * (1.0 - math.exp(-max(dt, 0.0) / DARKNESS_TIME))
    scene.darkness = float(self._darkness)

  # ---- every model frame

  def _geometry(self, sm) -> Scene:
    calib = sm["extrinsicsCalibration"]
    height = float(calib.height[0]) if len(calib.height) else HEIGHT_DEFAULT
    model = sm["modelV2"]
    scene = Scene(valid=True, height=height)

    scene.path = _xy(model.position, height)
    scene.lane_lines = [_xy(line, height) for line in list(model.laneLines)[:4]]
    scene.road_edges = [_xy(edge, height) for edge in list(model.roadEdges)[:2]]
    scene.lane_probs = np.array(list(model.laneLineProbs), dtype=np.float64)[:4]
    scene.road_edge_stds = np.array(list(model.roadEdgeStds), dtype=np.float64)[:2]
    scene.path_accel = np.array(list(model.acceleration.x), dtype=np.float64)
    scene.model_frame = int(sm.recv_frame["modelV2"])

    lanes = np.zeros((SAMPLES, 4), dtype=np.float32)
    for i, line in enumerate(scene.lane_lines):
      lanes[:, i] = _sample(line)
    road = np.zeros((SAMPLES, 4), dtype=np.float32)
    if len(scene.road_edges) == 2:
      road[:, 0], road[:, 1] = _sample(scene.road_edges[0]), _sample(scene.road_edges[1])
    else:
      road[:, 0], road[:, 1] = -50.0, 50.0
    road[:, 2] = _sample(scene.path)
    if len(scene.path) > 1 and len(scene.path_accel) == len(scene.path):
      road[:, 3] = np.interp(SAMPLE_X, scene.path[:, 0], scene.path_accel)
    scene.lane_samples, scene.road_samples = lanes, road

    meta = model.meta
    scene.lane_change = LANE_CHANGE_STATES.get(_name(meta.laneChangeState), 0)
    scene.lane_change_direction = LANE_CHANGE_DIRECTIONS.get(_name(meta.laneChangeDirection), 0)
    scene.should_stop = bool(model.action.shouldStop)
    scene.stop_x = self._stop_x(model, scene.path) if scene.should_stop else None

    if sm.seen["radarState"] and sm.valid["radarState"]:
      scene.leads = self._leads(sm["radarState"], scene)
    end = float(scene.path[-1, 0]) if len(scene.path) else 0.0
    if scene.leads:
      end = min(end, float(scene.leads[0].rear[0]) - 0.5)
    if scene.stop_x is not None:
      end = min(end, scene.stop_x)
    scene.path_end = max(end, 0.0)
    return scene

  @staticmethod
  def _stop_x(model, path: np.ndarray) -> float | None:
    """Where the predicted speed first drops near zero; while holding at a stop, just ahead of the bumper."""
    speeds = np.array(list(model.velocity.x), dtype=np.float64)
    if len(path) < 2 or len(speeds) != len(path):
      return None
    slow = np.flatnonzero(speeds < STOP_SPEED)
    if len(slow) == 0:
      return None
    return max(float(path[slow[0], 0]), RADAR_TO_CAMERA + STOP_MIN_AHEAD)

  def _leads(self, radar_state, scene: Scene) -> list[Lead]:
    leads: list[Lead] = []
    for primary, data in ((True, radar_state.leadOne), (False, radar_state.leadTwo)):
      latch = self._lead_brakes[primary]
      d_rel, y_rel = float(data.dRel), float(data.yRel)
      v_lead, a_lead = float(data.vLead), float(data.aLeadK)
      finite = all(math.isfinite(value) for value in (d_rel, y_rel, v_lead, a_lead))  # a NaN would blank the whole road
      if not data.present or not finite or (not primary and leads and abs(d_rel - leads[0].d_rel) < SAME_LEAD_DISTANCE):
        latch.on = False
        continue
      x = RADAR_TO_CAMERA + d_rel
      braking = latch.update(self._now, a_lead < LEAD_BRAKE_ON or v_lead < LEAD_STOPPED,
                             a_lead > LEAD_BRAKE_OFF and v_lead > LEAD_STOPPED + 0.3)
      leads.append(Lead(rear=np.array([x, -y_rel, scene.height]),  # radar y is left-positive
                        heading=_heading(scene.path, x + 2.0), d_rel=d_rel, v_rel=float(data.vRel), v_lead=v_lead,
                        a_lead=a_lead, braking=braking, radar=bool(data.radar), primary=primary))
    return leads


_noted_stop_sign: tuple[float, float, float, float] | None = None


def note_stop_sign(sighting: tuple[float, float, float, float] | None) -> None:
  """runtime_snapshot hook, once per frame: StarPilot's confirmed STOP sighting (stop_sign_sighting) or None."""
  global _noted_stop_sign
  _noted_stop_sign = sighting


def noted_stop_sign() -> tuple[float, float, float, float] | None:
  return _noted_stop_sign


def stop_sign_sighting(confirmation, now_ns: int) -> tuple[float, float, float, float] | None:
  """The sign StarPilot's StopConfirmation currently confirms, as (depth m, across, up, age s) in the road camera.

  Reuses StarPilot's own result: the distance it already computed and the box it already validated and matched;
  nothing is detected or matched again. across/up are the box centre's ray (x right, y down per unit depth).
  """
  try:
    sample = getattr(confirmation, "previous", None)
    if not getattr(confirmation, "confirmed_ns", 0) or confirmation.distance_m is None or sample is None or not sample.boxes:
      return None
    from openpilot.starpilot.speed_limits.vision.stop_observation import ROAD_FOCAL_PX, STOP_RANGE_OFFSET_M
    focal = ROAD_FOCAL_PX.get((sample.width, sample.height))
    if not focal:
      return None
    x1, y1, x2, y2 = max(sample.boxes, key=lambda box: box.confidence).bbox
    age = min(max((now_ns - sample.observed_ns) / 1e9, 0.0), 1.5)
    sighting = (float(confirmation.distance_m) - STOP_RANGE_OFFSET_M, ((x1 + x2) / 2 - sample.width / 2) / focal,
                ((y1 + y2) / 2 - sample.height / 2) / focal, age)
  except (AttributeError, TypeError, ValueError, KeyError, ZeroDivisionError):
    return None  # runs inside StarPilot's snapshot builder: an unexpected shape must never break it
  return sighting if all(math.isfinite(value) for value in sighting) else None


class StopSignTracker:
  """Places the sign StarPilot confirms on the road and keeps it there while we drive up to and past it.

  Per frame this is one subtraction (our travel); a new sighting adds one 3x3 rotation into the calib frame.
  """

  def __init__(self):
    self.position: np.ndarray | None = None
    self.opacity = 0.0
    self._unseen = 0.0
    self._last: tuple | None = None

  def update(self, sighting, view_from_calib: np.ndarray, height: float, v_ego: float, dt: float) -> StopSign | None:
    travel = max(v_ego, 0.0) * max(dt, 0.0)
    if self.position is not None:
      self.position[0] -= travel  # the sign stays put; we move
    if sighting is not None and sighting != self._last:
      self._last = sighting
      depth, across, up, age = sighting
      seen = np.asarray(view_from_calib).T @ np.array([across * depth, up * depth, depth])  # view -> calib
      seen[0] -= max(v_ego, 0.0) * age  # where it is now, not when the frame was taken
      seen[2] = height
      if self.position is None or np.hypot(*(seen[:2] - self.position[:2])) > STOP_SIGN_MOVE:
        self.position = seen
      else:
        self.position += (seen - self.position) * 0.35  # boxes jitter between frames; settle on the sign
      self._unseen = 0.0
    elif sighting is None:
      self._last = None
      self._unseen += max(dt, 0.0)
    shown = self.position is not None and self._unseen < STOP_SIGN_KEEP and self.position[0] > -10.0
    self.opacity += ((1.0 if shown else 0.0) - self.opacity) * min(max(dt, 0.0) / 0.3, 1.0)
    if self.position is None or (not shown and self.opacity < 0.02):
      self.position, self.opacity = None, 0.0
      return None
    return StopSign(self.position.copy(), self.opacity)
