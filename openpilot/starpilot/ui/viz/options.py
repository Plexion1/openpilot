"""The driver's scene view choices and where they are saved.

Params keys are compiled into libparams on prebuilt branches, so the choices live in a small JSON document in
StarPilot's storage root instead (``/data/starpilot/scene_views.json`` on the device), e.g.

  {"angle": "chase", "mode": "synthetic", "egoModel": "car", "egoColor": "Ocean Blue", "leadModel": "truck",
   "leadColor": "Silver"}

The on-screen switcher, the settings rows (settings.py, shown on the device and in Galaxy) and anything else may
write it; the UI re-reads it when its modification time changes. Unknown keys are kept, so newer builds can add
options without breaking older ones, and unknown values fall back to the defaults.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

FILE_NAME = "scene_views.json"


class ViewAngle(StrEnum):
  FRONT = "front"
  CHASE = "chase"
  TOP = "top"


class VizMode(StrEnum):
  CAMERA = "camera"
  SYNTHETIC = "synthetic"
  HYBRID = "hybrid"


class VehicleModel(StrEnum):
  CAR = "car"
  TRUCK = "truck"
  SPHERE = "sphere"


# Cycling order and on-screen names. Register new angles/modes here and in rigs.RIGS / renderer.MODE_STYLES.
ANGLE_LABELS = {ViewAngle.FRONT: "Front", ViewAngle.CHASE: "Chase", ViewAngle.TOP: "Top"}
MODE_LABELS = {VizMode.CAMERA: "Camera", VizMode.SYNTHETIC: "Synthetic", VizMode.HYBRID: "Hybrid"}
VEHICLE_LABELS = {VehicleModel.CAR: "Car", VehicleModel.TRUCK: "Truck", VehicleModel.SPHERE: "Sphere"}
ANGLE_ORDER = tuple(ANGLE_LABELS)
MODE_ORDER = tuple(MODE_LABELS)

# Paint colours offered for the vehicles (sRGB; the vehicle shader lights them).
COLORS = {
  "Pearl White": (226, 229, 232),
  "Silver": (160, 166, 174),
  "Graphite": (74, 78, 86),
  "Midnight": (28, 30, 36),
  "Racing Red": (190, 30, 36),
  "Sunset Orange": (228, 108, 30),
  "Solar Yellow": (232, 190, 40),
  "Forest Green": (32, 120, 70),
  "Ocean Blue": (34, 96, 204),
  "Galaxy Purple": (124, 80, 212),
}


def _next(order, current):
  return order[(order.index(current) + 1) % len(order)] if current in order else order[0]


@dataclass(frozen=True)
class VizSelection:
  angle: ViewAngle = ViewAngle.FRONT
  mode: VizMode = VizMode.CAMERA

  @property
  def stock(self) -> bool:
    """Front + camera is the unmodified StarPilot onroad view."""
    return self.angle == ViewAngle.FRONT and self.mode == VizMode.CAMERA

  def next_angle(self) -> VizSelection:
    return replace(self, angle=_next(ANGLE_ORDER, self.angle))

  def next_mode(self) -> VizSelection:
    return replace(self, mode=_next(MODE_ORDER, self.mode))


@dataclass(frozen=True)
class VehicleLook:
  model: VehicleModel
  color: str

  @property
  def rgb(self) -> tuple[int, int, int]:
    return COLORS.get(self.color, COLORS["Silver"])


@dataclass(frozen=True)
class SceneStyle:
  ego: VehicleLook = VehicleLook(VehicleModel.CAR, "Ocean Blue")
  lead: VehicleLook = VehicleLook(VehicleModel.CAR, "Silver")


def default_path() -> Path:
  from openpilot.starpilot.storage import starpilot_storage_root
  return starpilot_storage_root() / FILE_NAME


def _choice(value, choices, fallback):
  """``value`` when it is one of ``choices`` (a hand-edited or corrupted document may hold anything), else fallback."""
  return value if isinstance(value, str) and value in choices else fallback


def parse_selection(document: dict) -> VizSelection:
  return VizSelection(angle=ViewAngle(_choice(document.get("angle"), ViewAngle._value2member_map_, ViewAngle.FRONT)),
                      mode=VizMode(_choice(document.get("mode"), VizMode._value2member_map_, VizMode.CAMERA)))


def parse_style(document: dict) -> SceneStyle:
  default = SceneStyle()

  def look(prefix: str, fallback: VehicleLook) -> VehicleLook:
    return VehicleLook(VehicleModel(_choice(document.get(prefix + "Model"), VehicleModel._value2member_map_, fallback.model)),
                       _choice(document.get(prefix + "Color"), COLORS, fallback.color))
  return SceneStyle(ego=look("ego", default.ego), lead=look("lead", default.lead))


class SelectionStore:
  """Cached read of the scene view document; writes are atomic and only happen on a change."""

  def __init__(self, path: Path | None = None, poll_interval: float = 2.0):
    self._path = path
    self._poll_interval = poll_interval
    self._next_poll = 0.0
    self._mtime: float | None = None
    self._document: dict = {}
    self._selection = VizSelection()
    self._style = SceneStyle()

  @property
  def path(self) -> Path:
    if self._path is None:
      self._path = default_path()
    return self._path

  def get(self) -> VizSelection:
    self._poll()
    return self._selection

  def style(self) -> SceneStyle:
    self._poll()
    return self._style

  def set(self, selection: VizSelection) -> None:
    if selection != self._selection:
      self.update(angle=selection.angle.value, mode=selection.mode.value)

  def update(self, **fields) -> bool:
    """Merge fields into the document and save it. False when it could not be written (it still applies until restart).
    The file is re-read first, so a change another writer made since the last poll is kept."""
    self._refresh()
    self._document.update(fields)
    self._selection, self._style = parse_selection(self._document), parse_style(self._document)
    try:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      tmp = self.path.with_name(self.path.name + ".tmp")
      tmp.write_text(json.dumps(self._document, sort_keys=True) + "\n")
      os.replace(tmp, self.path)
      self._mtime = self.path.stat().st_mtime
    except OSError:
      return False
    return True

  def _poll(self) -> None:
    now = time.monotonic()
    if now >= self._next_poll:
      self._next_poll = now + self._poll_interval
      self._refresh()

  def _refresh(self) -> None:
    try:
      mtime = self.path.stat().st_mtime
    except (OSError, ValueError):
      return
    if mtime == self._mtime:
      return
    self._mtime = mtime
    try:
      document = json.loads(self.path.read_text())
    except (OSError, ValueError):
      return
    if not isinstance(document, dict):
      return
    self._document = document
    self._selection, self._style = parse_selection(document), parse_style(document)
