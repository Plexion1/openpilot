"""The driver's choice of camera angle and visualization mode, and where it is saved.

Params keys are compiled into libparams on prebuilt branches, so the selection lives in a small JSON document in
StarPilot's storage root instead (``/data/starpilot/scene_views.json`` on the device). Anything can write the same file
(Galaxy, SSH, a future settings page); the UI re-reads it when its modification time changes. Unknown keys are kept,
so newer builds can add options without breaking older ones.
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


# Cycling order and on-screen names. Register new angles/modes here and in rigs.RIGS / renderer.MODE_STYLES.
ANGLE_LABELS = {ViewAngle.FRONT: "Front", ViewAngle.CHASE: "Chase", ViewAngle.TOP: "Top"}
MODE_LABELS = {VizMode.CAMERA: "Camera", VizMode.SYNTHETIC: "Synthetic", VizMode.HYBRID: "Hybrid"}
ANGLE_ORDER = tuple(ANGLE_LABELS)
MODE_ORDER = tuple(MODE_LABELS)


def _next(order, current):
  return order[(order.index(current) + 1) % len(order)] if current in order else order[0]


@dataclass(frozen=True)
class VizSelection:
  angle: ViewAngle = ViewAngle.FRONT
  mode: VizMode = VizMode.CAMERA
  controls: bool = True  # show the on-screen VIEW / MODE switcher

  @property
  def stock(self) -> bool:
    """Front + camera is the unmodified StarPilot onroad view."""
    return self.angle == ViewAngle.FRONT and self.mode == VizMode.CAMERA

  def next_angle(self) -> VizSelection:
    return replace(self, angle=_next(ANGLE_ORDER, self.angle))

  def next_mode(self) -> VizSelection:
    return replace(self, mode=_next(MODE_ORDER, self.mode))


def default_path() -> Path:
  from openpilot.starpilot.storage import starpilot_storage_root
  return starpilot_storage_root() / FILE_NAME


class SelectionStore:
  """Cached read of the selection file; writes are atomic and only happen on a change."""

  def __init__(self, path: Path | None = None, poll_interval: float = 2.0):
    self._path = path
    self._poll_interval = poll_interval
    self._next_poll = 0.0
    self._mtime: float | None = None
    self._document: dict = {}
    self._selection = VizSelection()

  @property
  def path(self) -> Path:
    if self._path is None:
      self._path = default_path()
    return self._path

  def get(self) -> VizSelection:
    now = time.monotonic()
    if now >= self._next_poll:
      self._next_poll = now + self._poll_interval
      self._refresh()
    return self._selection

  def set(self, selection: VizSelection) -> None:
    if selection == self._selection:
      return
    self._selection = selection
    self._document.update(angle=selection.angle.value, mode=selection.mode.value, controls=selection.controls)
    try:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      tmp = self.path.with_name(self.path.name + ".tmp")
      tmp.write_text(json.dumps(self._document, sort_keys=True) + "\n")
      os.replace(tmp, self.path)
      self._mtime = self.path.stat().st_mtime
    except OSError:
      pass  # the choice still applies for this session

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
    self._selection = VizSelection(
      angle=ViewAngle(document["angle"]) if document.get("angle") in ViewAngle._value2member_map_ else ViewAngle.FRONT,
      mode=VizMode(document["mode"]) if document.get("mode") in VizMode._value2member_map_ else VizMode.CAMERA,
      controls=document.get("controls", True) is not False,
    )
