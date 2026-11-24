"""Scene view preferences as StarPilot settings rows.

The rows appear on the device's Onroad HUD settings page (StarShellSession.appearance_snapshot adds them on the Big
layout), on Galaxy's Visual settings page and, grouped under "Scene views", in Galaxy's onroad layout editor (both via
AppearanceOwner's include_large rows); AppearanceOwner.apply routes their changes here. They are saved in the scene
view document (options.py), not in Params, and follow the same rules as the other appearance rows: editable while
parked, and a save is refused when the document changed since the row was read.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from openpilot.starpilot.ui.feature_settings_state import FeatureRow, FeatureSettingsRequest
from openpilot.starpilot.ui.viz.options import (ANGLE_LABELS, COLORS, MODE_LABELS, VEHICLE_LABELS, SceneStyle, default_path,
                                                parse_selection, parse_style)

PREFIX = "SceneViews:"
MAX_BYTES = 4096

# key -> (label, document field, {label: stored value}, description)
ROWS = {
  PREFIX + "angle": ("Scene view angle", "angle", {label: value.value for value, label in ANGLE_LABELS.items()},
                     ("Camera angle for the driving screen: Front is the road camera, Chase follows from above and behind, " +
                      "Top looks straight down. Front with the Camera mode is the normal StarPilot view.")),
  PREFIX + "mode": ("Scene view mode", "mode", {label: value.value for value, label in MODE_LABELS.items()},
                    ("Camera shows the road camera; Synthetic draws the road, lanes, path and vehicles from the driving " +
                     "model only; Hybrid adds lane paint found in the camera image to the synthetic road.")),
  PREFIX + "egoModel": ("Your vehicle", "egoModel", {label: value.value for value, label in VEHICLE_LABELS.items()},
                        "How your car is drawn in the Chase and Top scene views."),
  PREFIX + "egoColor": ("Your vehicle color", "egoColor", {name: name for name in COLORS},
                        "Paint colour of your car in the scene views."),
  PREFIX + "leadModel": ("Lead vehicles", "leadModel", {label: value.value for value, label in VEHICLE_LABELS.items()},
                         "How the vehicles ahead of you are drawn in the scene views."),
  PREFIX + "leadColor": ("Lead vehicle color", "leadColor", {name: name for name in COLORS},
                         "Paint colour of the vehicles ahead of you in the scene views."),
}
LABELS = {key: spec[0] for key, spec in ROWS.items()}


def _read(path: Path) -> tuple[bytes | None, dict]:
  import json
  try:
    raw = path.read_bytes()
  except FileNotFoundError:
    return None, {}
  except OSError:
    raise
  if len(raw) > MAX_BYTES:
    return raw, {}
  try:
    document = json.loads(raw)
  except ValueError:
    return raw, {}
  return raw, document if isinstance(document, dict) else {}


def _current(document: dict) -> dict[str, str]:
  """Displayed label of every row (defaults where the document has nothing valid)."""
  selection, style = parse_selection(document), parse_style(document)
  default = SceneStyle()
  stored = {"angle": selection.angle.value, "mode": selection.mode.value,
            "egoModel": style.ego.model.value, "egoColor": style.ego.color,
            "leadModel": style.lead.model.value, "leadColor": style.lead.color}
  defaults = {"angle": "front", "mode": "camera", "egoModel": default.ego.model.value, "egoColor": default.ego.color,
              "leadModel": default.lead.model.value, "leadColor": default.lead.color}
  values, default_labels = {}, {}
  for key, (_label, field, choices, _description) in ROWS.items():
    by_value = {value: label for label, value in choices.items()}
    values[key] = by_value[stored[field]]
    default_labels[key] = by_value[defaults[field]]
  return values | {"__defaults__": default_labels}


def rows(parked: bool, path: Path | None = None) -> tuple[FeatureRow, ...]:
  path = path or default_path()
  try:
    raw, document = _read(path)
    readable = True
  except OSError:
    raw, document, readable = None, {}, False
  values = _current(document)
  defaults = values.pop("__defaults__")
  return tuple(FeatureRow(key, label, values[key], source=raw, choices=tuple(choices), available=parked and readable,
                          reason=description if readable else "The scene view settings file cannot be read",
                          default_value=defaults[key])
               for key, (label, _field, choices, description) in ROWS.items())


def apply(request: FeatureSettingsRequest, parked: Callable[[], bool], path: Path | None = None) -> bool:
  spec = ROWS.get(request.key)
  if spec is None or not parked():
    return False
  _label, field, choices, _description = spec
  if request.value not in choices:
    return False
  path = path or default_path()
  try:
    raw, document = _read(path)
  except OSError:
    return False
  if raw != request.expected:
    return False  # changed since the row was shown
  from openpilot.starpilot.ui.viz.options import SelectionStore
  store = SelectionStore(path, poll_interval=0.0)
  store.get()  # load the document so unknown fields are kept
  return store.update(**{field: choices[request.value]})
