"""Scene view widgets in StarPilot's onroad layout system.

Registered in onroad_customization.PROFILES["large"], so Galaxy's layout editor can show, hide, move, recolour and
layer them like any other widget, and drawn by OnroadView (onroad.py) in the chosen widget order:

* scene_view_switcher  the VIEW / MODE buttons, bottom right by default. Colours follow the widget's card colours.
* turn_signals         blinking green arrows either side of the current speed while a turn signal is on, using
                       StarPilot's own turn signal art; its blind spot icon instead while that side is occupied.

For now they are comma-screen widgets: StarPort (the Android Auto car screen) builds its layout without them, because
its layout module (starpilot/system/starport, outside the folders a Fast Update may replace) has to learn them first.
That integration is ready in history (adding them to projection_layout.PROJECTION_WIDGETS with car-screen defaults,
the arrows drawn on the car screen and the VIEW / MODE buttons hidden there) and ships later as one full update.

This module is imported by onroad_customization, which Galaxy loads without a display, so it must not import raylib
at module level.
"""
from __future__ import annotations

import math

SWITCHER_WIDGET = "scene_view_switcher"
TURN_SIGNAL_WIDGET = "turn_signals"
SCENE_VIEW_WIDGETS = (TURN_SIGNAL_WIDGET, SWITCHER_WIDGET)
SWITCHER_SIZE = (250, 222)       # controls.SIZE
TURN_SIGNAL_SIZE = (600, 150)
ARROW_SIZE = (104, 96)           # icons_mici/onroad/turn_signal_left.png at its native size
BLIND_SPOT_SIZE = (134, 150)     # icons_mici/onroad/blind_spot_left.png
ANCHOR_SHIFT = {TURN_SIGNAL_WIDGET: 0.5, SWITCHER_WIDGET: 1.0}  # follow the speed / the right edge when the road widens
COLORS = {SWITCHER_WIDGET: {"cardFill": "#00000096", "cardBorder": "#FFFFFF50", "text": "#FFFFFFFF"}}
BLINK_MS, BLIND_SPOT_BLINK_MS = 500, 250  # StarPilot's turn signal border timing


def widget_specs() -> dict[str, dict]:
  """Large-profile registry entries (onroad_customization.Widget). Both start enabled; Galaxy can turn them off."""
  return {
    TURN_SIGNAL_WIDGET: {
      "label": "Turn signals", "kind": TURN_SIGNAL_WIDGET, "width": TURN_SIGNAL_SIZE[0], "height": TURN_SIGNAL_SIZE[1],
      "default": {"x": 630, "y": 65, "enabled": True},
      "note": ("Blinking green arrows at each end while that turn signal is on, or the blind spot icon while that side is " +
               "occupied. Centre it on the speed to frame the number.")},
    SWITCHER_WIDGET: {
      "label": "Scene view buttons", "kind": SWITCHER_WIDGET, "width": SWITCHER_SIZE[0], "height": SWITCHER_SIZE[1],
      "default": {"x": 1556, "y": 804, "enabled": True},
      "note": ("VIEW and MODE buttons: tap to cycle the camera angle (Front, Chase, Top) and the visualization " +
               "(Camera, Synthetic, Hybrid). Front + Camera is the normal StarPilot view.")},
  }


class _SafeWidget:
  """An error while drawing is logged once and hides the widget for the rest of the session (never the whole HUD)."""
  failed = False

  def _draw(self, draw, *args) -> None:
    if self.failed:
      return
    try:
      draw(*args)
    except Exception:
      from openpilot.common.swaglog import cloudlog
      cloudlog.exception(f"scene views: {type(self).__name__} failed, hiding it")
      self.failed = True


class TurnSignalArrows(_SafeWidget):
  """The turn_signals widget, owned by OnroadView like StarPilot's other widgets."""

  def render(self, state, extra_width: float) -> None:
    self._draw(render_turn_signals, state, extra_width)


class SceneViewButtons(_SafeWidget):
  """The scene_view_switcher widget (VIEW / MODE buttons), owned by OnroadView like StarPilot's other widgets."""

  def __init__(self, fonts=None):
    self.fonts = fonts

  def render(self, state, extra_width: float) -> None:
    self._draw(render_switcher, state, extra_width, self.fonts)


def _position(state, key: str, extra_width: float):
  layout = state.customization["layouts"].get("large", {}).get(key)
  if layout is None or not layout.get("enabled"):
    return None
  return layout["x"] + ANCHOR_SHIFT[key] * extra_width, layout["y"]


def _hidden(state) -> bool:
  from openpilot.starpilot.ui.appearance_preferences import CameraViewChoice
  from openpilot.starpilot.ui.onroad_state import AlertSize
  return (state.alert.size == AlertSize.FULL or state.appearance.camera_view == CameraViewChoice.DRIVER or
          state.reverse_driver_camera)


def render_turn_signals(state, extra_width: float) -> None:
  position = _position(state, TURN_SIGNAL_WIDGET, extra_width)
  signals = state.border_signals
  if position is None or signals is None or not (signals.left_blinker or signals.right_blinker) or _hidden(state):
    return
  from openpilot.starpilot.ui.raylib import rl
  if not rl.is_window_ready():
    return  # never draw without a GL context (offline layout checks)
  from openpilot.starpilot.ui.onroad_border import draw_turn_signal
  time_s = state.observed_ns / 1e9 if state.observed_ns else rl.get_time()
  if not math.isfinite(time_s):
    return
  x, y = position
  width, height = TURN_SIGNAL_SIZE
  for active, blocked, direction in ((signals.left_blinker, signals.left_blindspot, -1),
                                     (signals.right_blinker, signals.right_blindspot, 1)):
    period = BLIND_SPOT_BLINK_MS if blocked else BLINK_MS
    if not active or int(time_s * 1000) % (2 * period) >= period:
      continue
    icon_width, icon_height = BLIND_SPOT_SIZE if blocked else ARROW_SIZE
    icon_x = x if direction < 0 else x + width - icon_width
    draw_turn_signal(icon_x, y + (height - icon_height) / 2, direction, blocked=blocked)


def render_switcher(state, extra_width: float, fonts=None) -> None:
  from openpilot.starpilot.ui.viz import controller
  position = _position(state, SWITCHER_WIDGET, extra_width)
  if position is None or _hidden(state) or controller.projecting():
    controller.shared().controls.hide()
    return
  from openpilot.starpilot.ui.raylib import rl
  if not rl.is_window_ready():
    return  # never draw without a GL context (offline layout checks)
  from openpilot.starpilot.ui.onroad_customization import rgba
  colors = [rl.Color(*rgba(state.customization, key, "large", SWITCHER_WIDGET)) for key in ("cardFill", "cardBorder", "text")]
  controller.render_switcher(*position, *colors, fonts=fonts)
