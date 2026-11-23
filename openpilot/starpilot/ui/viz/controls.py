"""On-screen VIEW / MODE switcher drawn in the bottom-right corner of the onroad camera layer.

Taps reach it only when no StarPilot control claimed them (it is consulted where a tap would otherwise toggle the
sidebar), so it can never take a touch away from an existing onroad control.
"""
from __future__ import annotations

import time

from openpilot.starpilot.ui.raylib import rl
from openpilot.starpilot.ui.viz.options import ANGLE_LABELS, MODE_LABELS, VizSelection

WIDTH, HEIGHT = 250, 104   # px on the 2160x1080 large UI (about 16 x 6.5 mm)
GAP, MARGIN = 14, 24
STALE_AFTER = 0.5          # s; buttons not drawn recently are not tappable


def _font():
  try:
    from openpilot.system.ui.lib.application import FontWeight, gui_app
    return gui_app.font(FontWeight.SEMI_BOLD)
  except Exception:
    return rl.get_font_default()


def _measure(font, text: str, size: int):
  try:
    from openpilot.system.ui.lib.text_measure import measure_text_cached
    return measure_text_cached(font, text, size)
  except Exception:
    return rl.Vector2(len(text) * size * 0.55, size)


class VizControls:
  def __init__(self):
    self._targets: dict[str, rl.Rectangle] = {}
    self._drawn_at = 0.0

  @staticmethod
  def layout(rect: rl.Rectangle) -> dict[str, rl.Rectangle]:
    x = rect.x + rect.width - MARGIN - WIDTH
    mode_y = rect.y + rect.height - MARGIN - HEIGHT
    return {"angle": rl.Rectangle(x, mode_y - GAP - HEIGHT, WIDTH, HEIGHT), "mode": rl.Rectangle(x, mode_y, WIDTH, HEIGHT)}

  def render(self, rect: rl.Rectangle, selection: VizSelection) -> None:
    self._targets = self.layout(rect)
    self._drawn_at = time.monotonic()
    font = _font()
    for key, box in self._targets.items():
      caption, value = ("VIEW", ANGLE_LABELS[selection.angle]) if key == "angle" else ("MODE", MODE_LABELS[selection.mode])
      rl.draw_rectangle_rounded(box, 0.45, 12, rl.Color(0, 0, 0, 150))
      rl.draw_rectangle_rounded_lines_ex(box, 0.45, 12, 3, rl.Color(255, 255, 255, 80))
      self._text(font, caption, 30, box.x + box.width / 2, box.y + 26, rl.Color(255, 255, 255, 160))
      self._text(font, value, 44, box.x + box.width / 2, box.y + 68, rl.WHITE)

  @staticmethod
  def _text(font, text: str, size: int, cx: float, cy: float, color) -> None:
    measured = _measure(font, text, size)
    rl.draw_text_ex(font, text, rl.Vector2(cx - measured.x / 2, cy - measured.y / 2), size, 0, color)

  def hit(self, x: float, y: float) -> str | None:
    """'angle' or 'mode' for a tap at layer coordinates (x, y), else None."""
    if time.monotonic() - self._drawn_at > STALE_AFTER:
      return None
    for key, box in self._targets.items():
      if box.x <= x <= box.x + box.width and box.y <= y <= box.y + box.height:
        return key
    return None
