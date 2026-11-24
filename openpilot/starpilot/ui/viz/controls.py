"""The VIEW / MODE switcher: two stacked buttons that cycle the camera angle and the visualization mode.

Drawn by the ``scene_view_switcher`` onroad widget (widgets.py), so it can be moved, hidden and layered from Galaxy's
layout editor like any other widget. Taps reach it only when no StarPilot control claimed them (it is consulted where
a tap would otherwise toggle the sidebar), so it never takes a touch away from an existing onroad control.
"""
from __future__ import annotations

import time

from openpilot.starpilot.ui.raylib import rl
from openpilot.starpilot.ui.viz.options import ANGLE_LABELS, MODE_LABELS, VizSelection

WIDTH, HEIGHT, GAP = 250, 104, 14   # px on the 2160x1080 large UI; the widget box is WIDTH x SIZE[1]
SIZE = (WIDTH, 2 * HEIGHT + GAP)
STALE_AFTER = 0.5                   # s; buttons not drawn recently are not tappable


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
  def layout(x: float, y: float) -> dict[str, rl.Rectangle]:
    return {"angle": rl.Rectangle(x, y, WIDTH, HEIGHT), "mode": rl.Rectangle(x, y + HEIGHT + GAP, WIDTH, HEIGHT)}

  def render(self, x: float, y: float, selection: VizSelection, fill: rl.Color | None = None,
             border: rl.Color | None = None, text: rl.Color | None = None, *, fonts=None, paused: bool = False) -> None:
    """``fonts`` is the onroad view's BitmapFonts, so the labels match the rest of the HUD. ``paused``: the GPU guard
    is showing the stock view for a while (see controller.GUARD_*)."""
    self._targets = self.layout(x, y)
    self._drawn_at = time.monotonic()
    font = fonts if fonts is not None else _font()
    fill = fill or rl.Color(0, 0, 0, 150)
    border = border or rl.Color(255, 255, 255, 80)
    text = text or rl.WHITE
    caption_color = rl.Color(text.r, text.g, text.b, int(text.a * 0.62))
    for key, box in self._targets.items():
      caption, value = ("VIEW", ANGLE_LABELS[selection.angle]) if key == "angle" else ("MODE", MODE_LABELS[selection.mode])
      rl.draw_rectangle_rounded(box, 0.45, 12, fill)
      if border.a:
        rl.draw_rectangle_rounded_lines_ex(box, 0.45, 12, 3, border)
      if paused and not selection.stock:
        self._text(font, "PAUSED", 30, box.x + box.width / 2, box.y + 26, rl.Color(255, 176, 32, 255))
      else:
        self._text(font, caption, 30, box.x + box.width / 2, box.y + 26, caption_color)
      self._text(font, value, 44, box.x + box.width / 2, box.y + 68, text)

  @staticmethod
  def _text(font, text: str, size: int, cx: float, cy: float, color) -> None:
    if hasattr(font, "draw"):  # StarPilot BitmapFonts
      from openpilot.starpilot.ui.presentation import FontRole
      width = font.measure(text, FontRole.SEMI_BOLD, size).width
      top, bottom = font.vertical_ink(text, FontRole.SEMI_BOLD, size)
      font.draw(text, FontRole.SEMI_BOLD, size, cx - width / 2, cy - (top + bottom) / 2, color)
      return
    measured = _measure(font, text, size)
    rl.draw_text_ex(font, text, rl.Vector2(cx - measured.x / 2, cy - measured.y / 2), size, 0, color)

  def hide(self) -> None:
    self._targets = {}

  def hit(self, x: float, y: float) -> str | None:
    """'angle' or 'mode' for a tap at onroad coordinates (x, y), else None."""
    if time.monotonic() - self._drawn_at > STALE_AFTER:
      return None
    for key, box in self._targets.items():
      if box.x <= x <= box.x + box.width and box.y <= y <= box.y + box.height:
        return key
    return None
