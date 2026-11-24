"""Entry points used by StarPilot: decides whether a scene view replaces the stock camera layer and draws it.

Hooks (a few lines in existing files):
  * AugmentedRoadView.render_camera_model_layer -> render_layer() first; when it returns True the stock camera and
    model renderer are skipped and every StarPilot widget, alert and border still draws on top
  * OnroadView draws the scene view widgets (widgets.py) in its widget order: the VIEW / MODE switcher and the turn
    signal arrows
  * StarShellPage._handle_mouse_release -> handle_tap() before a tap toggles the sidebar
  * AppearanceOwner lists the settings rows (settings.py) and applies their changes

The time spent is reported through the UI's frame phases like the stock layer: camera polling as "camera", the
scene and its drawing as "model", so slow frame diagnostics compare directly.
"""
from __future__ import annotations

import time

from openpilot.common.swaglog import cloudlog
from openpilot.starpilot.ui.raylib import rl
from openpilot.starpilot.ui.viz.camera_feed import CameraFeed
from openpilot.starpilot.ui.viz.controls import VizControls
from openpilot.starpilot.ui.viz.options import SelectionStore, VizMode, VizSelection
from openpilot.starpilot.ui.viz.renderer import VizRenderer
from openpilot.starpilot.ui.viz.rigs import RIGS, RigContext, stock_front_camera
from openpilot.starpilot.ui.viz.scene import SceneTracker

# AugmentedRoadView camera_view values (appearance CameraView param) that keep the stock layer.
CAMERA_VIEW_DRIVER, CAMERA_VIEW_NONE = 1, 4


def _phase(name: str, callback, *args):
  try:
    from openpilot.system.ui.lib.application import gui_app
  except Exception:
    return callback(*args)
  return gui_app.measure_frame_phase(name, callback, *args)


class SceneViews:
  def __init__(self, store: SelectionStore | None = None):
    self.store = store or SelectionStore()
    self.tracker = SceneTracker()
    self.renderer = VizRenderer()
    self.controls = VizControls()
    self.failed = False
    self._last_time: float | None = None
    self._angle = None

  def selection(self) -> VizSelection:
    return self.store.get()

  def render(self, view, rect: rl.Rectangle, sm, *, camera_view: int, reverse_driver: bool, poll_camera: bool,
             started_frame: int, status: str, color_strength, projection: bool, road_style: dict | None = None) -> bool:
    """Draw the selected scene view. Returns False when the stock camera/model layer should draw instead."""
    if self.failed or projection:
      return False
    selection = self.store.get()
    if selection.stock or camera_view in (CAMERA_VIEW_DRIVER, CAMERA_VIEW_NONE) or reverse_driver:
      self._last_time = None
      return False
    try:
      self._render(view, rect, sm, selection, camera_view, reverse_driver, poll_camera, started_frame, status, color_strength,
                   road_style)
    except Exception:
      # Never take the driving UI down: log once and fall back to the stock layer for this session.
      rl.end_shader_mode()  # in case the failure happened between begin/end
      cloudlog.exception("scene views: render failed, falling back to the stock camera view")
      self.failed = True
      return False
    return True

  def _render(self, view, rect, sm, selection: VizSelection, camera_view: int, reverse_driver: bool, poll_camera: bool,
              started_frame: int, status: str, color_strength, road_style: dict | None) -> None:
    now = time.monotonic()
    dt = 0.05 if self._last_time is None else min(max(now - self._last_time, 0.0), 0.5)
    self._last_time = now

    rig = RIGS[selection.angle]
    if rig.angle != self._angle:
      rig.reset()
      self._angle = rig.angle

    feed = CameraFeed(view)
    wants_camera = selection.mode in (VizMode.CAMERA, VizMode.HYBRID)

    def camera_work():
      feed.update_calibration()
      if poll_camera:
        feed.select_stream(sm, rig.reprojection_stream if wants_camera else None, camera_view, reverse_driver)
        feed.poll()
      return feed.textures() if wants_camera else None
    frame = _phase("camera", camera_work)

    def scene_work():
      scene = self.tracker.update(sm, started_frame, dt)
      source = feed.geometry()
      r = (rect.x, rect.y, rect.width, rect.height)
      front = stock_front_camera(source.intrinsics, source.view_from_calib, r, source.wide)
      camera = rig.camera(RigContext(r, scene, front, dt))
      strength = 1.0
      if frame is not None:
        strength = color_strength() if callable(color_strength) else float(color_strength)
      self.renderer.render(rect, rig, camera, scene, selection.mode, dt=dt, style=self.store.style(), road_style=road_style,
                           frame=frame, source=source, color_strength=strength, status=status, blink_time=rl.get_time())
    _phase("model", scene_work)

  def render_switcher(self, x: float, y: float, fill=None, border=None, text=None, fonts=None) -> None:
    if self.failed:
      return
    try:
      self.controls.render(x, y, self.store.get(), fill, border, text, fonts=fonts)
    except Exception:
      cloudlog.exception("scene views: switcher failed")
      self.failed = True

  def handle_tap(self, x: float, y: float) -> bool:
    """Tap at onroad coordinates; True when it changed the view (the caller then skips its own tap action)."""
    if self.failed:
      return False
    target = self.controls.hit(x, y)
    if target is None:
      return False
    selection = self.store.get()
    self.store.set(selection.next_angle() if target == "angle" else selection.next_mode())
    return True

  def close(self) -> None:
    self.renderer.close()


_shared: SceneViews | None = None


def shared() -> SceneViews:
  """The process-wide instance, so the camera layer, the widgets and the touch handler share one selection."""
  global _shared
  if _shared is None:
    _shared = SceneViews()
  return _shared


# ---- glue for the existing UI; ui_state is imported lazily so the classes above stay testable off-device


def render_layer(view, rect: rl.Rectangle, *, camera_view: int, reverse_driver: bool, poll_camera: bool,
                 road_style: dict | None = None) -> bool:
  """AugmentedRoadView hook. True when a scene view was drawn instead of the stock layer."""
  from openpilot.selfdrive.ui.ui_state import ui_state
  from openpilot.starpilot.ui.camera_color import road_color_strength
  # StarPort's car display renders this layer in its own read-only process; it keeps the stock view for now.
  projection = ui_state.projection_read_only and ui_state.replay_clock is None
  return shared().render(view, rect, ui_state.sm, camera_view=camera_view, reverse_driver=reverse_driver, poll_camera=poll_camera,
                         started_frame=ui_state.started_frame, status=ui_state.status.value,
                         color_strength=lambda: road_color_strength(ui_state), projection=projection, road_style=road_style)


def render_switcher(x: float, y: float, fill=None, border=None, text=None, fonts=None) -> None:
  """Draw the VIEW / MODE buttons at onroad coordinates (x, y) (the scene_view_switcher widget)."""
  shared().render_switcher(x, y, fill, border, text, fonts=fonts)


def handle_tap(x: float, y: float) -> bool:
  """StarShellPage hook for an onroad tap nothing else claimed, in onroad coordinates."""
  try:
    return shared().handle_tap(x, y)
  except Exception:
    cloudlog.exception("scene views: tap failed")
    return False
