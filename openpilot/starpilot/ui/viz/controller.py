"""Entry point used by the onroad camera layer: decides whether a scene view replaces the stock layer and draws it.

Hooks (each a couple of lines in existing files):
  * AugmentedRoadView.render_camera_model_layer -> SceneViews.render() first, SceneViews.render_controls() last
  * StarShellPage._handle_mouse_release -> SceneViews.handle_tap() before a tap toggles the sidebar
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
             started_frame: int, status: str, color_strength, projection: bool) -> bool:
    """Draw the selected scene view. Returns False when the stock camera/model layer should draw instead."""
    if self.failed or projection:
      return False
    selection = self.store.get()
    if selection.stock or camera_view in (CAMERA_VIEW_DRIVER, CAMERA_VIEW_NONE) or reverse_driver:
      self._last_time = None
      return False
    try:
      self._render(view, rect, sm, selection, camera_view, reverse_driver, poll_camera, started_frame, status, color_strength)
    except Exception:
      # A prototype must never take the driving UI down: log once and fall back to the stock layer for this session.
      rl.end_shader_mode()  # in case the failure happened between begin/end
      cloudlog.exception("scene views: render failed, falling back to the stock camera view")
      self.failed = True
      return False
    return True

  def _render(self, view, rect, sm, selection: VizSelection, camera_view: int, reverse_driver: bool, poll_camera: bool,
              started_frame: int, status: str, color_strength) -> None:
    now = time.monotonic()
    dt = 0.05 if self._last_time is None else min(max(now - self._last_time, 0.0), 0.5)
    self._last_time = now

    rig = RIGS[selection.angle]
    if rig.angle != self._angle:
      rig.reset()
      self._angle = rig.angle

    feed = CameraFeed(view)
    feed.update_calibration()
    wants_camera = selection.mode in (VizMode.CAMERA, VizMode.HYBRID)
    if poll_camera:
      feed.select_stream(sm, rig.reprojection_stream if wants_camera else None, camera_view, reverse_driver)
      feed.poll()

    scene = self.tracker.update(sm, started_frame)
    source = feed.geometry()
    r = (rect.x, rect.y, rect.width, rect.height)
    front = stock_front_camera(source.intrinsics, source.view_from_calib, r, source.wide)
    camera = rig.camera(RigContext(r, scene, front, dt))

    frame = feed.textures() if wants_camera else None
    strength = 1.0
    if frame is not None:
      strength = color_strength() if callable(color_strength) else float(color_strength)
    self.renderer.render(rect, rig, camera, scene, selection.mode, dt=dt, frame=frame, source=source,
                         color_strength=strength, status=status)

  def render_controls(self, rect: rl.Rectangle, *, interactive: bool) -> None:
    if self.failed or not interactive:
      return
    selection = self.store.get()
    if not selection.controls:
      return
    try:
      self.controls.render(rect, selection)
    except Exception:
      cloudlog.exception("scene views: controls failed")
      self.failed = True

  def handle_tap(self, x: float, y: float) -> bool:
    """Tap at onroad layer coordinates; True when it changed the view (the caller then skips its own tap action)."""
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
  """The process-wide instance, so the camera layer and the touch handler see the same buttons and selection."""
  global _shared
  if _shared is None:
    _shared = SceneViews()
  return _shared


# ---- glue for the existing UI; ui_state is imported lazily so the classes above stay testable off-device


def render_layer(view, rect: rl.Rectangle, *, camera_view: int, reverse_driver: bool, poll_camera: bool) -> bool:
  """AugmentedRoadView hook. True when a scene view (and its switcher) was drawn instead of the stock layer."""
  from openpilot.selfdrive.ui.ui_state import ui_state
  from openpilot.starpilot.ui.camera_color import road_color_strength
  views = shared()
  # StarPort's car display renders this layer in its own read-only process; it keeps the stock view for now.
  projection = ui_state.projection_read_only and ui_state.replay_clock is None
  drawn = views.render(view, rect, ui_state.sm, camera_view=camera_view, reverse_driver=reverse_driver, poll_camera=poll_camera,
                       started_frame=ui_state.started_frame, status=ui_state.status.value,
                       color_strength=lambda: road_color_strength(ui_state), projection=projection)
  if drawn:
    render_controls(rect, poll_camera=poll_camera)
  return drawn


def render_controls(rect: rl.Rectangle, *, poll_camera: bool) -> None:
  """Draw the VIEW / MODE switcher on the live device UI (not in replays or projections)."""
  from openpilot.selfdrive.ui.ui_state import ui_state
  shared().render_controls(rect, interactive=poll_camera and not ui_state.projection_read_only)


def handle_tap(x: float, y: float) -> bool:
  """StarShellPage hook for an onroad tap nothing else claimed, in onroad layer coordinates."""
  try:
    return shared().handle_tap(x, y)
  except Exception:
    cloudlog.exception("scene views: tap failed")
    return False
