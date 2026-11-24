"""Entry points used by StarPilot: decides whether a scene view replaces the stock camera layer and draws it.

Hooks (a few lines in existing files):
  * AugmentedRoadView.render_camera_model_layer -> render_layer() first; when it returns True the stock camera and
    model renderer are skipped and every StarPilot widget, alert and border still draws on top
  * OnroadView draws the scene view widgets (widgets.py) in its widget order: the VIEW / MODE switcher and the turn
    signal arrows
  * StarShellPage._handle_mouse_release -> handle_tap() before a tap toggles the sidebar
  * AppearanceOwner lists the settings rows (settings.py) and applies their changes

GPU budget: the driving model shares the GPU with the UI, so the 3D scene is drawn at RENDER_SCALE resolution into
an offscreen image during StarPilot's render-prepare step (gui_app.add_render_prepare, outside the frame, like the
torque bar and the map overlay) and the camera layer only stretches that image over its rectangle. StarPilot's
widgets, text and borders are still drawn at full resolution on top. The first frame after switching views is drawn
directly while the offscreen path warms up.

The time spent is reported through the UI's frame phases like the stock layer: camera polling as "camera", the
scene and its drawing as "model", so slow frame diagnostics compare directly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openpilot.common.swaglog import cloudlog
from openpilot.starpilot.ui.raylib import rl
from openpilot.starpilot.ui.viz.camera_feed import CameraFeed
from openpilot.starpilot.ui.viz.controls import VizControls
from openpilot.starpilot.ui.viz.options import SelectionStore, VizMode, VizSelection
from openpilot.starpilot.ui.viz.renderer import VizRenderer
from openpilot.starpilot.ui.viz.rigs import RIGS, RigContext, stock_front_camera
from openpilot.starpilot.ui.viz.scene import SceneTracker, StopSignTracker

# AugmentedRoadView camera_view values (appearance CameraView param) that keep the stock layer.
CAMERA_VIEW_DRIVER, CAMERA_VIEW_NONE = 1, 4
RENDER_SCALE = 0.5   # offscreen scene resolution relative to the camera layer (GPU cost scales with its square)

# GPU guard: the driving model shares the GPU. If it starts dropping frames while a scene view is on, draw the scene at
# a lower resolution; if it still drops at the lowest one, show the stock camera view for a while.
GUARD_SCALES = (RENDER_SCALE, 0.35, 0.25)
GUARD_DROP_PERCENT = 2.0  # modelV2.frameDropPerc (a filtered percentage; 0 when the model keeps up)
GUARD_SETTLE = 6.0        # s between steps, so the filtered drop rate can respond to the previous one
GUARD_PAUSE = 60.0        # s of stock view once the lowest resolution was not enough


def _phase(name: str, callback, *args):
  try:
    from openpilot.system.ui.lib.application import gui_app
  except Exception:
    return callback(*args)
  return gui_app.measure_frame_phase(name, callback, *args)


@dataclass
class _Job:
  """What the camera layer asked for, replayed by the next prepare step."""
  view: Any
  rect: tuple[float, float, float, float]
  sm: Any
  selection: VizSelection
  camera_view: int
  reverse_driver: bool
  poll_camera: bool
  started_frame: int
  status: str
  color_strength: Any
  road_style: dict | None
  stop_sign: tuple | None
  clock: Any


class SceneViews:
  def __init__(self, store: SelectionStore | None = None, render_scale: float = RENDER_SCALE):
    self.store = store or SelectionStore()
    self.tracker = SceneTracker()
    self.stop_signs = StopSignTracker()
    self.renderer = VizRenderer()
    self.controls = VizControls()
    self.failed = False
    self.render_scale = render_scale
    self._last_time: float | None = None
    self._angle = None
    self._job: _Job | None = None
    self._job_serial = 0       # bumps on every camera layer call
    self._prepared_serial = -1  # job serial the last prepare consumed
    self._ready: tuple | None = None  # (selection, rect) of the offscreen image drawn by this frame's prepare
    self._target = None
    self._guard_step = 0
    self._guard_next = 0.0     # monotonic time of the next allowed step
    self._paused_until = 0.0

  def selection(self) -> VizSelection:
    return self.store.get()

  @property
  def paused(self) -> bool:
    """The GPU guard is showing the stock view for a while."""
    return time.monotonic() < self._paused_until

  def render(self, view, rect: rl.Rectangle, sm, *, camera_view: int, reverse_driver: bool, poll_camera: bool,
             started_frame: int, status: str, color_strength, projection: bool, road_style: dict | None = None,
             stop_sign: tuple | None = None, now: float | None = None, clock=None) -> bool:
    """Draw the selected scene view. Returns False when the stock camera/model layer should draw instead.

    ``stop_sign`` is StarPilot's own confirmed STOP sighting (scene.stop_sign_sighting), passed through unchanged.
    ``now`` / ``clock`` give StarPilot's display clock in seconds (live: monotonic; replay: the recording's time), so
    motion, fades and blinking follow a replay's pace and pauses; without them the monotonic clock is used.
    """
    self._job = None
    if self.failed or projection or self.paused:
      return False
    selection = self.store.get()
    if selection.stock or camera_view in (CAMERA_VIEW_DRIVER, CAMERA_VIEW_NONE) or reverse_driver:
      self._last_time = None
      self._ready = None
      return False
    key = (selection, (rect.x, rect.y, rect.width, rect.height))
    self._job_serial += 1
    self._job = _Job(view, key[1], sm, selection, camera_view, reverse_driver, poll_camera, started_frame, status,
                     color_strength, road_style, stop_sign, clock or ((lambda: now) if now else None))
    try:
      if self._ready == key and self._target is not None:
        self._ready = None
        texture = self._target.texture
        rl.draw_texture_pro(texture, rl.Rectangle(0, 0, float(texture.width), -float(texture.height)), rect, rl.Vector2(0, 0), 0.0,
                            rl.WHITE)
      else:
        # Not prepared offscreen this frame (just switched, or no prepare step): draw it directly once.
        self._render(self._job, rect, None)
    except Exception:
      self._fail("render failed")
      return False
    return True

  def prepare(self) -> None:
    """gui_app render-prepare step: draw the scene the camera layer will show this frame into the offscreen image."""
    job = self._job
    if job is None or self.failed or self._prepared_serial == self._job_serial:
      self._ready = None
      return  # the camera layer did not ask for a scene view last frame
    self._prepared_serial = self._job_serial
    self._guard(job.sm)
    x, y, w, h = job.rect
    size = (max(int(round(w * self.render_scale)), 1), max(int(round(h * self.render_scale)), 1))
    try:
      if self._target is None or (self._target.texture.width, self._target.texture.height) != size:
        self._close_target()
        self._target = rl.load_render_texture(*size)
        if not self._target.id:
          raise RuntimeError("could not create the offscreen scene image")
        rl.set_texture_filter(self._target.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
      rl.begin_texture_mode(self._target)
      try:
        rl.clear_background(rl.BLACK)
        self._render(job, rl.Rectangle(0, 0, size[0], size[1]), self.render_scale)
      finally:
        rl.end_texture_mode()
      self._ready = (job.selection, job.rect)
    except Exception:
      self._fail("offscreen render failed")

  def _guard(self, sm) -> None:
    """Back off when the driving model is losing frames while we draw (see GUARD_*)."""
    try:
      drop = float(sm["modelV2"].frameDropPerc) if sm.seen["modelV2"] else 0.0
    except (KeyError, AttributeError):
      return
    now = time.monotonic()
    if drop <= GUARD_DROP_PERCENT or now < self._guard_next:
      return
    self._guard_next = now + GUARD_SETTLE
    if self._guard_step + 1 < len(GUARD_SCALES):
      self._guard_step += 1
      self.render_scale = min(self.render_scale, GUARD_SCALES[self._guard_step])
      cloudlog.warning(f"scene views: model dropping {drop:.1f}% of frames, drawing at {self.render_scale:.2f} resolution")
    else:
      self._paused_until = now + GUARD_PAUSE
      self._job, self._ready = None, None
      cloudlog.warning(f"scene views: model still dropping {drop:.1f}% of frames, showing the stock view for {GUARD_PAUSE:.0f} s")

  def _fail(self, what: str) -> None:
    # Never take the driving UI down: log once and fall back to the stock layer for this session.
    rl.end_shader_mode()  # in case the failure happened between begin/end
    cloudlog.exception(f"scene views: {what}, falling back to the stock camera view")
    self.failed = True
    self._ready = None

  def _render(self, job: _Job, out: rl.Rectangle, scale: float | None) -> None:
    """Draw ``job`` into ``out``: the camera layer itself (scale None) or the offscreen image at ``scale``."""
    now = job.clock() if job.clock is not None else None
    now = now if now else time.monotonic()
    dt = 0.05 if self._last_time is None else min(max(now - self._last_time, 0.0), 0.5)
    self._last_time = now
    selection, sm = job.selection, job.sm

    rig = RIGS[selection.angle]
    if rig.angle != self._angle:
      rig.reset()
      self._angle = rig.angle

    feed = CameraFeed(job.view)
    wants_camera = selection.mode in (VizMode.CAMERA, VizMode.HYBRID)

    def camera_work():
      feed.update_calibration()
      if job.poll_camera:
        feed.select_stream(sm, rig.reprojection_stream if wants_camera else None, job.camera_view, job.reverse_driver)
        feed.poll()
      return feed.textures() if wants_camera else None
    frame = _phase("camera", camera_work)

    def scene_work():
      scene = self.tracker.update(sm, job.started_frame, dt)
      if scene.valid:
        scene.stop_sign = self.stop_signs.update(job.stop_sign, feed.road_view_from_calib(), scene.height, scene.v_ego, dt)
      source = feed.geometry()
      r = job.rect
      front = stock_front_camera(source.intrinsics, source.view_from_calib, r, source.wide)
      camera = rig.camera(RigContext(r, scene, front, dt))
      if scale is not None:
        camera = camera.reframed(r, scale)
      strength = 1.0
      if frame is not None:
        strength = job.color_strength() if callable(job.color_strength) else float(job.color_strength)
      self.renderer.render(out, rig, camera, scene, selection.mode, dt=dt, style=self.store.style(), road_style=job.road_style,
                           frame=frame, source=source, color_strength=strength, status=job.status, blink_time=now)
    _phase("model", scene_work)

  def render_switcher(self, x: float, y: float, fill=None, border=None, text=None, fonts=None) -> None:
    if self.failed:
      return
    try:
      self.controls.render(x, y, self.store.get(), fill, border, text, fonts=fonts, paused=self.paused)
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

  def _close_target(self) -> None:
    if self._target is not None and self._target.id:
      rl.unload_render_texture(self._target)
    self._target = None

  def close(self) -> None:
    self._close_target()
    self.renderer.close()


_shared: SceneViews | None = None


def shared() -> SceneViews:
  """The process-wide instance, so the camera layer, the widgets and the touch handler share one selection."""
  global _shared
  if _shared is None:
    _shared = SceneViews()
    try:
      from openpilot.system.ui.lib.application import gui_app
      gui_app.add_render_prepare(_shared.prepare, _shared.close)
    except Exception:
      cloudlog.exception("scene views: no render-prepare step, drawing scene views directly")
  return _shared


# ---- glue for the existing UI; ui_state is imported lazily so the classes above stay testable off-device


def render_layer(view, rect: rl.Rectangle, *, camera_view: int, reverse_driver: bool, poll_camera: bool,
                 road_style: dict | None = None, stop_sign: tuple | None = None) -> bool:
  """AugmentedRoadView hook. True when a scene view was drawn instead of the stock layer."""
  from openpilot.selfdrive.ui.ui_state import ui_state
  from openpilot.starpilot.ui.camera_color import road_color_strength
  # StarPort's car display renders this layer in its own read-only process; it keeps the stock view for now.
  projection = ui_state.projection_read_only and ui_state.replay_clock is None
  return shared().render(view, rect, ui_state.sm, camera_view=camera_view, reverse_driver=reverse_driver, poll_camera=poll_camera,
                         started_frame=ui_state.started_frame, status=ui_state.status.value,
                         color_strength=lambda: road_color_strength(ui_state), projection=projection, road_style=road_style,
                         stop_sign=stop_sign, clock=lambda: ui_state.display_time_ns() / 1e9)


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
