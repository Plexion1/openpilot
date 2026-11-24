"""Entry points used by StarPilot: decides whether a scene view replaces the stock camera layer and draws it.

Hooks (a few lines in existing files):
  * AugmentedRoadView.render_camera_model_layer -> render_layer() first; when it returns True the stock camera and
    model renderer are skipped and every StarPilot widget, alert and border still draws on top
  * OnroadView draws the scene view widgets (widgets.py) in its widget order: the VIEW / MODE switcher and the turn
    signal arrows
  * StarShellPage._handle_mouse_release -> handle_tap() before a tap toggles the sidebar
  * StarShellSession creates the shared instance at start (install) and adds the settings rows (settings.py) to the
    Onroad HUD page; AppearanceOwner lists them for Galaxy and applies their changes
  * RuntimeSnapshotAdapter hands over StarPilot's confirmed STOP sighting each frame (scene.note_stop_sign)

GPU budget: the driving model shares the GPU with the UI, so the 3D scene is drawn at RENDER_SCALE resolution into
an offscreen image during StarPilot's render-prepare step (gui_app.add_render_prepare, outside the frame, like the
torque bar and the map overlay) and the camera layer only stretches that image over its rectangle. StarPilot's
widgets, text and borders are still drawn at full resolution on top. On the frame a scene view is switched on, the
stock layer draws once while the first offscreen image is prepared; when only the angle or mode changes, the previous
image shows for that one frame. Shaders take a while to compile on first use, so after a scene view has been used once
they are compiled while the car is off, one per second, and switching views onroad never stalls the UI.

The time spent is reported through the UI's frame phases like the stock layer: camera polling as "camera", the
scene and its drawing as "model", so slow frame diagnostics compare directly.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from openpilot.common.swaglog import cloudlog
from openpilot.starpilot.ui.raylib import rl
from openpilot.starpilot.ui.viz.camera_feed import CameraFeed
from openpilot.starpilot.ui.viz.controls import VizControls
from openpilot.starpilot.ui.viz.options import SelectionStore, VizMode, VizSelection
from openpilot.starpilot.ui.viz.renderer import VizRenderer
from openpilot.starpilot.ui.viz.rigs import RigContext, make_rigs, stock_front_camera
from openpilot.starpilot.ui.viz.scene import SceneTracker, StopSignTracker, noted_stop_sign

# AugmentedRoadView camera_view values (appearance CameraView param) that keep the stock layer.
CAMERA_VIEW_DRIVER, CAMERA_VIEW_NONE = 1, 4
RENDER_SCALE = 0.5   # offscreen scene resolution relative to the camera layer (GPU cost scales with its square)

# GPU guard: the driving model shares the GPU. If it starts dropping frames while a scene view is on, draw the scene at
# a lower resolution; if it still drops at the lowest one, show the stock camera view for a while.
GUARD_SCALES = (RENDER_SCALE, 0.35, 0.25)
GUARD_DROP_PERCENT = 2.0  # modelV2.frameDropPerc (a filtered percentage; 0 when the model keeps up)
GUARD_SETTLE = 6.0        # s between steps, so the filtered drop rate can respond to the previous one
GUARD_PAUSE = 60.0        # s of stock view once the lowest resolution was not enough

LABEL_TOP_BAND = 0.28     # top of the layer where StarPilot's speed and cruise widgets sit: lead labels avoid it
WARM_UP_DELAY = 10.0      # s after UI start before shaders are compiled (offroad, once a scene view has been used)
WARM_UP_SPACING = 1.0     # s between programs, so the offroad UI never stalls for long


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
  lead_display: tuple | None = None


class SceneViews:
  def __init__(self, store: SelectionStore | None = None, render_scale: float = RENDER_SCALE):
    self.store = store or SelectionStore()
    self.rigs = make_rigs()
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
    self._prepare_seen = -1    # job serial at the last prepare call (prepare runs every frame when installed)
    self._ready: tuple | None = None  # (selection, rect) of the offscreen image drawn by this frame's prepare
    self._lead_label: tuple | None = None  # (text, above, below) for that image: StarPilot's Lead Info
    self._target = None
    self._guard_step = 0
    self._guard_next = 0.0     # monotonic time of the next allowed step
    self._paused_until = 0.0
    self.prepare_step = False  # set once gui_app runs prepare() every frame; without it frames are drawn directly
    self.warm_up_allowed = None  # callable: True when compiling shaders now cannot stall driving (shared() sets it)
    self.warm_up_egl: bool | None = None  # camera frame format to compile the camera program for
    self._warm = False
    self._warm_next = time.monotonic() + WARM_UP_DELAY

  def selection(self) -> VizSelection:
    return self.store.get()

  @property
  def paused(self) -> bool:
    """The GPU guard is showing the stock view for a while."""
    return time.monotonic() < self._paused_until

  def render(self, view, rect: rl.Rectangle, sm, *, camera_view: int, reverse_driver: bool, poll_camera: bool,
             started_frame: int, status: str, color_strength, projection: bool, road_style: dict | None = None,
             stop_sign: tuple | None = None, now: float | None = None, clock=None, lead_display: tuple | None = None) -> bool:
    """Draw the selected scene view. Returns False when the stock camera/model layer should draw instead.

    ``stop_sign`` is StarPilot's own confirmed STOP sighting (scene.stop_sign_sighting), passed through unchanged.
    ``lead_display`` is the camera layer's (show lead, Lead Info mode, metric): its text is drawn above the lead.
    ``now`` / ``clock`` give StarPilot's display clock in seconds (live: monotonic; replay: the recording's time), so
    motion, fades and blinking follow a replay's pace and pauses; without them the monotonic clock is used.
    """
    self._job = None
    if self.failed or projection or self.paused:
      self._ready = None
      return False
    try:
      selection = self.store.get()
      if selection.stock or camera_view in (CAMERA_VIEW_DRIVER, CAMERA_VIEW_NONE) or reverse_driver:
        self._last_time = None
        self._ready = None
        return False
      self._job_serial += 1
      self._job = _Job(view, (rect.x, rect.y, rect.width, rect.height), sm, selection, camera_view, reverse_driver,
                       poll_camera, started_frame, status, color_strength, road_style, stop_sign,
                       clock or ((lambda: now) if now else None), lead_display)
      if self._ready is not None and self._target is not None:
        # this frame's offscreen image (the previous angle or mode for one frame right after a switch)
        self._ready = None
        texture = self._target.texture
        rl.draw_texture_pro(texture, rl.Rectangle(0, 0, float(texture.width), -float(texture.height)), rect, rl.Vector2(0, 0), 0.0,
                            rl.WHITE)
        self._draw_lead_label(rect)
        return True
      if self.prepare_step and self._job_serial - self._prepare_seen <= 2:
        return False  # just switched on: the stock layer covers this one frame, the next prepare step draws the scene
      self._render(self._job, rect, None)  # no prepare step running (previews, tests): draw directly at full resolution
      self._draw_lead_label(rect)
    except Exception:
      self._fail("render failed")
      return False
    return True

  def prepare(self) -> None:
    """gui_app render-prepare step: draw the scene the camera layer will show this frame into the offscreen image."""
    self._prepare_seen = self._job_serial
    job = self._job
    if job is None or self.failed or self._prepared_serial == self._job_serial:
      self._ready = None  # the camera layer did not ask for a scene view last frame
      self._warm_up()
      return
    self._prepared_serial = self._job_serial
    if self._guard(job.sm):
      return
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
        rl.end_shader_mode()  # a no-op unless drawing failed between a shader's begin and end
        rl.end_texture_mode()
      self._ready = (job.selection, job.rect)
    except Exception:
      self._fail("offscreen render failed")

  def _warm_up(self) -> None:
    """Compile one missing shader every WARM_UP_SPACING while offroad, once a scene view has been used on this device."""
    if self._warm or self.failed or self.warm_up_allowed is None:
      return
    now = time.monotonic()
    if now < self._warm_next:
      return
    self._warm_next = now + WARM_UP_SPACING
    try:
      if self.warm_up_allowed() and self.store.path.exists():
        self._warm = self.renderer.warm_up(self.warm_up_egl)
    except Exception:
      cloudlog.exception("scene views: shader warm-up failed")
      self._warm = True  # compile on first use instead

  def _guard(self, sm) -> bool:
    """Back off when the driving model is losing frames while we draw (see GUARD_*). True when it just paused."""
    try:
      drop = float(sm["modelV2"].frameDropPerc) if sm.seen["modelV2"] else 0.0
    except (KeyError, AttributeError):
      return False
    now = time.monotonic()
    if drop <= GUARD_DROP_PERCENT or now < self._guard_next:
      return False
    self._guard_next = now + GUARD_SETTLE
    if self._guard_step + 1 < len(GUARD_SCALES):
      self._guard_step += 1
      self.render_scale = min(self.render_scale, GUARD_SCALES[self._guard_step])
      cloudlog.warning(f"scene views: model dropping {drop:.1f}% of frames, drawing at {self.render_scale:.2f} resolution")
      return False
    self._paused_until = now + GUARD_PAUSE
    self._job, self._ready, self._last_time = None, None, None
    cloudlog.warning(f"scene views: model still dropping {drop:.1f}% of frames, showing the stock view for {GUARD_PAUSE:.0f} s")
    return True

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

    rig = self.rigs[selection.angle]
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
      front = stock_front_camera(source.intrinsics, source.view_from_calib, r, source.wide) if rig.needs_front else None
      camera = rig.camera(RigContext(r, scene, front, dt))
      self._lead_label = self._lead_text(job, scene, camera)  # layer pixels: drawn sharp over the stretched image
      if scale is not None:
        camera = camera.reframed(r, scale)
      strength = 1.0
      if frame is not None:
        strength = job.color_strength() if callable(job.color_strength) else float(job.color_strength)
      self.renderer.render(out, rig, camera, scene, selection.mode, dt=dt, style=self.store.style(), road_style=job.road_style,
                           frame=frame, source=source, color_strength=strength, status=job.status, blink_time=now)
    _phase("model", scene_work)

  def _lead_text(self, job: _Job, scene, camera) -> tuple | None:
    """StarPilot's Lead Info (distance or speed, as on the stock lead marker): its text and the screen points just above
    the lead (the top of its roof as seen from this angle) and just below its rear bumper (layer pixels; None where
    behind the camera)."""
    display = job.lead_display
    if display is None or not display[0] or not scene.valid or not scene.leads or not scene.leads[0].primary:
      return None
    from openpilot.starpilot.ui.appearance_preferences import format_lead_info
    from openpilot.starpilot.ui.viz.renderer import VEHICLE_MODELS
    sm = job.sm
    radar = sm["radarState"] if sm.seen["radarState"] and sm.valid["radarState"] else None
    text = format_lead_info(radar.leadOne if radar is not None else None, display[1], display[2])
    if not text:
      return None
    lead = scene.leads[0]
    _model, (length, _width, height), _wheel, _gloss = VEHICLE_MODELS[self.store.style().lead.model]
    c, s = math.cos(lead.heading), math.sin(lead.heading)
    x, y, z = float(lead.rear[0]), float(lead.rear[1]), float(lead.rear[2])
    # the middle and the front of the roof, just above it: whichever is higher on screen is the top of the car from this
    # angle (its roof seen from behind, its nose seen from straight above)
    roof = [point for point in (camera.project_point((x + c * length / 2, y + s * length / 2, z - height - 0.15)),
                                camera.project_point((x + c * length, y + s * length, z - height - 0.15))) if point is not None]
    bumper = camera.project_point((x - c * 0.3, y - s * 0.3, z))
    if not roof:
      return None
    top = min(roof, key=lambda point: point[1])
    return text, top[:2], bumper[:2] if bumper is not None else None

  def _draw_lead_label(self, rect: rl.Rectangle) -> None:
    """Like the stock marker's Lead Info: semi-bold 48 px, white with a thin black outline, kept inside the layer. It
    sits above the lead, or below it where that would cover the speed and cruise widgets at the top of the screen."""
    label = self._lead_label
    if label is None:
      return
    text, (x, roof_y), bumper = label
    if not (rect.x <= x <= rect.x + rect.width and rect.y <= roof_y <= rect.y + rect.height):
      return
    from openpilot.system.ui.lib.application import FontWeight, gui_app
    from openpilot.system.ui.lib.text_measure import measure_text_cached
    font = gui_app.font(FontWeight.SEMI_BOLD)
    size = measure_text_cached(font, text, 48)
    y = roof_y - size.y - 6
    if y < rect.y + rect.height * LABEL_TOP_BAND and bumper is not None:
      x, y = bumper[0], bumper[1] + 8
    x = min(max(x - size.x / 2, rect.x), rect.x + rect.width - size.x)
    y = min(max(y, rect.y), rect.y + rect.height - size.y)
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
      rl.draw_text_ex(font, text, rl.Vector2(x + dx, y + dy), 48, 0, rl.BLACK)
    rl.draw_text_ex(font, text, rl.Vector2(x, y), 48, 0, rl.WHITE)

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
      from openpilot.common.hardware import COMMA_HARDWARE
      from openpilot.system.ui.lib.application import gui_app
      gui_app.add_render_prepare(_shared.prepare, _shared.close)
      _shared.prepare_step = True
      _shared.warm_up_allowed = _offroad
      _shared.warm_up_egl = bool(COMMA_HARDWARE)  # CameraView uses EGL frames on comma hardware
    except Exception:
      cloudlog.exception("scene views: no render-prepare step, drawing scene views directly")
  return _shared


def _offroad() -> bool:
  from openpilot.selfdrive.ui.ui_state import ui_state
  return not ui_state.started and ui_state.replay_clock is None


def install() -> None:
  """Create the shared instance at UI start so its prepare step can compile shaders before the first drive."""
  try:
    shared()
  except Exception:
    cloudlog.exception("scene views: install failed")


# ---- glue for the existing UI; ui_state is imported lazily so the classes above stay testable off-device


def render_layer(view, rect: rl.Rectangle, *, camera_view: int, reverse_driver: bool, poll_camera: bool,
                 road_style: dict | None = None, lead_display: tuple | None = None) -> bool:
  """AugmentedRoadView hook. True when a scene view was drawn instead of the stock layer; any error keeps the stock one."""
  global _hook_failed
  if _hook_failed:
    return False
  try:
    from openpilot.selfdrive.ui.ui_state import ui_state
    from openpilot.starpilot.ui.camera_color import road_color_strength
    return shared().render(view, rect, ui_state.sm, camera_view=camera_view, reverse_driver=reverse_driver, poll_camera=poll_camera,
                           started_frame=ui_state.started_frame, status=ui_state.status.value,
                           color_strength=lambda: road_color_strength(ui_state), projection=projecting(), road_style=road_style,
                           stop_sign=noted_stop_sign(), clock=lambda: ui_state.display_time_ns() / 1e9, lead_display=lead_display)
  except Exception:
    cloudlog.exception("scene views: camera layer hook failed, keeping the stock camera view")
    _hook_failed = True
    return False


_hook_failed = False


def projecting() -> bool:
  """StarPort's car display (its own read-only process) keeps the stock camera view for now."""
  from openpilot.selfdrive.ui.ui_state import ui_state
  return bool(ui_state.projection_read_only) and ui_state.replay_clock is None


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
