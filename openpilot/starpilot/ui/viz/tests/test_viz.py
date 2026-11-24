"""Scene view math, rigs, scene extraction, settings and widget registration. No GPU or raylib window needed."""
import copy
import json
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.starpilot.ui.feature_settings_state import FeatureSettingsRequest
from openpilot.starpilot.ui.viz import settings, shaders
from openpilot.starpilot.ui.viz.geometry import look_at, top_down_rotation
from openpilot.starpilot.ui.viz.options import SelectionStore, VehicleModel, ViewAngle, VizMode, VizSelection
from openpilot.starpilot.ui.viz.rigs import RIGS, RigContext, stock_front_camera
from openpilot.starpilot.ui.viz.scene import (LEAD_BRAKE_HOLD, RADAR_TO_CAMERA, SAMPLE_X, SAMPLES, STOP_MIN_AHEAD,
                                              STOP_SIGN_KEEP, Scene, SceneTracker, StopSign, StopSignTracker, stop_sign_sighting)
from openpilot.starpilot.ui.viz.widgets import SCENE_VIEW_WIDGETS, SWITCHER_WIDGET, TURN_SIGNAL_WIDGET

RECT = (30.0, 30.0, 2100.0, 1020.0)
NARROW = DEVICE_CAMERAS[("tizi", "ox03c10")].narrow_road


def _front():
  view_from_calib = view_frame_from_device_frame @ rot_from_euler([0.0, 0.0067, 0.0213])
  return stock_front_camera(NARROW.intrinsics, view_from_calib, RECT, wide=False)


def test_uv_ray_inverts_projection():
  camera = RIGS[ViewAngle.CHASE].camera(RigContext(RECT, Scene(valid=True), _front(), 0.05))
  point = np.array([[18.0, -2.5, 1.22]])
  screen, depth = camera.project(point)
  u, v = (screen[0, 0] - RECT[0]) / RECT[2], (screen[0, 1] - RECT[1]) / RECT[3]
  ray = camera.uv_to_ray(RECT) @ np.array([u, v, 1.0])
  t = (1.22 - camera.position[2]) / ray[2]
  np.testing.assert_allclose(camera.position + ray * t, point[0], atol=1e-6)
  assert depth[0] > 0


def test_look_at_is_a_rotation():
  rotation = look_at([-8.0, 0.0, -2.5], [12.0, 0.0, 1.2])
  np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
  assert np.isclose(np.linalg.det(rotation), 1.0)
  np.testing.assert_allclose(top_down_rotation() @ top_down_rotation().T, np.eye(3))


def test_rigs_frame_the_car():
  for v_ego in (0.0, 30.0):
    scene = Scene(valid=True, v_ego=v_ego)
    for angle in (ViewAngle.CHASE, ViewAngle.TOP):
      RIGS[angle].reset()
      camera = RIGS[angle].camera(RigContext(RECT, scene, _front(), 0.05))
      front = camera.project_point([RADAR_TO_CAMERA, 0.0, scene.height])
      rear = camera.project_point(scene.ego_ground - np.array([2.3, 0.0, 0.0]))
      assert RECT[0] + RECT[2] * 0.4 < front[0] < RECT[0] + RECT[2] * 0.6, angle
      assert RECT[1] + RECT[3] * 0.5 < front[1] < rear[1] < RECT[1] + RECT[3], (angle, v_ego)  # whole car in the lower half
      ahead = camera.project_point([40.0, 0.0, scene.height])
      assert ahead is not None and ahead[1] < front[1], angle  # forward is up the screen


def test_front_rig_is_the_stock_camera():
  front = _front()
  assert RIGS[ViewAngle.FRONT].camera(RigContext(RECT, Scene(valid=True), front, 0.05)) is front
  vanishing = front.project_point([1000.0, 0.0, 0.0])
  assert abs(vanishing[0] - (RECT[0] + RECT[2] / 2)) < RECT[2] * 0.1


def test_reprojection_uses_only_the_road_camera():
  assert RIGS[ViewAngle.CHASE].reprojection_stream == RIGS[ViewAngle.TOP].reprojection_stream == "narrow"
  assert RIGS[ViewAngle.FRONT].reprojection_stream is None  # the front view keeps StarPilot's own camera choice


def test_shader_constants_match_the_scene():
  assert shaders.SAMPLES == SAMPLES
  assert f"#define SAMPLES {SAMPLES}" in shaders.SYNTHETIC_GROUND_SHADER
  assert "texture0" in shaders.camera_ground_shader(True, "vec3 road_color(vec3 c, float s) { return c; }")


class _Msg(SimpleNamespace):
  def __str__(self):
    return self.name if hasattr(self, "name") else super().__str__()


def _line(y, n=33):
  xs = np.linspace(0.0, 100.0, n)
  return _Msg(x=list(xs), y=[y] * n, z=[0.0] * n)


def _lead(d, v_lead=20.0, a_lead=0.0):
  return _Msg(present=d is not None, dRel=d[0] if d else 0.0, yRel=d[1] if d else 0.0, vRel=-1.0, vLead=v_lead, aLeadK=a_lead,
              radar=False)


class _SM:
  """The SubMaster surface SceneTracker reads."""

  def __init__(self, lead_one=(20.0, 0.5), lead_two=(21.0, 0.4), speeds=None, should_stop=False, lane_change="off"):
    speeds = speeds if speeds is not None else [15.0] * 33
    self.data = {
      "modelV2": _Msg(position=_line(0.0), laneLines=[_line(-5.4), _line(-1.8), _line(1.8), _line(5.4)],
                      laneLineProbs=[0.5, 0.9, 0.9, 0.5], roadEdges=[_line(-7.0), _line(2.5)], roadEdgeStds=[0.3, 0.3],
                      acceleration=_Msg(x=[0.5] * 33), velocity=_Msg(x=speeds),
                      meta=_Msg(laneChangeState=_Msg(name=lane_change), laneChangeDirection=_Msg(name="left")),
                      action=_Msg(shouldStop=should_stop)),
      "radarState": _Msg(leadOne=_lead(lead_one), leadTwo=_lead(lead_two)),
      "extrinsicsCalibration": _Msg(height=[1.25]),
      "carState": _Msg(vEgo=15.0, aEgo=0.0, leftBlinker=True, rightBlinker=False, leftBlindspot=False, rightBlindspot=False,
                       standstill=False, gearShifter=_Msg(name="drive"), brakePressed=False, regenBraking=False),
      "selfdriveState": _Msg(experimentalMode=False), "longitudinalPlan": _Msg(allowThrottle=True),
      "narrowRoadCameraState": _Msg(exposureValPercent=10.0),
    }
    names = list(self.data)
    self.updated, self.recv_frame = dict.fromkeys(names, True), dict.fromkeys(names, 1)
    self.seen, self.valid = dict.fromkeys(names, True), dict.fromkeys(names, True)

  def __getitem__(self, name):
    return self.data[name]


def test_scene_tracker_extracts_geometry_and_leads():
  scene = SceneTracker().update(_SM())
  assert scene.valid and scene.height == 1.25
  assert np.all(scene.path[:, 2] == 1.25)  # flat road under the camera
  assert len(scene.leads) == 1  # leadTwo 1 m behind leadOne is the same car
  lead = scene.leads[0]
  assert lead.rear[1] == -0.5  # radar y is left-positive, calib y is right-positive
  assert lead.rear[0] == RADAR_TO_CAMERA + 20.0
  assert scene.path_end == lead.rear[0] - 0.5  # the path stops at the lead
  assert scene.signals.left_blinker and not scene.signals.right_blinker
  assert scene.lane_change == 0


def test_road_samples_follow_the_model_lines():
  scene = SceneTracker().update(_SM())
  assert scene.lane_samples.shape == scene.road_samples.shape == (SAMPLES, 4)
  np.testing.assert_allclose(scene.lane_samples[5], [-5.4, -1.8, 1.8, 5.4], atol=1e-5)
  np.testing.assert_allclose(scene.road_samples[5, :3], [-7.0, 2.5, 0.0], atol=1e-5)
  assert SAMPLE_X[0] == 0.0 and np.all(np.diff(SAMPLE_X) > 0) and np.all(np.diff(SAMPLE_X, 2) > 0)  # dense near the car


def test_scene_tracker_keeps_distinct_second_lead_and_caches():
  tracker = SceneTracker()
  sm = _SM(lead_two=(45.0, -3.0))
  first = tracker.update(sm)
  assert len(first.leads) == 2
  assert tracker.update(sm).revision == first.revision  # nothing changed, nothing rebuilt
  sm.recv_frame["modelV2"] = 2
  assert tracker.update(sm).revision == first.revision + 1


def test_stop_point_comes_from_the_predicted_speed():
  speeds = list(np.linspace(10.0, 0.0, 33))
  scene = SceneTracker().update(_SM(lead_one=None, lead_two=None, speeds=speeds, should_stop=True))
  first_slow = int(np.flatnonzero(np.array(speeds) < 0.5)[0])
  assert scene.stop_x == scene.path[first_slow, 0]
  assert scene.path_end == scene.stop_x
  held = SceneTracker().update(_SM(lead_one=None, lead_two=None, speeds=[0.0] * 33, should_stop=True))
  assert held.stop_x == RADAR_TO_CAMERA + STOP_MIN_AHEAD  # stopped at it: just ahead of the bumper
  assert SceneTracker().update(_SM(lead_one=None, lead_two=None, speeds=speeds, should_stop=False)).stop_x is None


def test_lead_brake_lights_latch():
  tracker = SceneTracker()
  sm = _SM(lead_two=None)
  assert not tracker.update(sm).leads[0].braking
  sm.data["radarState"] = _Msg(leadOne=_lead((20.0, 0.5), a_lead=-1.5), leadTwo=_lead(None))
  sm.recv_frame["radarState"] = 2
  assert tracker.update(sm).leads[0].braking
  sm.data["radarState"] = _Msg(leadOne=_lead((20.0, 0.5), a_lead=0.3), leadTwo=_lead(None))
  sm.recv_frame["radarState"] = 3
  assert tracker.update(sm, dt=LEAD_BRAKE_HOLD / 4).leads[0].braking  # held for a moment
  sm.recv_frame["radarState"] = 4
  assert not tracker.update(sm, dt=LEAD_BRAKE_HOLD).leads[0].braking
  sm.data["radarState"] = _Msg(leadOne=_lead((8.0, 0.0), v_lead=0.0), leadTwo=_lead(None))
  sm.recv_frame["radarState"] = 5
  assert tracker.update(sm).leads[0].braking  # stopped cars show brake lights


def test_darkness_follows_camera_exposure_with_hysteresis():
  tracker = SceneTracker()
  sm = _SM()
  assert tracker.update(sm).darkness == 0.0
  sm.data["narrowRoadCameraState"] = _Msg(exposureValPercent=95.0)
  for _ in range(200):
    scene = tracker.update(sm, dt=0.05)
  assert scene.darkness > 0.95
  sm.data["narrowRoadCameraState"] = _Msg(exposureValPercent=60.0)  # between the thresholds: stays dark
  assert tracker.update(sm, dt=0.05).darkness > 0.95


def test_lane_change_state_and_ego_brakes():
  sm = _SM(lane_change="laneChangeStarting")
  sm.data["carState"].brakePressed = True
  scene = SceneTracker().update(sm)
  assert scene.lane_change == 2 and scene.lane_change_direction == -1
  assert scene.signals.braking


def test_selection_store_round_trip(tmp_path):
  path = tmp_path / "scene_views.json"
  path.write_text(json.dumps({"angle": "top", "mode": "synthetic", "leadModel": "truck", "futureOption": 3}))
  store = SelectionStore(path, poll_interval=0.0)
  assert store.get() == VizSelection(ViewAngle.TOP, VizMode.SYNTHETIC)
  assert store.style().lead.model == VehicleModel.TRUCK and store.style().ego.model == VehicleModel.CAR
  store.set(store.get().next_angle())
  saved = json.loads(path.read_text())
  assert saved["angle"] == "front" and saved["futureOption"] == 3 and saved["leadModel"] == "truck"  # keeps other keys
  assert SelectionStore(path, poll_interval=0.0).get().angle == ViewAngle.FRONT


def test_selection_store_ignores_bad_files(tmp_path):
  path = tmp_path / "scene_views.json"
  path.write_text("{not json")
  assert SelectionStore(path, poll_interval=0.0).get().stock
  path.write_text(json.dumps({"angle": "sideways", "mode": 7, "egoColor": "Plaid", "egoModel": "boat"}))
  store = SelectionStore(path, poll_interval=0.0)
  assert store.get().stock and store.style().ego.color == "Ocean Blue" and store.style().ego.model == VehicleModel.CAR


def test_cycle_orders():
  selection = VizSelection()
  assert selection.stock
  assert [selection := selection.next_mode() for _ in range(3)][-1].mode == VizMode.CAMERA
  assert VizSelection().next_angle().angle == ViewAngle.CHASE


def test_settings_rows_save_to_the_scene_view_document(tmp_path):
  path = tmp_path / "scene_views.json"
  rows = {row.key: row for row in settings.rows(parked=True, path=path)}
  assert rows["SceneViews:egoModel"].value == "Car" and rows["SceneViews:angle"].value == "Front"
  assert rows["SceneViews:egoColor"].default_value == "Ocean Blue"
  request = FeatureSettingsRequest("SceneViews:leadModel", rows["SceneViews:leadModel"].source, "Truck")
  assert not settings.apply(request, parked=lambda: False, path=path)  # parked only
  assert settings.apply(request, parked=lambda: True, path=path)
  assert json.loads(path.read_text())["leadModel"] == "truck"
  assert not settings.apply(request, parked=lambda: True, path=path)  # stale source: the document changed
  rows = {row.key: row for row in settings.rows(parked=True, path=path)}
  assert rows["SceneViews:leadModel"].value == "Truck"
  assert settings.apply(FeatureSettingsRequest("SceneViews:angle", rows["SceneViews:angle"].source, "Chase"), lambda: True, path)
  assert SelectionStore(path, poll_interval=0.0).get().angle == ViewAngle.CHASE
  bad = FeatureSettingsRequest("SceneViews:egoColor", settings.rows(True, path)[0].source, "Plaid")
  assert not settings.apply(bad, lambda: True, path)


def test_widgets_are_registered_for_the_big_layout_and_starport():
  from openpilot.starpilot.ui.onroad_customization import (DEFAULT_WIDGET_ORDER, PROFILES, customization_metadata, default_document,
                                                           validate_document)
  large = PROFILES["large"]["widgets"]
  assert {TURN_SIGNAL_WIDGET, SWITCHER_WIDGET} <= set(large) and not set(SCENE_VIEW_WIDGETS) & set(PROFILES["compact"]["widgets"])
  assert set(DEFAULT_WIDGET_ORDER["large"]) == set(large)
  galaxy = customization_metadata()["profiles"]["large"]
  assert set(SCENE_VIEW_WIDGETS) <= set(galaxy["widgets"]) and set(galaxy["widgetOrder"]) == set(galaxy["widgets"])

  document = default_document()
  assert document["layouts"]["large"][SWITCHER_WIDGET]["enabled"]
  assert validate_document(document) == document
  older = copy.deepcopy(document)
  for key in SCENE_VIEW_WIDGETS:
    del older["layouts"]["large"][key]
  older["widgetOrder"] = {"large": [key for key in DEFAULT_WIDGET_ORDER["large"] if key not in SCENE_VIEW_WIDGETS]}
  upgraded = validate_document(older)  # a layout saved before the widgets existed gains them
  assert upgraded["layouts"]["large"][TURN_SIGNAL_WIDGET] == large[TURN_SIGNAL_WIDGET]["default"]
  assert set(SCENE_VIEW_WIDGETS) <= set(upgraded["widgetOrder"]["large"])


@pytest.mark.parametrize("viewport", [(1860, 1080), (1920, 1080), (2880, 1080)])
def test_starport_layout_has_the_widgets_where_the_car_screen_draws_them(viewport):
  from openpilot.starpilot.system.starport.projection_layout import (default_layout_for_viewport, layout_metadata_for_viewport,
                                                                     projection_customization, validate_layout_for_viewport)
  from openpilot.starpilot.ui.onroad_customization import default_document
  from openpilot.starpilot.ui.viz.widgets import ANCHOR_SHIFT
  metadata = layout_metadata_for_viewport(viewport)
  assert set(SCENE_VIEW_WIDGETS) <= set(metadata["widgets"]) and set(SCENE_VIEW_WIDGETS) <= set(metadata["widgetOrder"])
  # the car screen keeps the stock camera view: no VIEW / MODE buttons there, and they are not offered in its editor
  assert metadata["widgets"][SWITCHER_WIDGET]["hidden"] and not metadata["widgets"][SWITCHER_WIDGET]["default"]["enabled"]
  assert metadata["widgets"][TURN_SIGNAL_WIDGET]["default"]["enabled"]

  document = default_layout_for_viewport(viewport)
  older = copy.deepcopy(document)
  for key in SCENE_VIEW_WIDGETS:
    del older["widgets"][key]
  assert validate_layout_for_viewport(older, viewport)["widgets"][TURN_SIGNAL_WIDGET] == document["widgets"][TURN_SIGNAL_WIDGET]

  # the projected view adds the widget's anchor shift back, so it lands where the car screen's editor shows it
  converted = projection_customization(document, default_document())["layouts"]["large"]
  for key in SCENE_VIEW_WIDGETS:
    drawn_x = converted[key]["x"] + ANCHOR_SHIFT[key] * (viewport[0] - 1860)
    assert drawn_x == document["widgets"][key]["x"]


def test_shaders_compile_as_glsl_es(tmp_path):
  """The comma's GPU only takes GLSL ES; check every shader with the Khronos reference compiler when it is installed."""
  validator = shutil.which("glslangValidator")
  if validator is None:
    pytest.skip("glslangValidator not installed")
  road_color = "highp vec3 road_color(highp vec3 rgb, highp float strength) { return rgb * strength; }"
  sources = {"quad.vert": shaders.VERTEX_SHADER, "ground.frag": shaders.SYNTHETIC_GROUND_SHADER,
             "camera.frag": shaders.camera_ground_shader(False, road_color), "camera_egl.frag": shaders.camera_ground_shader(True, road_color),
             "vehicle.frag": shaders.VEHICLE_SHADER}
  for name, source in sources.items():
    path = tmp_path / name
    path.write_text(source)
    result = subprocess.run([validator, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, f"{name}: {result.stdout}"


def test_stop_sign_comes_from_starpilots_own_confirmation():
  from openpilot.starpilot.speed_limits.vision.stop_observation import STOP_RANGE_OFFSET_M, StopBox, StopEvidence
  sample = StopEvidence("session", "road", 1_000_000_000, 1, 7, 1928, 1208,
                        (StopBox(0.65, (100, 100, 120, 120)), StopBox(0.9, (1164 + 100, 554 - 20, 1164 + 140, 554 + 20))))
  confirmed = SimpleNamespace(confirmed_ns=sample.observed_ns, distance_m=22.0, previous=sample)
  depth, across, up, age = stop_sign_sighting(confirmed, now_ns=1_250_000_000)
  assert depth == 22.0 - STOP_RANGE_OFFSET_M  # StarPilot's distance is where the car stops, past the sign
  assert np.isclose(across, (1164 + 120 - 964) / 2648.0) and np.isclose(up, (554 - 604) / 2648.0)  # most confident box
  assert age == 0.25
  assert stop_sign_sighting(SimpleNamespace(confirmed_ns=0, distance_m=None, previous=sample), now_ns=0) is None


def test_stop_sign_stays_on_the_road_while_we_drive_past():
  tracker = StopSignTracker()
  calib = view_frame_from_device_frame  # straight-ahead calibration
  sighting = (20.0, 0.1, -0.05, 0.0)
  sign = tracker.update(sighting, calib, 1.2, v_ego=0.0, dt=0.05)
  np.testing.assert_allclose(sign.position, [20.0, 2.0, 1.2], atol=1e-9)
  for _ in range(10):  # the same sighting again: only our own travel moves it
    sign = tracker.update(sighting, calib, 1.2, v_ego=5.0, dt=0.1)
  np.testing.assert_allclose(sign.position[:2], [15.0, 2.0], atol=1e-9)
  assert sign.opacity > 0.9
  for _ in range(int(STOP_SIGN_KEEP / 0.1) + 20):  # out of sight for good: fades away
    sign = tracker.update(None, calib, 1.2, v_ego=0.0, dt=0.1)
  assert sign is None
  assert isinstance(tracker.update((10.0, 0.0, 0.0, 0.0), calib, 1.2, 0.0, 0.05), StopSign)  # a new sign starts afresh


def test_gpu_guard_backs_off_when_the_model_drops_frames(tmp_path):
  from openpilot.starpilot.ui.viz import controller
  views = controller.SceneViews(SelectionStore(tmp_path / "scene_views.json", poll_interval=0.0))
  sm = _SM()
  sm.data["modelV2"].frameDropPerc = 0.5
  views._guard(sm)
  assert views.render_scale == controller.RENDER_SCALE  # the model keeps up: nothing changes
  sm.data["modelV2"].frameDropPerc = 8.0
  scales = []
  for _ in range(len(controller.GUARD_SCALES) - 1):
    views._guard(sm)
    views._guard(sm)  # within the settle time: no second step
    scales.append(views.render_scale)
    views._guard_next = 0.0
  assert scales == list(controller.GUARD_SCALES[1:])
  views._guard(sm)  # lowest resolution still too much: the stock view takes over for a while
  assert views._paused_until > 0
  assert not views.render(None, None, sm, camera_view=2, reverse_driver=False, poll_camera=False, started_frame=0, status="engaged",
                          color_strength=1.0, projection=False)


class _FakeGL:
  """The few raylib calls the controller makes around the offscreen image, recorded instead of drawn."""

  def __init__(self, monkeypatch):
    from openpilot.starpilot.ui.raylib import rl
    self.drawn = []
    texture = SimpleNamespace(width=1050, height=510, id=1)
    for name in ("begin_texture_mode", "end_texture_mode", "clear_background", "set_texture_filter", "unload_render_texture", "end_shader_mode"):
      monkeypatch.setattr(rl, name, lambda *a, **k: None)
    monkeypatch.setattr(rl, "load_render_texture", lambda w, h: SimpleNamespace(id=1, texture=SimpleNamespace(width=w, height=h, id=1)))
    monkeypatch.setattr(rl, "draw_texture_pro", lambda *a, **k: self.drawn.append("image"))
    self.texture = texture


def _render(views, sm):
  from openpilot.starpilot.ui.raylib import rl
  return views.render(None, rl.Rectangle(30, 30, 2100, 1020), sm, camera_view=2, reverse_driver=False, poll_camera=False,
                      started_frame=0, status="engaged", color_strength=1.0, projection=False)


def test_switching_on_shows_the_stock_layer_for_one_frame_then_the_offscreen_image(tmp_path, monkeypatch):
  from openpilot.starpilot.ui.viz import controller
  gl = _FakeGL(monkeypatch)
  store = SelectionStore(tmp_path / "scene_views.json", poll_interval=0.0)
  store.set(VizSelection(ViewAngle.CHASE, VizMode.SYNTHETIC))
  views = controller.SceneViews(store)
  views.prepare_step = True
  rendered = []
  monkeypatch.setattr(views, "_render", lambda job, out, scale: rendered.append(scale))
  sm = _SM()
  views.prepare()             # frame 1: nothing asked for yet
  assert not _render(views, sm)  # the stock layer draws this one frame; nothing is drawn at full resolution
  assert rendered == [] and gl.drawn == []
  views.prepare()             # frame 2: the scene goes into the half resolution image
  assert rendered == [controller.RENDER_SCALE]
  assert _render(views, sm) and gl.drawn == ["image"]
  store.set(VizSelection(ViewAngle.TOP, VizMode.SYNTHETIC))
  views.prepare()
  assert _render(views, sm)   # a new angle reuses the image of this frame's prepare step (one frame late at most)

  direct = controller.SceneViews(store)  # no prepare step running (previews, tests): drawn directly
  monkeypatch.setattr(direct, "_render", lambda job, out, scale: rendered.append(scale))
  assert _render(direct, sm) and rendered[-1] is None


def test_paused_guard_skips_the_frame_and_never_shows_a_stale_image(tmp_path, monkeypatch):
  from openpilot.starpilot.ui.viz import controller
  _FakeGL(monkeypatch)
  store = SelectionStore(tmp_path / "scene_views.json", poll_interval=0.0)
  store.set(VizSelection(ViewAngle.CHASE, VizMode.SYNTHETIC))
  views = controller.SceneViews(store)
  views.prepare_step = True
  rendered = []
  monkeypatch.setattr(views, "_render", lambda job, out, scale: rendered.append(scale))
  sm = _SM()
  _render(views, sm)
  views._guard_step = len(controller.GUARD_SCALES) - 1  # already at the lowest resolution
  sm.data["modelV2"].frameDropPerc = 8.0
  views.prepare()
  assert rendered == [] and views.paused and views._ready is None
  assert not _render(views, sm)


def test_warm_up_compiles_one_program_per_step_and_only_after_a_scene_view_was_used(tmp_path, monkeypatch):
  from openpilot.starpilot.ui.viz import controller, renderer
  compiled = []

  class FakeProgram:
    def __init__(self, fragment, names):
      compiled.append(names)
      self.valid = True

  monkeypatch.setattr(renderer, "_Program", FakeProgram)
  monkeypatch.setattr(renderer.rl, "gen_image_color", lambda *a: None)
  monkeypatch.setattr(renderer.rl, "load_texture_from_image", lambda image: SimpleNamespace(id=1))
  monkeypatch.setattr(renderer.rl, "unload_image", lambda image: None)
  store = SelectionStore(tmp_path / "scene_views.json", poll_interval=0.0)
  views = controller.SceneViews(store)
  views.warm_up_allowed, views.warm_up_egl = (lambda: True), True
  steps = []
  for _ in range(6):
    views._warm_next = 0.0
    views.prepare()
    steps.append(len(compiled))
  assert steps == [0] * 6  # never used on this device: nothing is compiled
  store.set(VizSelection(ViewAngle.CHASE, VizMode.SYNTHETIC))
  steps = []
  for _ in range(5):
    views._warm_next = 0.0
    views.prepare()
    steps.append(len(compiled))
  assert steps == [1, 2, 3, 3, 3] and views._warm  # ground, vehicles, camera: one per step, then done


def test_lead_info_is_drawn_above_the_3d_lead_like_the_stock_marker(tmp_path):
  from openpilot.starpilot.ui.appearance_preferences import LeadInfoMode
  from openpilot.starpilot.ui.viz import controller
  views = controller.SceneViews(SelectionStore(tmp_path / "scene_views.json", poll_interval=0.0))
  sm = _SM(lead_one=(30.0, 0.0))
  scene = SceneTracker().update(sm)
  camera = RIGS[ViewAngle.CHASE].camera(RigContext(RECT, scene, None, 0.05))
  job = SimpleNamespace(sm=sm, lead_display=(True, LeadInfoMode.DISTANCE, False))
  text, (x, y), (below_x, below_y) = views._lead_text(job, scene, camera)
  assert text == "98 ft"  # StarPilot's own format_lead_info for 30 m
  lead_x, lead_y, _ = camera.project_point(scene.leads[0].rear)
  assert abs(x - lead_x) < 40 and y < lead_y < below_y  # centred over the lead: above its roof, or under its bumper
  job.lead_display = (True, LeadInfoMode.SPEED, True)
  assert views._lead_text(job, scene, camera)[0].endswith("km/h")
  for display in (None, (True, LeadInfoMode.OFF, False), (False, LeadInfoMode.DISTANCE, False)):
    job.lead_display = display
    assert views._lead_text(job, scene, camera) is None  # Lead Info off: nothing, as on the stock view


def test_top_view_keeps_the_lead_below_the_speed_and_cruise_widgets():
  from openpilot.starpilot.ui.viz.controller import LABEL_TOP_BAND
  from openpilot.starpilot.ui.viz.rigs import TopDownRig
  rig = RIGS[ViewAngle.TOP]
  for distance in (10.0, 25.0, 40.0):
    sm = _SM(lead_one=(distance, 0.0))
    scene = SceneTracker().update(sm)
    rig.reset()
    camera = rig.camera(RigContext(RECT, scene, None, 0.05))
    lead = scene.leads[0]
    front = camera.project_point(lead.rear + np.array([4.6, 0.0, 0.0]))
    assert front[1] > RECT[1] + RECT[3] * (1.0 - TopDownRig.content_top) - 1.0, distance
    assert front[1] > RECT[1] + RECT[3] * LABEL_TOP_BAND, distance  # clear of the widgets at the top of the screen
