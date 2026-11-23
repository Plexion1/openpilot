"""Scene view math, rigs, scene extraction and selection storage. No GPU or raylib needed."""
import json
from types import SimpleNamespace

import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.starpilot.ui.viz.geometry import (ground_disc, look_at, offset_line, project_ribbon,
                                                 top_down_rotation)
from openpilot.starpilot.ui.viz.options import SelectionStore, ViewAngle, VizMode, VizSelection
from openpilot.starpilot.ui.viz.rigs import RIGS, RigContext, stock_front_camera
from openpilot.starpilot.ui.viz.scene import BEHIND_EXTENSION, RADAR_TO_CAMERA, Scene, SceneTracker

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


def test_ribbon_clipped_at_near_plane_reaches_screen_bottom():
  front = _front()
  line = np.column_stack((np.linspace(-15, 60, 30), np.full(30, 1.5), np.full(30, 1.22)))
  polygon = project_ribbon(front, *offset_line(line, 0.1))
  assert polygon is not None and np.isfinite(polygon).all()
  assert polygon[:, 1].max() > RECT[1] + RECT[3]  # starts below the layer, like the stock lines


def test_offset_line_is_perpendicular_to_heading():
  line = np.array([[0.0, 0.0, 1.2], [0.0, 10.0, 1.2]])  # heading right
  left, right = offset_line(line, 0.5)
  np.testing.assert_allclose(right[:, 0] - left[:, 0], [-1.0, -1.0])  # right of travel is backwards
  np.testing.assert_allclose(right[:, 1] - left[:, 1], [0.0, 0.0], atol=1e-12)


def test_ground_disc_strip_covers_every_rim_point():
  disc = ground_disc((10.0, 0.0, 1.2), 1.0, segments=20)
  assert len(disc) == 20 and len({tuple(p) for p in disc}) == 20


def test_rigs_frame_the_car():
  scene = Scene(valid=True, v_ego=20.0)
  ctx = RigContext(RECT, scene, _front(), 0.05)
  for angle in (ViewAngle.CHASE, ViewAngle.TOP):
    camera = RIGS[angle].camera(ctx)
    x, y, _ = camera.project_point(scene.ego_center)
    assert RECT[0] + RECT[2] * 0.4 < x < RECT[0] + RECT[2] * 0.6, angle
    assert y > RECT[1] + RECT[3] * 0.55, angle  # lower part of the screen, road ahead above it
    ahead = camera.project_point([40.0, 0.0, 1.22])
    assert ahead is not None and ahead[1] < y, angle  # forward is up the screen


def test_front_rig_is_the_stock_camera():
  front = _front()
  assert RIGS[ViewAngle.FRONT].camera(RigContext(RECT, Scene(valid=True), front, 0.05)) is front
  vanishing = front.project_point([1000.0, 0.0, 0.0])
  assert abs(vanishing[0] - (RECT[0] + RECT[2] / 2)) < RECT[2] * 0.1


class _Msg(SimpleNamespace):
  pass


def _line(y, z=1.22):
  xs = np.linspace(0.0, 100.0, 33)
  return _Msg(x=list(xs), y=[y] * 33, z=[z] * 33)


def _lead(d):
  return _Msg(present=d is not None, dRel=d[0] if d else 0.0, yRel=d[1] if d else 0.0, vRel=-1.0, radar=False)


class _SM:
  """The SubMaster surface SceneTracker reads."""

  def __init__(self, lead_one=(20.0, 0.5), lead_two=(21.0, 0.4)):
    self.data = {
      "modelV2": _Msg(position=_line(0.0, 0.0), laneLines=[_line(-5.0), _line(-1.6), _line(1.6), _line(5.0)],
                      laneLineProbs=[0.5, 0.9, 0.9, 0.5], roadEdges=[_line(-6.0), _line(6.0)], roadEdgeStds=[0.3, 0.3],
                      acceleration=_Msg(x=[0.0] * 33)),
      "radarState": _Msg(leadOne=_lead(lead_one), leadTwo=_lead(lead_two)),
      "extrinsicsCalibration": _Msg(height=[1.25]),
      "carState": _Msg(vEgo=15.0), "selfdriveState": _Msg(experimentalMode=False), "longitudinalPlan": _Msg(allowThrottle=True),
    }
    names = list(self.data)
    self.updated, self.recv_frame = dict.fromkeys(names, True), dict.fromkeys(names, 1)
    self.seen, self.valid = dict.fromkeys(names, True), dict.fromkeys(names, True)

  def __getitem__(self, name):
    return self.data[name]


def test_scene_tracker_extracts_geometry_and_leads():
  scene = SceneTracker().update(_SM())
  assert scene.valid and scene.height == 1.25
  assert scene.path[0, 2] == 1.25  # path dropped from camera height onto the road
  assert scene.lane_lines[0][0, 0] == -BEHIND_EXTENSION  # lines continue behind the car
  assert len(scene.leads) == 1  # leadTwo 1 m behind leadOne is the same car
  lead = scene.leads[0]
  assert lead.center[1] == -0.5  # radar y is left-positive, calib y is right-positive
  assert lead.center[0] > RADAR_TO_CAMERA + 20.0


def test_scene_tracker_keeps_distinct_second_lead_and_caches():
  tracker = SceneTracker()
  sm = _SM(lead_two=(45.0, -3.0))
  first = tracker.update(sm)
  assert len(first.leads) == 2
  assert tracker.update(sm).revision == first.revision  # nothing changed, nothing rebuilt
  sm.recv_frame["modelV2"] = 2
  assert tracker.update(sm).revision == first.revision + 1


def test_selection_store_round_trip(tmp_path):
  path = tmp_path / "scene_views.json"
  path.write_text(json.dumps({"angle": "top", "mode": "synthetic", "futureOption": 3}))
  store = SelectionStore(path, poll_interval=0.0)
  assert store.get() == VizSelection(ViewAngle.TOP, VizMode.SYNTHETIC)
  store.set(store.get().next_angle())
  saved = json.loads(path.read_text())
  assert saved["angle"] == "front" and saved["futureOption"] == 3  # wraps around, keeps unknown keys
  assert SelectionStore(path, poll_interval=0.0).get().angle == ViewAngle.FRONT


def test_selection_store_ignores_bad_files(tmp_path):
  path = tmp_path / "scene_views.json"
  path.write_text("{not json")
  assert SelectionStore(path, poll_interval=0.0).get().stock
  path.write_text(json.dumps({"angle": "sideways", "mode": 7}))
  assert SelectionStore(path, poll_interval=0.0).get().stock


def test_cycle_orders():
  selection = VizSelection()
  assert selection.stock
  assert [selection := selection.next_mode() for _ in range(3)][-1].mode == VizMode.CAMERA
  assert VizSelection().next_angle().angle == ViewAngle.CHASE
