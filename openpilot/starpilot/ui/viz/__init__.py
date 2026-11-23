"""Scene views: alternate onroad visualizations, any camera angle combined with any visualization mode.

Angles (rigs.py)
  Front  the road camera's own view, framed exactly like the stock onroad view
  Chase  above and behind the car, locked to its heading
  Top    straight down, car near the bottom; zooms out with speed so the lead stays in view

Modes (renderer.py MODE_STYLES, shaders.py)
  Camera     live video. Front + Camera is the unmodified StarPilot view; Chase and Top reproject the road camera onto
             the road plane (inverse perspective mapping), so only the road surface is correct and anything standing
             on it (cars, poles) smears away from the camera
  Synthetic  no video at all: sky, road and grid drawn from the driving model output, lane lines, road edges, the path
             and spheres for the car and the lead vehicles
  Hybrid     experimental: the synthetic road plus lane paint picked out of the camera image (bright-and-white or
             yellow pixels that are brighter than the road on both sides, within the model's road edges)

How it hooks into StarPilot (8 lines in existing files)
  * AugmentedRoadView.render_camera_model_layer calls controller.render_layer() first; when a scene view is selected it
    draws instead of the stock camera + model layer, so every StarPilot widget, alert and border still draws on top.
    The same layer is used by StarPilot's on-device replay, which therefore shows the selected view too.
  * The VIEW / MODE buttons (controls.py) sit in the bottom-right corner of the camera layer. Taps reach them only
    when no StarPilot control claimed the touch (StarShellPage, just before a tap would toggle the sidebar).
  * The selection is saved in /data/starpilot/scene_views.json (options.py), e.g.
      {"angle": "chase", "mode": "synthetic", "controls": true}
    Params keys are compiled into the prebuilt branch, so a file is used instead; anything may write it and the UI
    picks changes up within two seconds. "controls": false hides the buttons.
  * Only .py files under the UI runtime roots changed, so the prebuilt receipt stays valid and nothing is rebuilt.

Safety
  Driving functions are untouched; this only changes what the camera layer paints. Any exception while drawing a scene
  view is logged once and the stock camera view takes over for the rest of the session. A shader the GPU cannot
  compile falls back to the synthetic road. StarPort's car display keeps the stock view.

Extending
  New angle: subclass rigs.Rig (return a geometry.VirtualCamera), add a ViewAngle and its label in options.py, register
  it in rigs.RIGS. New mode: add a VizMode and label in options.py and a ModeStyle in renderer.MODE_STYLES; a new
  ground treatment is a new shader_mode branch in shaders.py. Scene data comes from scene.SceneTracker, which reads a
  SubMaster-like object, so replays, previews and tests can drive it without the live services.

Module map
  geometry.py     pure numpy projection math (VirtualCamera, ribbons, near-plane clipping)
  scene.py        model / radar / calibration -> Scene snapshot (cached per message)
  rigs.py         camera angles
  shaders.py      GLSL: ground/sky/camera reprojection pass and the sphere impostor
  renderer.py     draws a frame: ground pass, road edges, lanes, path, shadows, spheres
  camera_feed.py  the only place that touches CameraView internals (frames, streams, calibration)
  controls.py     on-screen switcher
  options.py      selection model and its JSON file
  controller.py   entry points used by the hooks; owns the shared instance and the safety fallback
  tests/          unit tests for the math, rigs, scene extraction and storage (no GPU needed)
"""
