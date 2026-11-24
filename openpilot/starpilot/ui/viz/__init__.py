"""Scene views: alternate onroad visualizations, any camera angle combined with any visualization mode.

Angles (rigs.py)
  Front  the road camera's own view, framed exactly like the stock onroad view
  Chase  above and behind the car, locked to its heading; eases back a little with speed
  Top    straight down, car near the bottom; zooms out with speed so the lead stays in view

Modes (renderer.py MODE_STYLES, shaders.py)
  Camera     live video. Front + Camera is the unmodified StarPilot view; Chase and Top reproject the main road camera
             onto the road plane (inverse perspective mapping), so only the road surface is correct and anything
             standing on it smears away from the camera. The wide camera is never used (interior, blur, reflections).
  Synthetic  no video: road, lane lines (dashed between lanes, scrolling at road speed), road edges, the planned path,
             vehicles and lights drawn from the driving model, radar state and car state
  Hybrid     experimental: the synthetic road plus lane paint picked out of the camera image

What the scene shows
  * our car and the lead vehicles as a car, pickup truck or sphere in the chosen colours (settings.py), with shadows
  * brake lights (ours from the pedal, regen or firm deceleration; leads inferred from deceleration or being stopped),
    turn signals blinking with the car's, reversing lights, and at night headlights with beams on the road and
    taillight glow (darkness from the road camera exposure, like the UI's brightness)
  * the lane being changed into, tinted blue (red while that blind spot is occupied)
  * a glowing red stop line where the model wants to stop (modelV2.action.shouldStop at the predicted stopping point)
  * a stop sign beside the road while StarPilot shows its STOP icon: the same confirmed sighting and distance StarPilot
    already computes (runtime_snapshot hands it over with scene.note_stop_sign each frame), placed once and then kept
    on the road from our own travel; nothing is detected or matched again. In the top view it lies flat like a map
    marker
  * the path in StarPilot's path style (acceleration, solid colour or rainbow) and custom lane line colours
  * our engagement status as a soft glow under the car

How it hooks into StarPilot (a few lines in existing files)
  * AugmentedRoadView.render_camera_model_layer calls controller.render_layer() first; when a scene view is selected it
    draws instead of the stock camera + model layer, so every StarPilot widget, alert and border still draws on top.
    The same layer is used by StarPilot's on-device replay, which therefore shows the selected view too.
  * Two onroad widgets in StarPilot's layout system (widgets.py), placed, hidden, recoloured and layered from Galaxy's
    layout editor: the VIEW / MODE buttons (bottom right by default) and turn signal arrows beside the speed. StarPort's
    car-screen layout has them too: the arrows draw there; the car screen keeps the stock camera view, so its VIEW /
    MODE buttons stay hidden. Taps reach the buttons only when no StarPilot control claimed the touch (StarShellPage,
    just before a tap would toggle the sidebar).
  * Settings rows (settings.py): angle, mode and the look of our car and the lead vehicles, on the device's Onroad HUD
    page, Galaxy's settings and the layout editor ("Scene views").
  * StarShellSession creates the shared instance at start (controller.install), so the shaders compile while parked.
  * Choices are saved in /data/starpilot/scene_views.json (options.py), e.g.
      {"angle": "chase", "mode": "synthetic", "egoModel": "car", "egoColor": "Ocean Blue", "leadModel": "truck"}
    Params keys are compiled into the prebuilt branch, so a file is used instead; the UI picks changes up within two
    seconds.
  * Nothing native changed, so nothing is rebuilt. StarPort's projection_layout.py is outside the UI roots that
    StarPilot's Fast Update may replace, so an update carrying it installs as a full validated update.

Cost
  The driving model runs on the same GPU, so the GPU budget is what matters most.
  GPU: the scene is drawn at half resolution into an offscreen image during StarPilot's render-prepare step (one
  ground pass with cheap per-pixel work away from the road, and a ray-marched quad per vehicle sized to it), then
  stretched over the camera layer. A guard watches modelV2.frameDropPerc and lowers the resolution, then falls back to
  the stock view for a minute, if the model starts losing frames while a scene view is on (controller.GUARD_*).
  CPU: one packed uniform upload and one draw call for the ground, one of each for all vehicles; road geometry is
  resampled and uploaded once per model frame. Reported under the UI's "camera" and "model" frame phases like the stock
  layer. Shaders are compiled while the car is off (once a scene view has been used), so switching never stalls.

Safety
  Driving functions are untouched; this only changes what the camera layer paints. Any exception while drawing a scene
  view is logged once and the stock camera view takes over for the rest of the session. A shader the GPU cannot
  compile falls back to the synthetic road (or hides the vehicles). StarPort's car display keeps the stock view.

Extending
  New angle: subclass rigs.Rig (return a geometry.VirtualCamera), add a ViewAngle and its label in options.py, register
  it in rigs.RIGS. New mode: add a VizMode and label in options.py and a ModeStyle in renderer.MODE_STYLES; a new
  ground treatment is a new shader_mode branch in shaders.py. New vehicle model: a signed distance function and its
  lamps in shaders.VEHICLE_SHADER, a VehicleModel in options.py and its size in renderer.VEHICLE_MODELS. Scene data
  comes from scene.SceneTracker, which reads a SubMaster-like object, so replays, previews and tests can drive it
  without the live services.

Module map
  geometry.py     pure numpy camera math (VirtualCamera, look-at, pinhole)
  scene.py        model / radar / car state / calibration -> Scene snapshot (geometry cached per message)
  rigs.py         camera angles
  shaders.py      GLSL: ground pass (road, markings, path, lights, shadows, camera reprojection) and vehicles
  renderer.py     draws a frame: ground pass, stop line glow, vehicles
  camera_feed.py  the only place that touches CameraView internals (frames, streams, calibration)
  controls.py     the VIEW / MODE buttons
  widgets.py      the onroad widgets' registry entries and drawing
  settings.py     settings rows (device and Galaxy)
  options.py      choices and their JSON file
  controller.py   entry points used by the hooks; owns the shared instance and the safety fallback
  tests/          unit tests for the math, rigs, scene extraction, settings, widget registration (no GPU needed)
"""
