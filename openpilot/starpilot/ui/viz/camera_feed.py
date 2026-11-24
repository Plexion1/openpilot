"""Adapter between the scene views and the stock onroad CameraView / AugmentedRoadView.

The stock view keeps owning the VisionIPC clients, textures and EGL images; this class only borrows them. Every use of
CameraView internals is collected here so an upstream change to cameraview.py only needs this file updated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from openpilot.cereal.visionipc import VisionStreamType
from openpilot.common.transformations.camera import DEVICE_CAMERAS

NARROW = VisionStreamType.VISION_STREAM_NARROW_ROAD
WIDE = VisionStreamType.VISION_STREAM_WIDE_ROAD
DEFAULT_DEVICE_CAMERA = DEVICE_CAMERAS["tici", "ar0231"]


@dataclass(frozen=True)
class CameraGeometry:
  intrinsics: np.ndarray       # pixels of this stream's frames
  view_from_calib: np.ndarray  # rotation, calib frame -> camera view frame
  wide: bool


@dataclass(frozen=True)
class FrameTextures:
  texture: Any               # EGL external texture, or the Y plane
  uv: Any | None             # interleaved UV plane, None with EGL
  egl: bool
  uv_width: float            # texture pixels that map to u = 1 (stride for the Y plane)
  uv_height: float


class CameraFeed:
  def __init__(self, view):
    self.view = view

  @property
  def egl(self) -> bool:
    return bool(self.view._use_egl)

  def geometry(self) -> CameraGeometry:
    view = self.view
    wide = view.stream_type == WIDE
    device_camera = view.device_camera or DEFAULT_DEVICE_CAMERA
    config = device_camera.wide_road if wide else device_camera.narrow_road
    return CameraGeometry(config.intrinsics, view.view_from_wide_calib if wide else view.view_from_calib, wide)

  def update_calibration(self) -> None:
    self.view._update_calibration()

  def road_view_from_calib(self) -> np.ndarray:
    """The main road camera's calibrated rotation (StarPilot's stop sign boxes come from that camera)."""
    return self.view.view_from_calib

  def select_stream(self, sm, preferred: str | None, camera_view: int, reverse_driver: bool) -> None:
    """``preferred`` None keeps StarPilot's own choice (Auto/Standard/Wide); otherwise "wide" or "narrow"."""
    view = self.view
    if preferred is None:
      view._switch_stream_if_needed(sm, camera_view, reverse_driver)
      return
    want = WIDE if preferred == "wide" else NARROW
    if want == WIDE and view.available_streams and WIDE not in view.available_streams:
      want = NARROW
    if view.stream_type != want:
      view.switch_stream(want)

  def poll(self) -> bool:
    return bool(self.view._poll_camera())

  def textures(self) -> FrameTextures | None:
    """Bind/upload the current frame exactly like CameraView does and return what the shader samples."""
    view = self.view
    frame = view.frame
    if frame is None:
      return None
    if view._use_egl:
      return self._egl_textures(view, frame)
    if not view.texture_y or not view.texture_uv:
      return None
    if view._texture_needs_update:
      y_size = view.texture_y.width * view.texture_y.height
      uv_size = 2 * view.texture_uv.width * view.texture_uv.height
      data = frame.data
      if frame.uv_offset < y_size or len(data) < frame.uv_offset + uv_size:
        return None
      from openpilot.starpilot.ui.raylib import rl
      rl.update_texture(view.texture_y, data[:y_size])
      rl.update_texture(view.texture_uv, data[frame.uv_offset:frame.uv_offset + uv_size])
      view._texture_needs_update = False
    return FrameTextures(view.texture_y, view.texture_uv, False, float(view.texture_y.width), float(frame.height))

  @staticmethod
  def _egl_textures(view, frame) -> FrameTextures | None:
    from openpilot.system.ui.lib.egl import bind_egl_image_to_texture, create_egl_image
    if view.egl_texture is None:
      return None
    image = view.egl_images.get(frame.idx)
    if image is None:
      image = create_egl_image(frame.width, frame.height, frame.stride, frame.fd, frame.uv_offset)
      if not image:
        return None
      view.egl_images[frame.idx] = image
    view.egl_texture.width = frame.width
    view.egl_texture.height = frame.height
    bind_egl_image_to_texture(view.egl_texture.id, image)
    return FrameTextures(view.egl_texture, None, True, float(frame.width), float(frame.height))
