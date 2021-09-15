"""Uncategorized helper functions related to the Objectron dataset.
"""

import re
import struct

import numpy as np

from google3.experimental.users.daeyun.objectron_nerf import camera_utils
from google3.pyglib.contrib.gpathlib import gpath
from google3.third_party.mediapipe.modules.objectron.calculators import a_r_capture_metadata_pb2 as ar_metadata_protocol


def find_sequence_name_from_any_string(source: str) -> str:
  """A convenience function that extract a "video id" substring.

  Args:
    source: e.g. "video_frames/images_8/chair/batch-24/0/00000.png"

  Returns:
    e.g. "chair/batch-24/0"
    None if no match.

  """
  out = re.search(r'(?:^|/)([a-zA-Z][^/]+/batch-\d+/\d+)(?:[^\d]|$)', source)
  if out is None:
    return None
  return out.group(1)


def arcamera_to_camera(camera):
  R = np.array(camera.view_matrix).reshape(4, 4)[:3, :3]
  t = np.array(camera.view_matrix).reshape(4, 4)[:3, -1, None]
  K = np.array(camera.intrinsics).reshape(3, 3)
  cam = camera_utils.PerspectiveCamera(R, t, s=1.0, K=K)
  return cam


def make_poses_bounds_array(pbdata_path, near=0.01, far=5.0):
  # e.g. /cns/lu-d/home/xeno/pursuit/datasets/videonet3d/sfm_arframe/chair/batch-24/33/sfm_arframe.pbdata
  # NOTE: sfm_arframe.pbdata has higher quality cameras, but some frames are dropped.
  frame_data = get_frame_data(pbdata_path)
  rows = []
  for arcam in [item.camera for item in frame_data]:
    cam_to_world = np.array(arcam.transform).reshape(4, 4)[:3]
    w = arcam.image_resolution_width
    h = arcam.image_resolution_height
    assert w >= h, (w, h)  # assume landscape
    row = arcamera_to_llff(
        cam_to_world,
        image_hw=[w, h],
        focal=arcam.intrinsics[0],
        near=near,
        far=far)
    rows.append(row)
  return np.vstack(rows)


def llff_pose(cam_to_world_mat34, image_hw, focal, near, far):
  # Returns an array of shape (17,).
  col = np.concatenate([image_hw, [focal]]).reshape(-1, 1)
  mat35 = np.hstack([cam_to_world_mat34, col])
  return np.concatenate([mat35.ravel(), [near, far]])


def point_cloud_from_frame_data(frames):
  points = []
  for frame in frames:
    current_points = [
        np.array([v.x, v.y, v.z]) for v in frame.raw_feature_points.point
    ]
    points.extend(current_points)
  points = np.array(points)
  return points


# Code copied from notebooks in Objectron's git repo (below):


def get_frame_data(remote_geometry_filename):
  ret = []
  geometry_file_path = gpath.GPath(remote_geometry_filename)
  proto_buf = geometry_file_path.read_bytes()

  i = 0
  frame_number = 0

  while i < len(proto_buf):
    # Read the first four Bytes in little endian '<' integers 'I' format
    # indicating the length of the current message.
    msg_len = struct.unpack('<I', proto_buf[i:i + 4])[0]
    i += 4
    message_buf = proto_buf[i:i + msg_len]
    i += msg_len
    frame_data = ar_metadata_protocol.ARFrame()
    frame_data.ParseFromString(message_buf)
    ret.append(frame_data)
  return ret


def get_geometry_data(remote_geometry_filename):
  sequence_geometry = []
  geometry_file_path = gpath.GPath(remote_geometry_filename)
  proto_buf = geometry_file_path.read_bytes()

  i = 0
  frame_number = 0

  while i < len(proto_buf):
    # Read the first four Bytes in little endian '<' integers 'I' format
    # indicating the length of the current message.
    msg_len = struct.unpack('<I', proto_buf[i:i + 4])[0]
    i += 4
    message_buf = proto_buf[i:i + msg_len]
    i += msg_len
    frame_data = ar_metadata_protocol.ARFrame()
    frame_data.ParseFromString(message_buf)

    transform = np.reshape(frame_data.camera.transform, (4, 4))
    projection = np.reshape(frame_data.camera.projection_matrix, (4, 4))
    view = np.reshape(frame_data.camera.view_matrix, (4, 4))
    position = transform[:3, -1]

    current_points = [
        np.array([v.x, v.y, v.z]) for v in frame_data.raw_feature_points.point
    ]
    current_points = np.array(current_points)

    sequence_geometry.append((transform, projection, view, current_points))
  return sequence_geometry
