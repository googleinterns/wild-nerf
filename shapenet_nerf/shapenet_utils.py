"""Helper functions for NeRF experiments using ShapeNet.
"""

import pickle
from typing import Any, Dict

from absl import logging
import numpy as np
import tensorflow.google as tf

from google3.pyglib.contrib.gpathlib import gpath
from google3.research.vision.scene_understanding import isun
from google3.research.vision.scene_understanding.data.tfds import example_parser
from google3.research.vision.scene_understanding.datasets.shapenet import shapenet_specs


def tf_dataset(split: str) -> tf.data.Dataset:
  """Instantiate SSTable tf.data.Dataset."""

  if split not in ("train", "test", "val", "toy"):
    raise ValueError(f"Unrecognized split {split}")

  frame_spec = shapenet_specs.frame_feature_spec(
      shapenet_specs.CAMERA_NAMES_3DR2N2)
  sstables_path = ("/namespace/scene_understanding/datasets/shapenet_core_v1/"
                   f"frame_sstables_3dr2n2/{split}*.sst")
  frame_shard_file_name_ds = tf.data.Dataset.list_files(
      sstables_path, shuffle=False)
  dataset = frame_shard_file_name_ds.interleave(
      lambda x: tf.data.SSTableDataset(x),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(lambda k, v: v)
  dataset = dataset.map(
      lambda x: example_parser.decode_serialized_example(x, frame_spec))
  return dataset


def parse_frame_data(frame_data: Dict[str, Any]) -> np.ndarray:
  """Extract and convert relevant data.

  Args:
    frame_data:
      An example from ShapeNet tf.data.Dataset instance.

  Returns:
    A tuple (rgb, seg, pose) where,
      rgb is a uint8 array of shape (24, 224, 224, 3).
      seg is a uint8 array of shape (24, 224, 224, 1).
        Object pixels have value 255.
      pose is a float64 array of shape (24, 17).

  Raises:
    ValueError: Sanity checks to make sure assumptions hold for all examples.
  """
  poses_bounds = []
  images_rgb = []
  images_seg = []

  for camera_name in shapenet_specs.CAMERA_NAMES_3DR2N2:
    if not (np.allclose(frame_data["pose"]["R"].numpy(), np.eye(3)) and
            np.allclose(frame_data["pose"]["t"].numpy(), 0)):
      raise ValueError("Expected object pose to be identity.")

    camera = frame_data["cameras"][camera_name]

    cam_to_world = isun.geometry.Isometry(
        **camera["extrinsics"]).inverse().matrix3x4().numpy().astype(np.float64)
    camera_center = cam_to_world[:, 3]

    # See LLFF spec.
    x = cam_to_world[:, 0]
    y = -cam_to_world[:, 1]
    z = -cam_to_world[:, 2]
    llff_c2w = np.stack([-y, x, z, camera_center], axis=1)

    intrinsics = camera["intrinsics"]

    image_hw = [
        int(intrinsics["image_height"]),
        int(intrinsics["image_width"]),
    ]
    focal = float(intrinsics["K"][0, 0])

    if not np.isclose(intrinsics["K"][0, 0], intrinsics["K"][1, 1]):
      raise ValueError("Expected x and y focal lengths to be the same.")

    if intrinsics["K"][0, 0] <= 0:
      raise ValueError("Expected focal length to be positive.")

    if not (np.allclose(intrinsics["distortion"]["radial"].numpy(), 0) and
            np.allclose(intrinsics["distortion"]["tangential"].numpy(), 0)):
      raise ValueError("Expected no lens distortion.")

    # ShapeNetCore v1 spec says, "each model is diagonally normalized to fit
    # within a unit cube centered at the origin". Cameras are at least 1 unit
    # distance away from the origin, so those values are safe.
    near = 0.2
    far = 5

    if not np.isclose(np.linalg.norm(frame_data["objects"]["size"].numpy()), 1):
      raise ValueError("Expected object size to be diagonally normalized to 1.")

    col = np.array([image_hw[0], image_hw[1], focal]).reshape(-1, 1)
    mat35: np.ndarray = np.hstack([llff_c2w, col])

    assert mat35.data.c_contiguous
    row = np.concatenate([mat35.ravel(), [near, far]])
    poses_bounds.append(row)

    images_rgb.append(camera["color_image"].numpy())
    images_seg.append(camera["instance_image"].numpy())

  images_rgb = np.stack(images_rgb)
  images_seg = np.stack(images_seg)
  poses_bounds = np.stack(poses_bounds)

  return images_rgb, images_seg, poses_bounds


def upload_shapenet_pkl_data(split: str, first_n: int, cell: str):
  """Upload processed ShapeNet data to CNS.

  Args:
    split: Split name, e.g. train, test, val
    first_n: How many examples to upload, from the beginning.
    cell: Name of the CNS cell. e.g. "lu-d"
  """
  dataset = tf_dataset(split)
  for i, frame_data in enumerate(dataset.take(first_n)):
    rgb, seg, pose = parse_frame_data(frame_data)
    data = {"rgb": rgb, "seg": seg, "pose": pose}
    data_bytes = pickle.dumps(data)

    filename = f"/cns/{cell}/home/daeyun/shapenet/v5/{split}/{i:05d}.pkl"
    logging.info("%d, %s", i, filename)

    gp = gpath.GPath(filename)
    gp.parent.mkdir(parents=True, exist_ok=True)
    gp.write_bytes(data_bytes)
