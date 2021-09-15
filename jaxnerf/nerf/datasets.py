# Lint as: python3
"""Different datasets implementation plus a general port for all the datasets."""
INTERNAL = False  # pylint: disable=g-statement-before-imports
# BEGIN GOOGLE-INTERNAL
INTERNAL = True
# END GOOGLE-INTERNAL
import hashlib
import json
import os
from os import path
import pickle
import queue
import threading
from typing import List, Union

from absl import logging
from cvx2 import latest as cv2  # pylint: disable=g-importing-member,g-import-not-at-top
import jax
import numpy as np
from PIL import Image

from google3.experimental.users.daeyun.jaxnerf.nerf import transforms
from google3.experimental.users.daeyun.jaxnerf.nerf import utils
from google3.pyglib.contrib.gpathlib import gpath

if not INTERNAL:
  import cv2  # pylint: disable=g-import-not-at-top
# BEGIN GOOGLE-INTERNAL
# END GOOGLE-INTERNAL


def get_dataset(split, args, use_world_coord_rays=True):
  return dataset_dict[args.dataset](
      split, args, use_world_coord_rays=use_world_coord_rays)


def convert_to_ndc(origins, directions, focal, w, h, near=1.):
  """Convert a set of rays to NDC coordinates."""
  raise NotImplementedError("Not tested.")
  # Shift ray origins to near plane
  t = -(near + origins[..., 2]) / directions[..., 2]
  origins = origins + t[..., None] * directions

  dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
  ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

  # Projection
  o0 = -((2 * focal) / w) * (ox / oz)
  o1 = -((2 * focal) / h) * (oy / oz)
  o2 = 1 + 2 * near / oz

  d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
  d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
  d2 = -2 * near / oz

  origins = np.stack([o0, o1, o2], -1)
  directions = np.stack([d0, d1, d2], -1)
  return origins, directions


class Dataset(threading.Thread):
  """Dataset Base Class."""

  def __init__(self, split, args, use_world_coord_rays):
    super(Dataset, self).__init__()
    self.queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True
    self.use_pixel_centers = args.use_pixel_centers
    self.split = split
    self.use_world_coord_rays = use_world_coord_rays
    self.batch_size = args.batch_size // jax.process_count()
    if split == "train":
      self._train_init(args)
    elif split == "test":
      self._test_init(args)
    else:
      raise ValueError(
          "the split argument should be either \"train\" or \"test\", set"
          "to {} here.".format(split))
    # For now, they are all same sizes.
    self.image_sizes = np.array(
        [(self.h, self.w) for _ in range(self.n_examples)],
        dtype=np.int32)
    self.batching = args.batching
    self.render_path = args.render_path
    self.start()

    # Some of the values not defined in __init__:
    #   camera_ids (N*H*W,) or (N,H*W) depending on `batching`.
    #   camtoworlds (N, 3, 4)
    #     Re-centered cameras.

  def __iter__(self):
    return self

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: dict, has "pixels", "rays", "camera_ids".

      For example,
      batch['pixels'] (1, 4096, 3) = (D,B,3) where D=num devices, B=batch size.
      batch['rays'].* (1, 4096, 3)
      batch['camera_ids'] (1, 4096) np.int32
      test_batch['pixels'] (224, 224, 3)
      test_batch['rays'].* (224, 224, 3)
      test_batch['camera_ids'] (224, 224)
    """
    x = self.queue.get()
    if self.split == "train":
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: dict, has "pixels" and "rays".
    """
    x = self.queue.queue[0].copy()  # Make a copy of the front of the queue.
    if self.split == "train":
      return utils.shard(x)
    else:
      return utils.to_device(x)

  # Overrides threading.Thread.run
  def run(self):
    if self.split == "train":
      next_func = self._next_train
    else:
      next_func = self._next_test
    while True:
      self.queue.put(next_func())

  @property
  def size(self):
    return self.n_examples

  def _train_init(self, args):
    """Initialize training."""
    self._load_renderings(args)
    self._generate_rays(use_world_coords=self.use_world_coord_rays)

    num_images = self.images.shape[0]
    num_rays_per_image = self.resolution
    # Flattened. e.g. [ 0,  0,  0, ..., 20, 20, 20] for 21 images.
    # (N*224*224,)
    self.camera_ids = np.repeat(
        np.arange(num_images, dtype=np.int32), num_rays_per_image)

    if self.split == "train":
      indices = self.train_indices
    else:
      indices = self.eval_indices

    ray_indices = np.arange(self.camera_ids.shape[0])
    self.available_ray_indices = ray_indices[np.isin(self.camera_ids, indices)]

    if args.batching == "all_images":
      # flatten the ray and image dimension together.
      # (N, 224, 224, 3) to (N*224*224, 3), for both images and rays.*.
      self.images = self.images.reshape([-1, 3])
      self.rays = utils.namedtuple_map(lambda r: r.reshape([-1, r.shape[-1]]),
                                       self.rays)
    elif args.batching == "single_image":
      self.images = self.images.reshape([-1, self.resolution, 3])
      self.rays = utils.namedtuple_map(
          lambda r: r.reshape([-1, self.resolution, r.shape[-1]]), self.rays)
      # camera_ids[i,:]=i. Repetitive but kept for consistency.
      self.camera_ids = self.camera_ids.reshape(-1, self.resolution)
    else:
      raise NotImplementedError(
          f"{args.batching} batching strategy is not implemented.")

  def _test_init(self, args):
    self._load_renderings(args)
    self._generate_rays(use_world_coords=self.use_world_coord_rays)
    self.it = 0

    num_images = self.images.shape[0]
    num_rays_per_image = self.resolution
    self.camera_ids = np.repeat(
        np.arange(num_images, dtype=np.int32), num_rays_per_image).reshape(
            self.rays.origins.shape[:-1])

    # Linear indices shaped into (H_i, W_i) arrays.
    ray_ids = []
    last_index = 0
    for image in self.images:
      num_pixels = np.prod(image.shape[:2])
      ray_ids.append(
          np.arange(last_index,
                    last_index + num_pixels).reshape(image.shape[:2]))
      last_index += num_pixels
    self.ray_ids = ray_ids

    # No flattening for test.

  def _next_train(self):
    """Sample next training batch."""

    if self.batching == "all_images":
      ray_indices = np.random.choice(
          self.available_ray_indices, self.batch_size, replace=False)
      batch_pixels = self.images[ray_indices]
      batch_rays = utils.namedtuple_map(lambda r: r[ray_indices], self.rays)
      batch_cameras = self.camera_ids[ray_indices]
    elif self.batching == "single_image":
      raise NotImplementedError("Single image batching mode is currently not "
                                "considered.")
      image_index = np.random.randint(0, self.n_examples, ())
      ray_indices = np.random.randint(0, self.rays[0][0].shape[0],
                                      (self.batch_size,))
      batch_pixels = self.images[image_index][ray_indices]
      batch_rays = utils.namedtuple_map(lambda r: r[image_index][ray_indices],
                                        self.rays)
      batch_cameras = self.camera_ids[image_index][ray_indices]
    else:
      raise NotImplementedError(
          f"{self.batching} batching strategy is not implemented.")
    return {
        "pixels": batch_pixels,
        "rays": batch_rays,
        "camera_ids": batch_cameras,
        # We can infer ray values and camera ids from ray ids.
        "ray_ids": ray_indices,
    }

  def _next_test(self):
    """Sample next test example."""
    idx = self.eval_indices[self.it]
    self.it = (self.it + 1) % len(self.eval_indices)

    if self.render_path:
      return {"rays": utils.namedtuple_map(lambda r: r[idx], self.render_rays)}
    else:
      return {
          # (H, W, 3), 0<=x<=1
          "pixels": self.images[idx],
          # (H, W, 3), named tuple.
          "rays": utils.namedtuple_map(lambda r: r[idx], self.rays),
          # (H, W), filled with a single camera id.
          "camera_ids": self.camera_ids[idx],
          "ray_ids": self.ray_ids[idx],
          "example_id": idx,  # Expected to be the same as camera id.
      }

  def _generate_rays(self, use_world_coords=True):
    """Generating rays for all pixels in all images."""
    # BEGIN GOOGLE-INTERNAL
    # See go/pixelcenters
    # END GOOGLE-INTERNAL
    pixel_center = 0.5 if self.use_pixel_centers else 0.0
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(self.w, dtype=np.float32) + pixel_center,  # X-Axis (columns)
        np.arange(self.h, dtype=np.float32) + pixel_center,  # Y-Axis (rows)
        indexing="xy")  # (224, 224) each
    # e.g. x[0] = [  0.5   1.5   2.5 ... 221.5 222.5 223.5]

    if use_world_coords:
      camtoworlds = self.camtoworlds
    else:
      camtoworlds = np.stack([
          np.eye(3, 4) for _ in range(self.camtoworlds.shape[0])
      ]).astype(self.camtoworlds.dtype)

    # (224, 224, 3) each.
    # dir*f is the vector from (0,0,0) to (x_pixel-w/2, -(y_pixel-h/2), -f)
    camera_dirs = np.stack([(x - self.w * 0.5) / self.focal,
                            -(y - self.h * 0.5) / self.focal, -np.ones_like(x)],
                           axis=-1)
    # (N, 224, 224, 3) = (1, 224, 224, 1, 3) * (N, 1, 1, 3, 3)
    # N=21 if train else 123 (120 generated views + 3 test set examples).
    directions = ((camera_dirs[None, ..., None, :] *
                   camtoworlds[:, None, None, :3, :3]).sum(axis=-1))

    # (N, 224, 224, 3)
    origins = np.broadcast_to(camtoworlds[:, None, None, :3, -1],
                              directions.shape)
    # (N, 224, 224, 3)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    self.rays = utils.Rays(
        origins=origins, directions=directions, viewdirs=viewdirs)


class Blender(Dataset):
  """Blender Dataset."""

  def _load_renderings(self, args):
    """Load images from disk."""
    if args.render_path:
      raise ValueError("render_path cannot be used for the blender dataset.")
    with utils.open_file(
        path.join(args.data_dir, "transforms_{}.json".format(self.split)),
        "r") as fp:
      meta = json.load(fp)
    images = []
    cams = []
    for i in range(len(meta["frames"])):
      frame = meta["frames"][i]
      fname = os.path.join(args.data_dir, frame["file_path"] + ".png")
      with utils.open_file(fname, "rb") as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        if args.factor == 2:
          [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
          image = cv2.resize(
              image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
        elif args.factor > 0:
          raise ValueError("Blender dataset only supports factor=0 or 2, {} "
                           "set.".format(args.factor))
      cams.append(np.array(frame["transform_matrix"], dtype=np.float32))
      images.append(image)
    self.images = np.stack(images, axis=0)
    if args.white_bkgd:
      self.images = (
          self.images[..., :3] * self.images[..., -1:] +
          (1. - self.images[..., -1:]))
    else:
      self.images = self.images[..., :3]
    self.h, self.w = self.images.shape[1:3]
    self.resolution = self.h * self.w
    self.camtoworlds = np.stack(cams, axis=0)
    camera_angle_x = float(meta["camera_angle_x"])
    self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
    self.n_examples = self.images.shape[0]


class LLFF(Dataset):
  """LLFF Dataset."""

  def _load_renderings(self, args):
    """Load images from disk."""
    # Load images.
    imgdir_suffix = ""
    if args.factor > 0:
      imgdir_suffix = "_{}".format(args.factor)
      factor = args.factor
    else:
      factor = 1
    imgdir = path.join(args.data_dir, "images" + imgdir_suffix)
    if not utils.file_exists(imgdir):
      raise ValueError("Image folder {} doesn't exist.".format(imgdir))

    # Sorting is important. Otherwise examples cannot be identified by indices.
    imgfiles = [
        path.join(imgdir, f)
        for f in sorted(utils.listdir(imgdir))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]
    if args.first_n_images:
      imgfiles = imgfiles[:args.first_n_images]
    images = []
    for imgfile in imgfiles:
      with utils.open_file(imgfile, "rb") as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        images.append(image)
    # For shapenet, len(images)=24, images[0].shape=(224, 224, 3)
    images = np.stack(images, axis=-1)  # (224, 224, 3, 24)
    logging.info("Loaded images: %s", str(images.shape))

    # Load poses and bds.
    with utils.open_file(path.join(args.data_dir, "poses_bounds.npy"),
                         "rb") as fp:
      poses_arr = np.load(fp)
    return self._process_renderings(args, images, poses_arr, factor)

  def _process_renderings(self, args, images, poses_arr, factor):
    """Preprocess loaded images."""
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])
    if poses.shape[-1] != images.shape[-1]:
      logging.warning("Mismatch between imgs {} and poses {}".format(
          images.shape[-1], poses.shape[-1]))

    # Update poses according to downsampling.
    poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    # Correct rotation matrix ordering and move variable dim to axis 0.
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(images, -1, 0)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale according to a default bd factor.
    scale = 1. / (bds.min() * .75)
    poses[:, :3, 3] *= scale
    bds *= scale

    # Recenter poses.
    poses = self._recenter_poses(poses)  # (24, 3, 5)

    # Generate a spiral/spherical ray path for rendering videos.
    if args.spherify:
      poses = self._generate_spherical_poses(poses, bds)
      self.spherify = True
    else:
      self.spherify = False
    if not args.spherify and self.split == "test":
      self._generate_spiral_poses(poses, bds)

    # Select the split. Train and eval can overlap.
    # TODO(daeyun): Needs refactoring for separate quantitative evaluation.
    self.train_indices = [
        int(item.strip()) for item in args.example_indices_train.split(",")
    ]
    self.eval_indices = [
        int(item.strip()) for item in args.example_indices_eval.split(",")
    ]
    for index in self.train_indices + self.eval_indices:
      if index < 0 or index >= images.shape[0]:
        raise ValueError(f"Invalid example index found: {index}")

    # images: (N, 224, 224, 3), N=24
    # poses: (N, 3, 5)

    # If you want subset, select here.
    self.images = images  # (N, 224, 224, 3)
    self.camtoworlds = poses[:, :3, :4]  # (N, 3, 4)
    self.focal = poses[0, -1, -1]  # 240.18476
    self.h, self.w = images.shape[1:3]  # 224, 224
    self.resolution = self.h * self.w  # 50176

    # For now, all images have the same focal length, unless noise is added.
    self.fx_fy_cx_cy = np.zeros(
        shape=(self.images.shape[0], 4), dtype=np.float32)
    self.fx_fy_cx_cy[:, :2] = self.focal
    self.fx_fy_cx_cy[:, 2] = self.w / 2.
    self.fx_fy_cx_cy[:, 3] = self.h / 2.

    # Original poses without noise.
    self.camtoworlds_original = self.camtoworlds.copy()
    self.fx_fy_cx_cy_original = self.fx_fy_cx_cy.copy()

    # Example indices within the entire dataset.
    dataset_indices = self.train_indices if self.split == "train" else self.eval_indices

    # Random perturbation.
    for i_cam in dataset_indices:
      # Only add noise to training examples. Training examples included as part
      # of evaluation set will still have noise.
      if i_cam in self.train_indices:
        if args.pose_noise_rot != 0 or args.pose_noise_trans != 0:
          Rt = random_camera_matrix(
              args.pose_noise_rot,
              args.pose_noise_trans,
              seed=integer_seed(["pose", i_cam]))
          self.camtoworlds[i_cam] = Rt.dot(
              np.concatenate(
                  [self.camtoworlds[i_cam],
                   np.array([0, 0, 0, 1])[None]],
                  axis=0))
        if args.cam_noise_focal != 0:
          self.fx_fy_cx_cy[i_cam, :2] += args.cam_noise_focal \
              * np.random.default_rng(
                  integer_seed(["focal", i_cam])).standard_normal(2)
          if args.tags and "focal_noise_same_aspect" in args.tags:
            self.fx_fy_cx_cy[i_cam, 1] = self.fx_fy_cx_cy[i_cam, 0]
        if args.cam_noise_center != 0:
          self.fx_fy_cx_cy[i_cam, 2:] += args.cam_noise_center \
              * np.random.default_rng(
                  integer_seed(["center", i_cam])).standard_normal(2)

    noise_degree = (np.arccos(
        np.clip((self.camtoworlds_original *
         self.camtoworlds).sum(1)[:, :3], -1, 1)) / np.pi * 180).mean()
    logging.info("Mean camera angular noise: %.2f degrees", noise_degree)
    # TODO(daeyun): This currently includes both train and test. Remove test.
    print((np.arccos(
        np.clip((self.camtoworlds_original *
         self.camtoworlds).sum(1)[:, :3], -1, 1)) / np.pi * 180))

    # n_examples=N, see above.
    if args.render_path:
      self.n_examples = self.render_poses.shape[0]
    else:
      self.n_examples = images.shape[0]

  def _generate_rays(self, use_world_coords=True):
    """Generate normalized device coordinate rays for llff."""
    if self.split == "test":
      n_render_poses = self.render_poses.shape[0]  # (120, 3, 4)
      self.camtoworlds = np.concatenate([self.render_poses, self.camtoworlds],
                                        axis=0)

    super()._generate_rays(use_world_coords=use_world_coords)

    if not self.spherify:
      # TODO(daeyun): For now, always set --spherify to true.
      raise NotImplementedError("Not tested.")
      ndc_origins, ndc_directions = convert_to_ndc(self.rays.origins,
                                                   self.rays.directions,
                                                   self.focal, self.w, self.h)
      self.rays = utils.Rays(
          origins=ndc_origins,
          directions=ndc_directions,
          viewdirs=self.rays.viewdirs)

    # Split poses from the dataset and generated poses
    if self.split == "test":
      self.camtoworlds = self.camtoworlds[n_render_poses:]
      split = [np.split(r, [n_render_poses], 0) for r in self.rays]
      split0, split1 = zip(*split)
      self.render_rays = utils.Rays(*split0)
      self.rays = utils.Rays(*split1)

  def _recenter_poses(self, poses):
    """Recenter poses according to the original NeRF code."""
    poses_ = poses.copy()
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = self._poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses

  def _poses_avg(self, poses):
    """Average poses according to the original NeRF code."""
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = self._normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([self._viewmatrix(vec2, up, center), hwf], 1)
    return c2w

  def _viewmatrix(self, z, up, pos):
    """Construct lookat view matrix."""
    vec2 = self._normalize(z)
    vec1_avg = up
    vec0 = self._normalize(np.cross(vec1_avg, vec2))
    vec1 = self._normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

  def _normalize(self, x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

  def _generate_spiral_poses(self, poses, bds):
    """Generate a spiral path for rendering."""
    c2w = self._poses_avg(poses)
    # Get average pose.
    up = self._normalize(poses[:, :3, 1].sum(0))
    # Find a reasonable "focus depth" for this dataset.
    close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz
    # Get radii for spiral path.
    tt = poses[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    n_views = 120
    n_rots = 2
    # Generate poses for spiral path.
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w_path[:, 4:5]
    zrate = .5
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_views + 1)[:-1]:
      c = np.dot(c2w[:3, :4], (np.array(
          [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads))
      z = self._normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
      render_poses.append(np.concatenate([self._viewmatrix(z, up, c), hwf], 1))
    self.render_poses = np.array(render_poses).astype(np.float32)[:, :3, :4]

  def _generate_spherical_poses(self, poses, bds):
    """Generate a 360 degree spherical path for rendering."""
    # pylint: disable=g-long-lambda
    p34_to_44 = lambda p: np.concatenate([
        p,
        np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
    ], 1)
    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
      a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
      b_i = -a_i @ rays_o
      pt_mindist = np.squeeze(-np.linalg.inv(
          (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
      return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)
    vec0 = self._normalize(up)
    vec1 = self._normalize(np.cross([.1, .2, .3], vec0))
    vec2 = self._normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)
    poses_reset = (
        np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4]))
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc
    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):
      camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
      up = np.array([0, 0, -1.])
      vec2 = self._normalize(camorigin)
      vec0 = self._normalize(np.cross(vec2, up))
      vec1 = self._normalize(np.cross(vec2, vec0))
      pos = camorigin
      p = np.stack([vec0, vec1, vec2, pos], 1)
      new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    new_poses = np.concatenate([
        new_poses,
        np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)
    ], -1)
    poses_reset = np.concatenate([
        poses_reset[:, :3, :4],
        np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
    ], -1)
    if self.split == "test":
      self.render_poses = new_poses[:, :3, :4]
    return poses_reset


class ShapeNetSegmented(LLFF):
  """ShapeNet images with white background."""

  def _load_renderings(self, args):
    """Load images from a .pkl file, locally or remotely."""
    # Load images.
    if args.factor > 0:
      factor = args.factor
    else:
      factor = 1
    pkl_filename = args.data_dir
    file = gpath.GPath(pkl_filename)
    if not file.is_file():
      raise ValueError("Image file {} doesn't exist.".format(pkl_filename))

    data_dict = pickle.loads(file.read_bytes())
    images = data_dict["rgb"].astype(np.float32) / 255.
    segmentations = data_dict["seg"].astype(np.float32) / 255.
    white_bg = np.ones_like(images)
    # The segmentation images are not completely binary. Fade edges smoothly.
    images = images*segmentations + white_bg*(1-segmentations)
    # (224, 224, 3, 24)
    images = images.transpose(1, 2, 3, 0).copy()

    # Camera poses.
    poses_arr = data_dict["pose"]
    return self._process_renderings(args, images, poses_arr, factor)


def integer_seed(objs: List[Union[str, int]]) -> int:
  """Reproducibly and readably generate a seed value, from strings and integers.
  """
  assert isinstance(objs, list), isinstance(objs, tuple)
  h = hashlib.sha256()
  for obj in objs:
    if isinstance(obj, str):
      h.update(obj.encode('utf-8'))
    else:
      assert isinstance(obj, int)
      h.update(obj.to_bytes(16, 'little', signed=False))
  return int(h.hexdigest(), 16) % 2**64


def random_camera_matrix(std_rotation, std_translation, seed=None):
  return random_camera_matrix_v1(std_rotation, std_translation, seed=None)


def random_camera_matrix_v1(std_rotation, std_translation, seed=None):
  """Same parametrization as learned poses."""
  delta = np.random.default_rng(seed).standard_normal(6)
  delta[:3] *= std_rotation
  delta[3:] *= std_translation
  mat34 = transforms.se3_exp_matrix(delta)[:3]
  return np.array(mat34)


def random_camera_matrix_v0(std_rotation, std_translation, seed=None):
  """Original implementation.

  Alternative way to generate random cameras.
  """
  delta = np.random.default_rng(seed).standard_normal(6)
  delta[:3] *= std_rotation
  delta[3:] *= std_translation
  mat34 = np.array([
      (1, -delta[2], delta[1], delta[3]),
      (delta[2], 1, -delta[0], delta[4]),
      (-delta[1], delta[0], 1, delta[5]),
  ],
                   dtype=delta.dtype)
  # Orthonormal.
  Q = np.linalg.qr(mat34[:, :3])[0]
  # Force positive diagonals.
  Q *= np.sign(np.diag(Q))
  mat34[:, :3] = Q
  return mat34


dataset_dict = {
    "blender": Blender,
    "llff": LLFF,
    "shapenetseg": ShapeNetSegmented,
}
