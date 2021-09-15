# Lint as: python3
"""Different model implementation plus a general port for all the models."""
from typing import Any, Callable, List

from absl import logging
from absl.logging import DEBUG, INFO
import chex
from flax import linen as nn
from jax import random
import jax.numpy as jnp

from google3.experimental.users.daeyun.jaxnerf.nerf import model_utils
from google3.experimental.users.daeyun.jaxnerf.nerf import transforms
from google3.experimental.users.daeyun.jaxnerf.nerf import utils
from google3.experimental.users.daeyun.jaxnerf.nerf import utils_extra


def get_model(key, example_batch, args, dataset):
  """A helper function that wraps around a 'model zoo'."""
  model_dict = {
      "nerf": construct_nerf,
  }
  return model_dict[args.model](
      key, example_batch, args, dataset)


if 'global_state' not in globals():
  global_state = {}

class NerfModel(nn.Module):
  """Nerf NN Model with both coarse and fine MLPs."""
  num_coarse_samples: int  # The number of samples for the coarse nerf.
  num_fine_samples: int  # The number of samples for the fine nerf.
  use_viewdirs: bool  # If True, use viewdirs as an input.
  near: float  # The distance to the near plane
  far: float  # The distance to the far plane
  noise_std: float  # The std dev of noise added to raw sigma.
  net_depth: int  # The depth of the first part of MLP.
  net_width: int  # The width of the first part of MLP.
  net_depth_condition: int  # The depth of the second part of MLP.
  net_width_condition: int  # The width of the second part of MLP.
  net_activation: Callable[..., Any]  # MLP activation
  skip_layer: int  # How often to add skip connections.
  num_rgb_channels: int  # The number of RGB channels.
  num_sigma_channels: int  # The number of density channels.
  white_bkgd: bool  # If True, use a white background.
  min_deg_point: int  # The minimum degree of positional encoding for positions.
  max_deg_point: int  # The maximum degree of positional encoding for positions.
  deg_view: int  # The degree of positional encoding for viewdirs.
  lindisp: bool  # If True, sample linearly in disparity rather than in depth.
  rgb_activation: Callable[..., Any]  # Output RGB activation.
  sigma_activation: Callable[..., Any]  # Output sigma activation.
  legacy_posenc_order: bool  # Keep the same ordering as the original tf code.
  tags: List[str]  # Tags describing which experiments we want to do.
  num_cameras: int  # Number of camera poses to keep track of.
  camtoworlds: jnp.ndarray  # Camera matrices. (N, 4, 4)
  image_sizes: jnp.ndarray  # Image sizes. (N, 2)
  fx_fy_cx_cy: jnp.ndarray  # fx, fy focal lengths. (N, 2)

  @nn.compact
  def __call__(self, rng_0, rng_1, rays, randomized, barf_alpha, camera_ids,
               ray_ids, return_extra=False):
    """Nerf Model.

    Args:
      rng_0: jnp.ndarray, random number generator for coarse model sampling.
      rng_1: jnp.ndarray, random number generator for fine model sampling.
      rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
      randomized: bool, use randomized stratified sampling.
      barf_alpha: Positional encoding weight for bundle-adjusting NeRF (BARF)
      camera_ids: jnp.ndarray, which camera each ray came from.
      ray_ids: jnp.ndarray, unique indices of the rays in the dataset.

    Returns:
      ret: list, [(rgb_coarse, disp_coarse, acc_coarse), (rgb, disp, acc)]
    """
    # Not logged in the jitted version.
    logging.log_first_n(INFO, "NeRF forward pass started.", 10)
    is_jit_tracing = isinstance(barf_alpha, float)
    if is_jit_tracing:
      logging.log(INFO, "Model tags: %s", str(self.tags))
      logging.log(INFO, "Number of rays: %d", rays[0].shape[0])
      logging.log(INFO, "camera_ids: %s", str(camera_ids.shape))
      logging.log(INFO, "ray_ids: %s", str(ray_ids.shape))
      logging.log(INFO, "num_cameras: %d", self.num_cameras)
      logging.log(INFO, "barf_alpha: %s", str(barf_alpha))

      chex.assert_rank(barf_alpha, 0)  # alpha is scalar.
      chex.assert_rank(rays[0], 2)
      chex.assert_rank(camera_ids, 1)
      chex.assert_rank(ray_ids, 1)
      chex.assert_shape(self.camtoworlds, (self.num_cameras, 4, 4))
      chex.assert_scalar_in(barf_alpha, 0, 1)
      chex.assert_equal_shape_prefix(
          [rays.origins, rays.directions, rays.viewdirs, camera_ids, ray_ids], 1)
      chex.assert_equal_shape_prefix([ self.image_sizes, self.fx_fy_cx_cy, self.camtoworlds ], 1)

    # 6-dof rotation and translation per image.
    delta_pose = self.param(
        "delta_pose",
        lambda rng, shape: jnp.zeros(shape),  # jnp.float32 by default.
        (self.num_cameras, 6),
    )
    if is_jit_tracing:
      logging.log(INFO, "Delta 6-dof pose first row: %s", str(delta_pose[0]))

    # fx, fx, cx, cy
    # Not used in all experiments, but all variables are still created here.
    delta_intrinsics = self.param(
        "delta_intrinsics",
        lambda rng, shape: jnp.zeros(shape),
        (self.num_cameras, 4),
    )

    if is_jit_tracing:
      logging.log(INFO, "Delta intrinsics first row: %s",
                  str(delta_intrinsics[0]))

    fx = self.fx_fy_cx_cy[:, 0]
    fy = self.fx_fy_cx_cy[:, 1]
    cx = self.fx_fy_cx_cy[:, 2]
    cy = self.fx_fy_cx_cy[:, 3]

    if "learn_delta_focal" in self.tags:
      if "focal_optim_same_aspect" in self.tags:
        fx += delta_intrinsics[:, 0]
        fy += delta_intrinsics[:, 0]
      else:
        fx += delta_intrinsics[:, 0]
        fy += delta_intrinsics[:, 1]

    if "learn_delta_center" in self.tags:
      cx += delta_intrinsics[:, 2]
      cy += delta_intrinsics[:, 3]

    ret_extra = {}
    ret_extra["cx"] = cx
    ret_extra["cy"] = cy
    ret_extra["fx"] = fx
    ret_extra["fy"] = fy
    ret_extra["delta_pose"] = delta_pose
    ret_extra["ray_ids"] = ray_ids
    ret_extra["rays_precomputed"] = rays
    ret_extra["camera_ids_precomputed"] = camera_ids

    # Differentiable. Comment it out to force it to use the precomputed rays
    # instead, for debugging.
    rays, camera_ids = model_utils.generate_rays(
        flat_indices=ray_ids,
        camtoworlds=self.camtoworlds,  # Constant initial params.
        image_hw=self.image_sizes,
        cx=cx, cy=cy, fx=fx, fy=fy)

    # Capture transformed rays as metadata.
    ret_extra["rays"] = rays
    ret_extra["camera_ids"] = camera_ids

    # Initially, with `delta_intrinsics` set to zero, the following conditions
    # will be true, vs. precomputed.
    # TODO(daeyun): Turn it into a unit test.
    # chex.assert_trees_all_close(rays_, rays, rtol=1e-5, atol=1e-5)
    # chex.assert_trees_all_close(camera_ids_, camera_ids, rtol=1e-5, atol=1e-5)

    # Apply estimated camera rotation and translation.
    if "learn_delta_extrinsics" in self.tags:
      delta_mat44 = jnp.stack(
          [transforms.se3_exp_matrix(row) for row in delta_pose])
      delta_mat44 = delta_mat44[camera_ids]
      rays = utils.namedtuple_map(
          lambda r: ((delta_mat44[:, :3, :3] @ r[:, :, None] +
                      delta_mat44[:, :3, 3, None])).reshape(-1, 3), rays)
      logging.log_first_n(INFO, "Applied estimated pose.", 10)

    # Stratified sampling along rays
    key, rng_0 = random.split(rng_0)
    # `samples` is origin + z_val*direction
    z_vals, samples = model_utils.sample_along_rays(
        key,
        rays.origins,
        rays.directions,
        self.num_coarse_samples,
        self.near,
        self.far,
        randomized,
        self.lindisp,
    )
    samples_enc = model_utils.posenc(
        samples,
        self.min_deg_point,
        self.max_deg_point,
        self.legacy_posenc_order,
        alpha=barf_alpha,
    )

    # Construct the "coarse" MLP.
    coarse_mlp = model_utils.MLP(
        net_depth=self.net_depth,
        net_width=self.net_width,
        net_depth_condition=self.net_depth_condition,
        net_width_condition=self.net_width_condition,
        net_activation=self.net_activation,
        skip_layer=self.skip_layer,
        num_rgb_channels=self.num_rgb_channels,
        num_sigma_channels=self.num_sigma_channels)

    # Point attribute predictions
    if self.use_viewdirs:
      viewdirs_enc = model_utils.posenc(
          rays.viewdirs,
          0,
          self.deg_view,
          self.legacy_posenc_order,
          alpha=barf_alpha,
      )
      raw_rgb, raw_sigma = coarse_mlp(samples_enc, viewdirs_enc)
    else:
      raw_rgb, raw_sigma = coarse_mlp(samples_enc)
    # Add noises to regularize the density predictions if needed
    key, rng_0 = random.split(rng_0)
    raw_sigma = model_utils.add_gaussian_noise(
        key,
        raw_sigma,
        self.noise_std,
        randomized,
    )
    rgb = self.rgb_activation(raw_rgb)
    sigma = self.sigma_activation(raw_sigma)
    # Volumetric rendering.
    comp_rgb, disp, acc, weights = model_utils.volumetric_rendering(
        rgb,
        sigma,
        z_vals,
        rays.directions,
        white_bkgd=self.white_bkgd,
    )
    ret = [
        (comp_rgb, disp, acc),
    ]
    # Hierarchical sampling based on coarse predictions
    if self.num_fine_samples > 0:
      z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
      key, rng_1 = random.split(rng_1)
      # `samples` is origin + z_val*direction
      z_vals, samples = model_utils.sample_pdf(
          key,
          z_vals_mid,
          weights[..., 1:-1],
          rays.origins,
          rays.directions,
          z_vals,
          self.num_fine_samples,
          randomized,
      )
      samples_enc = model_utils.posenc(
          samples,
          self.min_deg_point,
          self.max_deg_point,
          self.legacy_posenc_order,
          alpha=barf_alpha,
      )

      # Construct the "fine" MLP.
      fine_mlp = model_utils.MLP(
          net_depth=self.net_depth,
          net_width=self.net_width,
          net_depth_condition=self.net_depth_condition,
          net_width_condition=self.net_width_condition,
          net_activation=self.net_activation,
          skip_layer=self.skip_layer,
          num_rgb_channels=self.num_rgb_channels,
          num_sigma_channels=self.num_sigma_channels)

      if self.use_viewdirs:
        raw_rgb, raw_sigma = fine_mlp(samples_enc, viewdirs_enc)
      else:
        raw_rgb, raw_sigma = fine_mlp(samples_enc)
      key, rng_1 = random.split(rng_1)
      raw_sigma = model_utils.add_gaussian_noise(
          key,
          raw_sigma,
          self.noise_std,
          randomized,
      )
      rgb = self.rgb_activation(raw_rgb)
      sigma = self.sigma_activation(raw_sigma)
      comp_rgb, disp, acc, unused_weights = model_utils.volumetric_rendering(
          rgb,
          sigma,
          z_vals,
          rays.directions,
          white_bkgd=self.white_bkgd,
      )
      ret.append((comp_rgb, disp, acc))
    logging.log_first_n(INFO, "End of nerf forward pass", 10)
    if return_extra:
      return ret, ret_extra
    return ret


def construct_nerf(key, example_batch, args, dataset):
  """Construct a Neural Radiance Field.

  Args:
    key: jnp.ndarray. Random number generator.
    example_batch: dict, an example of a batch of data.
    args: FLAGS class. Hyperparameters of nerf.
    dataset: The dataset object example_batch came from.

  Returns:
    model: nn.Model. Nerf model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
  """
  net_activation = getattr(nn, str(args.net_activation))
  rgb_activation = getattr(nn, str(args.rgb_activation))
  sigma_activation = getattr(nn, str(args.sigma_activation))

  # Assert that rgb_activation always produces outputs in [0, 1], and
  # sigma_activation always produce non-negative outputs.
  x = jnp.exp(jnp.linspace(-90, 90, 1024))
  x = jnp.concatenate([-x[::-1], x], 0)

  rgb = rgb_activation(x)
  if jnp.any(rgb < 0) or jnp.any(rgb > 1):
    raise NotImplementedError(
        "Choice of rgb_activation `{}` produces colors outside of [0, 1]"
        .format(args.rgb_activation))

  sigma = sigma_activation(x)
  if jnp.any(sigma < 0):
    raise NotImplementedError(
        "Choice of sigma_activation `{}` produces negative densities".format(
            args.sigma_activation))

  if args.num_train_cameras:
    num_cameras = args.num_train_cameras
  else:
    # This is the dataset passed to get_model.
    num_cameras = dataset.camtoworlds.shape[0]

  camtoworlds = jnp.array(dataset.camtoworlds)
  camtoworlds = jnp.concatenate([
      camtoworlds, jnp.array([0, 0, 0, 1]).reshape(-1, 1, 4).tile(
          (24,1,1))], axis=1)

  image_sizes = jnp.array(dataset.image_sizes)
  fx_fy_cx_cy = jnp.array(dataset.fx_fy_cx_cy)

  model = NerfModel(
      min_deg_point=args.min_deg_point,
      max_deg_point=args.max_deg_point,
      deg_view=args.deg_view,
      num_coarse_samples=args.num_coarse_samples,
      num_fine_samples=args.num_fine_samples,
      use_viewdirs=args.use_viewdirs,
      near=args.near,
      far=args.far,
      noise_std=args.noise_std,
      white_bkgd=args.white_bkgd,
      net_depth=args.net_depth,
      net_width=args.net_width,
      net_depth_condition=args.net_depth_condition,
      net_width_condition=args.net_width_condition,
      skip_layer=args.skip_layer,
      num_rgb_channels=args.num_rgb_channels,
      num_sigma_channels=args.num_sigma_channels,
      lindisp=args.lindisp,
      net_activation=net_activation,
      rgb_activation=rgb_activation,
      sigma_activation=sigma_activation,
      legacy_posenc_order=args.legacy_posenc_order,
      tags=utils_extra.parse_tags(args),
      num_cameras=num_cameras,
      camtoworlds=camtoworlds,
      image_sizes=image_sizes,
      fx_fy_cx_cy=fx_fy_cx_cy)
  print("num_cameras: {}, tags: {}".format(model.num_cameras, model.tags))
  for tag in model.tags:
    if tag not in utils.VALID_TAGS:
      raise ValueError("Unrecognized tag: {}".format(tag))

  rays = example_batch["rays"]
  camera_ids = example_batch["camera_ids"]
  ray_ids = example_batch["ray_ids"]
  key1, key2, key3 = random.split(key, num=3)

  logging.info("rays: %s", str(utils.namedtuple_map(lambda x: x.shape, rays)))
  logging.info("camera_ids: %s", str(camera_ids.shape))
  logging.info("ray_ids: %s", str(ray_ids.shape))
  logging.info("camtoworlds: %s", str(camtoworlds.shape))

  init_variables = model.init(
      key1,
      rng_0=key2,
      rng_1=key3,
      rays=utils.namedtuple_map(lambda x: x[0], rays),
      randomized=args.randomized,
      barf_alpha=1.0,
      camera_ids=camera_ids[0],
      ray_ids=ray_ids[0],
      )

  return model, init_variables
