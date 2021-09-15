"""Tests for model_utils."""

from unittest import mock

import jax.numpy as jnp
import numpy as np

from google3.experimental.users.daeyun.jaxnerf.nerf import datasets
from google3.experimental.users.daeyun.jaxnerf.nerf import model_utils
from google3.testing.pybase import googletest


class ModelUtilsTest(googletest.TestCase):

  def test_generate_rays(self):
    """Tests for model_utils.generate_rays.

    Checks if model_utils.generate_rays generates the same rays as
    Dataset._generate_rays.
    """
    # Create a mock dataset object with the camera values we want to test.
    dataset = mock.MagicMock()
    dataset.use_pixel_centers = True
    # Sample parameters from ShapeNet.
    dataset.w = 224
    dataset.h = 224
    dataset.camtoworlds = np.array(
        [[[0.92515334, 0.03917265, -0.37756686, -0.447942],
          [-0.29496273, 0.70026623, -0.6500955, -0.77126745],
          [0.23893135, 0.71280618, 0.65940811, 0.78231584]],
         [[0.98747369, -0.12547567, -0.09566391, -0.11360313],
          [0.01201645, 0.66434125, -0.74733275, -0.8874754],
          [0.15732557, 0.73682188, 0.65752727, 0.78082921]]],
        dtype=np.float64)
    dataset.focal = 240.1847686767578
    dataset.resolution = 224 * 224
    dataset._load_renderings.return_value = None
    dataset.images = np.zeros((2, 224, 224, 3), dtype=np.float32)

    # Arguments, typically a FLAGS instance.
    args = mock.MagicMock()
    args.batching = 'all_images'
    args.use_world_coord_rays = True

    # Generate rays and flatten.
    datasets.Dataset._generate_rays(dataset, args)
    datasets.Dataset._train_init(dataset, args)

    # Same camera intrinsics as above.
    ray_ids = jnp.array([2, 3, 100000])
    cx = jnp.full(shape=(2,), fill_value=112)
    cy = jnp.full(shape=(2,), fill_value=112)
    fx = jnp.full(shape=(2,), fill_value=dataset.focal)
    fy = jnp.full(shape=(2,), fill_value=dataset.focal)
    image_hw = jnp.full(shape=(2, 2), fill_value=224)

    # Call our new implementation.
    rays, camera_ids = model_utils.generate_rays(ray_ids, dataset.camtoworlds,
                                                 image_hw, cx, cy, fx, fy)

    # Rays and camera ids are equal.
    ray_ids_np = np.array(ray_ids)
    np.testing.assert_array_almost_equal(rays.directions,
                                         dataset.rays.directions[ray_ids_np])
    np.testing.assert_array_almost_equal(rays.origins,
                                         dataset.rays.origins[ray_ids_np])
    np.testing.assert_array_almost_equal(rays.viewdirs,
                                         dataset.rays.viewdirs[ray_ids_np])

    np.testing.assert_array_equal(camera_ids, [0, 0, 1])
    np.testing.assert_array_equal(dataset.camera_ids[ray_ids_np], [0, 0, 1])


if __name__ == '__main__':
  googletest.main()
