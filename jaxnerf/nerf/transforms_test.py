import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla

from google3.experimental.users.daeyun.jaxnerf.nerf import transforms
from google3.testing.pybase import googletest


class TransformsTest(googletest.TestCase):

  def _hat(self, delta: np.ndarray) -> np.ndarray:
    """SE3 hat-map."""
    return np.array([(0, -delta[2], delta[1], delta[3]),
                     (delta[2], 0, -delta[0], delta[4]),
                     (-delta[1], delta[0], 0, delta[5]), (0, 0, 0, 0)],
                    dtype=delta.dtype)

  def test_se3_exp_matrix(self):
    """Tests for transforms.se3_exp_matrix.

    Checks if the quaternion-based derivation returns the same output as the
    matrix exponential.
    """
    for i in range(1000):
      pose = np.random.default_rng(i).uniform(0, 10, size=6)
      np.testing.assert_allclose(
          transforms.se3_exp_matrix(pose),
          sla.expm(self._hat(pose)),
          rtol=1e-5,
          atol=1e-3)

    for i in range(1000):
      pose = np.random.default_rng(i).uniform(0, 1e-4, size=6)
      np.testing.assert_allclose(
          transforms.se3_exp_matrix(pose),
          sla.expm(self._hat(pose)),
          atol=1e-7)


if __name__ == '__main__':
  googletest.main()
