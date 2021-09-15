import jax
import jax.numpy as jnp


@jax.jit
def se3_exp_matrix(tangent):
  """Computes SE3 rotation-translation matrix.

  Ported from the jaxlie library:
    https://github.com/brentyi/jaxlie/blob/master/jaxlie/_se3.py#L106

  Equivalent to `SE3.exp(tangent).as_matrix()`.

  Args:
    tangent: A vector of size 6.

  Returns:
    A matrix of shape (4, 4)

  """
  assert tangent.shape == (6,)
  epsilon = 1e-5

  theta_squared = tangent[:3] @ tangent[:3]
  theta_pow_4 = theta_squared * theta_squared
  use_taylor = theta_squared < epsilon

  # Avoid NaNs.
  safe_theta_squared = jnp.where(
      use_taylor,
      1.0,
      theta_squared,
  )
  safe_theta = jnp.sqrt(safe_theta_squared)
  safe_half_theta = 0.5 * safe_theta

  real_factor = jnp.where(
      use_taylor,
      1.0 - theta_squared / 8.0 + theta_pow_4 / 384.0,
      jnp.cos(safe_half_theta),
  )

  imaginary_factor = jnp.where(
      use_taylor,
      0.5 - theta_squared / 48.0 + theta_pow_4 / 3840.0,
      jnp.sin(safe_half_theta) / safe_theta,
  )

  wxyz = jnp.concatenate([
      real_factor[None],
      imaginary_factor * tangent[:3],
  ])

  norm = wxyz @ wxyz
  q = wxyz * jnp.sqrt(2.0 / norm)
  q = jnp.outer(q, q)
  rotation = jnp.array([
      [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
      [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
      [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
  ])

  wx, wy, wz = tangent[:3]
  skew_omega = jnp.array([
      [0.0, -wz, wy],
      [wz, 0.0, -wx],
      [-wy, wx, 0.0],
  ])

  V = jnp.where(
      use_taylor,
      rotation,
      (jnp.eye(3) + (1.0 - jnp.cos(safe_theta)) /
       (safe_theta_squared) * skew_omega + (safe_theta - jnp.sin(safe_theta)) /
       (safe_theta_squared * safe_theta) * (skew_omega @ skew_omega)),
  )

  translation = V @ tangent[3:]

  return jnp.eye(4).at[:3, :3].set(rotation).at[:3, 3].set(translation)


@jax.jit
def axis_angle_rotation_translation(pose):
  """Computes transformation matrix from axis-angle representation.

  Args:
    pose: A vector of size 6.

  Returns:
    A matrix of shape (4, 4)

  """
  rotation = Rxyz(pose[0], pose[1], pose[2])
  translation = pose[3:]
  return jnp.eye(4).at[:3, :3].set(rotation).at[:3, 3].set(translation)


def Rxyz(theta_x, theta_y, theta_z):
  """Axis-angle rotations around x, y, z axes."""
  return Rz(theta_z).dot(Ry(theta_y).dot(Rx(theta_x)))


def Rx(theta):
  """Axis-angle rotation around x axis."""
  return jnp.array([
      [1, 0, 0],
      [0, jnp.cos(theta), jnp.sin(theta)],
      [0, -jnp.sin(theta), jnp.cos(theta)],
  ])


def Ry(theta):
  """Axis-angle rotation around y axis."""
  return jnp.array([
      [jnp.cos(theta), 0, -jnp.sin(theta)],
      [0, 1, 0],
      [jnp.sin(theta), 0, jnp.cos(theta)],
  ])


def Rz(theta):
  """Axis-angle rotation around z axis."""
  return jnp.array([
      [jnp.cos(theta), jnp.sin(theta), 0],
      [-jnp.sin(theta), jnp.cos(theta), 0],
      [0, 0, 1],
  ])
