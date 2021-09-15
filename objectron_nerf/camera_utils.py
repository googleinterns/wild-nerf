"""Utility functions for quick camera coordinate conversion and projection."""

import hashlib
import math

import numpy as np
import numpy.linalg as la


class Camera(object):

  def __init__(self, R, t, s: float = None):
    """A wrapper object for frequently used 3D camera operations.

    Initialization
    assumes world-to-camera parameters.

    Args:
      R: (3,3) Rotation matrix. Must be orthogonal.
      t: (3,) Translation vector.
      s: Pre-transformation scale. Applied before rotating and translating.
    """

    # self.{R, t, s} must be world-to-camera.
    self.R = R
    self.t = t[:, None] if len(t.shape) == 1 else t
    self.s = 1.0 if s is None else float(s)

    self.R_inv, self.t_inv, self.s_pre_inv = self._inverse(
        self.R, self.t, self.s)
    self.pos = self.t_inv.ravel()

    self.viewdir = self.cam_to_world(np.array([0, 0, -1]).reshape(
        -1, 3)).ravel() - self.pos
    self.viewdir /= la.norm(self.viewdir)

    self.up_vector = self.cam_to_world(np.array([0, 1, 0]).reshape(
        -1, 3)).ravel() - self.pos
    self.up_vector /= la.norm(self.up_vector)

  def Rt(self):
    return np.hstack((self.R, self.t))

  def Rt_inv(self):
    return np.hstack((self.R_inv, self.t_inv))

  def sRt(self):
    return np.hstack((self.s * self.R, self.t))

  def sRt_inv(self):
    return np.hstack((self.s_pre_inv * self.R_inv, self.t_inv))

  def world_to_cam(self, xyzs):
    return self.sRt().dot(self._hom(xyzs).T).T

  def cam_to_world(self, xyzs):
    return self.sRt_inv().dot(self._hom(xyzs).T).T

  def cam_to_world_normal(self, xyzs):
    return self.R_inv.dot(xyzs.T).T

  def sRt_hash(self):
    """SHA1 hash of sRt matrix ignoring very small numerical differences.

    Mostly used to generate ad-hoc string IDs for cameras during
    development.

    Returns: A hexadecimal string of length 40.

    """
    sRt_flat = np.ascontiguousarray(self.sRt(), dtype=np.float64).ravel()
    values = ['{:.4f}'.format(item) for item in sRt_flat]

    # '-0.0000' and '0.0000' are replaced by '0'
    values = ['0' if float(value) == 0.0 else value for value in values]
    values_string = ','.join(values)

    sha1 = hashlib.sha1()
    sha1.update(values_string.encode('utf8'))
    ret = sha1.hexdigest()

    return ret

  @classmethod
  def _inverse(cls, R: np.ndarray, t: np.ndarray, s: float = 1):
    cls._check_orthogonal(R)
    if len(t.shape) == 1:
      t = t[:, None]

    R_inv = R.T
    t_inv = -R.T.dot(t) / s
    s_inv = 1.0 / s

    return R_inv, t_inv, s_inv

  @classmethod
  def _hom(cls, pts):
    assert pts.shape[1] in [2, 3]
    return np.hstack((pts, np.ones((pts.shape[0], 1))))

  @classmethod
  def _hom_inv(cls, pts):
    assert pts.shape[1] in [3, 4]
    return pts[:, :-1] / pts[:, -1, None]

  @classmethod
  def _check_orthogonal(cls, R, eps=1e-4):
    assert la.norm(R.T - la.inv(R), 2) < eps


class PerspectiveCamera(Camera):

  def __init__(self,
               R: np.ndarray,
               t: np.ndarray,
               K: np.ndarray,
               s: float = None):
    """Same as `Camera` except it takes a 3x3 projection matrix K."""
    super().__init__(R=R, t=t, s=s)

    # self.{R, t, s} must be world-to-camera.
    assert R.shape == (3, 3)
    assert K.shape == (3, 3)
    if len(t.shape) == 1:
      t = t[None, :]
    assert t.shape == (3, 1)

    self._check_orthogonal(R)

    self.K = K
    self.K_inv = la.inv(K)

  def __str__(self):
    return 'R:\n{}\nt:\n{}\nK:\n{}'.format(self.R, self.t, self.K)

  def position(self):
    return (-self.R.T.dot(self.t) / self.s).ravel()

  def image_to_world(self, xys):
    if xys.shape[1] == 2:
      xys = self._hom(xys)
    return self.cam_to_world(self.K_inv.dot(xys.T).T)

  def world_to_image(self, xyzs):
    P = self.projection_mat34()
    return self._hom_inv(P.dot(self._hom(xyzs).T).T)

  def cam_to_image(self, xyzs):
    return self._hom_inv(self.K.dot(xyzs.T).T)

  def projection_mat34(self):
    return self.K.dot(np.hstack((self.s * self.R, self.t)))


def inverse_Rt34(Rt, sanity_check=True):
  """Safely compute the inverse of SE(3) transformations."""
  assert Rt.shape == (3, 4)
  R = Rt[:, :3]
  t = Rt[:, 3:4]
  R_inv = R.T
  t_inv = -R.T.dot(t)
  Rt_inv = np.concatenate([R_inv, t_inv], axis=1)

  if sanity_check:
    Rt44_ = np.vstack([Rt, [0, 0, 0, 1]])
    assert np.allclose(
        Rt_inv, la.inv(Rt44_)[:3], rtol=1e-4,
        atol=1e-6), (Rt_inv, la.inv(Rt44_)[:3])

  return Rt_inv
