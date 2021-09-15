"""Commonly used local file IO functions."""
import os
from os import path

from cvx2 import latest as cv2
import numpy as np


def ensure_dir_exists(dirname):
  """Analogous to `mkdir -p {dirname}`"""

  if not os.path.isdir(dirname):
    os.makedirs(dirname)
  # Could be an existing file, not a directory.
  assert os.path.isdir(dirname)


def read_rgb(filename, format='hwc', resize=None):
  assert format in ('hwc', 'chw'), format
  assert path.isfile(filename), filename
  image = cv2.cvtColor(
      cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

  if isinstance(resize, (int, float)):
    image = cv2.resize(
        image,
        tuple((np.array(image.shape[:2]) * resize).astype(int).tolist()[::-1]),
        interpolation=cv2.INTER_LINEAR)
  elif isinstance(resize, (list, tuple)):
    # if tuple, height,width should be provided.
    image = cv2.resize(
        image, tuple(resize[1], resize[0]), interpolation=cv2.INTER_LINEAR)

  if format == 'chw':
    # NOTE: Might need to make sure its contiguous.
    image = image.transpose((2, 0, 1))
  return image
