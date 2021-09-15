"""Helper functions for dealing with video files for data preparation.
"""

import glob
import os
import subprocess

from absl import logging
from cvx2 import latest as cv2
import numpy as np

from google3.experimental.users.daeyun.objectron_nerf import io_utils
from google3.research.xeno.nnets.common import files


def remote_extract_frames_ffmpeg(remote_filename, local_out_dir):
  with files.maybe_remote_clone(remote_filename) as local_path:
    logging.info('Reading frames from remote file %s (local path: %s)',
                 remote_filename, local_path)

    local_out_dir = os.path.expanduser(local_out_dir)
    io_utils.ensure_dir_exists(local_out_dir)

    out_pattern = os.path.join(local_out_dir, '%05d.png')

    command = f'ffmpeg -i {local_path} -vsync vfr -start_number 0 {out_pattern}'
    logging.info('Running command: %s', command)
    os.system(command)


def recursive_image_resize(source_dir, out_dir, downsample=8, ext='png'):
  io_utils.ensure_dir_exists(out_dir)
  filenames = sorted(
      glob.glob(os.path.join(source_dir, f'**/*.{ext}'), recursive=True))

  for filename in filenames:
    relpath = os.path.relpath(filename, source_dir)
    outfile = os.path.join(out_dir, relpath)
    if os.path.isfile(outfile):
      continue

    io_utils.ensure_dir_exists(os.path.dirname(outfile))

    print(f'{filename}, {outfile}')

    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    height, width, _ = img.shape
    resized_height = height // downsample
    resized_width = width // downsample

    img = cv2.resize(
        img, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    cv2.imwrite(outfile, img)
    assert os.path.isfile(outfile)


def remote_video_frame_gen(remote_filename: str, downsample: int = 1):
  with files.maybe_remote_clone(remote_filename) as local_path:
    logging.info('Reading frames from remote file %s (local path: %s)',
                 remote_filename, local_path)

    capture = cv2.VideoCapture(local_path)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info('%d total frames', frame_count)

    while capture.isOpened():
      ret, img = capture.read()
      if not ret:
        break

      # See /research/drishti/app/pursuit/deep_pursuit_3d/dataset/converter/flume_annotation_to_tfrecord_lib.py
      height, width, _ = img.shape

      if width > height:
        # TODO(daeyun): Rotation orientation might not be consistent.
        # Ideally we should check the metadata.
        img = np.rot90(img, -1)
        height, width, _ = img.shape

      resized_height = height // downsample
      resized_width = width // downsample

      img = cv2.resize(
          img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      timestamp_mcsec = int(capture.get(cv2.CAP_PROP_POS_MSEC) * 1000)

      yield img, timestamp_mcsec

    capture.release()
