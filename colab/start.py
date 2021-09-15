"""Run this script in colab.

%run -i /google/src/cloud/daeyun/my_workspace_name/google3/experimental/users/daeyun/colab/start.py

Auto-reload wrapper:
aimport('google3.experimental.users.daeyun.jaxnerf.train', alias='train')
"""
# pylint: disable=unused-import
import glob
import importlib
import os
from os import path
import re
import sys

from colabtools import adhoc_import
from colabtools import googlefiles
from IPython import get_ipython

workspace_name = re.search('cloud/[^/]+/([^/]+)/google3', __file__).group(1)
print(f'workspace: {workspace_name}')

googlefiles.EnableGoogleFilesStat()
googlefiles.EnableOpenGoogleFiles()

with adhoc_import.Google3CitcClient(
    client_name=workspace_name, behavior='preferred', verbose=True):
  from google3.learning.deepmind.python.adhoc_import import colab_import
  from google3.learning.deepmind.python.adhoc_import import binary_import

if 'ipython' not in globals():
  ipython = get_ipython()
  ipython.magic('reload_ext autoreload')
  ipython.magic('autoreload 1')

colab_import.InitializeGoogle3(
    source=('daeyun', workspace_name),
    behavior='preferred',
    verbose=True,
    package_restrict='google3.experimental.users.daeyun',
)


def aimport(module_name, alias=None):
  if alias is None:
    alias = module_name
  with adhoc_import.Google3CitcClient(
      client_name=workspace_name, behavior='preferred', verbose=True,
      package_restrict='google3.experimental.users.daeyun'):
    module = importlib.import_module(module_name)
    module = adhoc_import.Reload(module, reset_flags=True)
    ipython.magic('aimport {}'.format(module_name))
    globals()[alias] = module


def encode_yaml(filename: str) -> str:
  """Read a yaml file, validate, and return a hex encoded string."""
  with open(filename, "r") as fin:
    content = fin.read()
  try:
    yaml.load(content, Loader=yaml.SafeLoader)
  except Exception as ex:
    print("Could not parse config file {} with content:\n{}".format(
        filename, content), file=sys.stderr)
    raise ex
  return content.encode("utf-8").hex()


def load_extra_config_encoded(file_basename):
  """Read a yaml file in ./configs_extra directory."""
  basedir = os.path.join(os.path.dirname(__file__), "configs_extra")
  if not file_basename.endswith(".yaml"):
    file_basename += ".yaml"
  filename = os.path.join(basedir, file_basename)
  if not os.path.isfile(filename):
    raise FileNotFoundError(f"File does not exist: {filename}")
  return encode_yaml(filename)


def get_exp_name():
  """Automatic experiment name generation for new training jobs."""
  config_name = FLAGS.config.replace(os.sep, "_")
  if FLAGS.is_train:
    cur_time = time.localtime()
    exp_name = config_name + "_" + ("{:02}{:02}{:02}{:02}{:02}".format(
        cur_time.tm_mon, cur_time.tm_mday, cur_time.tm_hour, cur_time.tm_min,
        cur_time.tm_sec))
    FLAGS.train_dir = os.path.join(FLAGS.train_dir, exp_name)
  else:
    exp_name = config_name

  return "{exp_name} : {exp_type}".format(
      exp_name=exp_name, exp_type="Train" if FLAGS.is_train else "Eval")

del __file__
