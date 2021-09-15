"""Utility functions used in train.py and eval.py. Extends utils.py."""
import pprint
from typing import List

from absl import flags
from absl import logging
import numpy as np
import yaml


def define_flags():
  flags.DEFINE_list("config_extra_encoded", None,
                    "Hex-encoded yaml config string from xm_launcher.py."
                    "Comma separated. ")


def decode_yaml(encoded: str):
  content = bytes.fromhex(encoded)
  return yaml.load(content, Loader=yaml.SafeLoader)


def update_flags_from_dict(args, arg_dict):
  """Update the flags in `args` with `arg_dict`. Used in notebooks."""
  # Only allow args to be updated if they already exist.
  invalid_args = list(set(arg_dict.keys()) - set(dir(args)))
  if invalid_args:
    raise ValueError(
        f"Invalid args {invalid_args}.")
  args.__dict__.update(arg_dict)


def update_extra_flags(args):
  """Update the flags in `args` with the contents of the config YAML file."""
  if args.config_extra_encoded:
    for encoded in args.config_extra_encoded:
      configs = decode_yaml(encoded)
      logging.info("Extra params:\n%s", pprint.pformat(configs))
      update_flags_from_dict(args, configs)
    args.__dict__["config_extra_encoded"] = None


def log_params(args: flags.FlagValues):
  """Log the values of frequently changing arguments."""
  names = [
      "batch_size",
      "data_dir",
      "train_dir",
      "white_bkgd",
      "max_steps",
      "tags",
      "num_train_cameras",
  ]

  logged_str = "\n".join([f"{name}: {getattr(args, name)}" for name in names])
  logging.info(f"Params:\n%s", logged_str)


def parse_tags(args: flags.FlagValues) -> List[str]:
  """Parse comma separated tags from flags.

  Args:
    args: A FLAGS instance.

  Returns:
    A list of strings. Duplicates are removed and order is preserved.

  """
  if args.tags:
    if isinstance(args.tags, (list, tuple)):
      tags = list(args.tags)
    elif isinstance(args.tags, str):
      tags = [tag.strip() for tag in args.tags.split(",")]
    else:
      raise ValueError(f"Unrecognized tags: {args.tags}")
  else:
    tags = []
  return list(dict.fromkeys(tags))


def flags_dict_by_module(args: flags.FlagValues, name_substr: str):
  """Extract a dict of FLAGS instance, for tensorboard hparam logs.

  Args:
    args: A FLAGS instance.
    name_substr: Substring of the module name to search for.

  Returns:
    A dict of flag names and values.

  """
  by_module_name = args.flags_by_module_dict()
  ret = {}
  for module_name, module_flags in by_module_name.items():
    if name_substr in module_name:
      for flag in module_flags:
        ret[flag.name] = getattr(args, flag.name)
  return ret


def display_gif(filename):
  # Lazy import. Only used in notebooks.
  from IPython import display
  with open(filename, "rb") as f:
    im = display.Image(data=f.read(), format="png")
  return im


def show_pred_gt(pred, gt, save_superimposed=False):
  # Lazy import.
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(1, 4, figsize=(13, 3))
  axes[0].imshow(pred)
  axes[0].axis("off")
  axes[0].set_title("Pred")
  axes[1].imshow(gt)
  axes[1].axis("off")
  axes[1].set_title("GT")
  l2 = ((pred - gt)**2).mean(-1)
  im = axes[2].imshow(l2)
  axes[2].axis("off")
  axes[2].set_title("L2 Error")
  divider = make_axes_locatable(axes[2])
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar(im, cax=cax, orientation="vertical")
  superimposed = np.zeros_like(pred)
  superimposed[:, :, 0] = pred.mean(-1)
  superimposed[:, :, 1] = gt.mean(-1)
  axes[3].imshow(superimposed)
  axes[3].axis("off")
  axes[3].set_title("Superimposed (R=pred, G=GT, B=0)")
  plt.tight_layout()
  plt.show()

  if save_superimposed:
    # Notebook only.
    from google3.experimental.users.daeyun.jaxnerf.nerf import utils
    utils.save_global_state("superimposed", superimposed, append=True)
