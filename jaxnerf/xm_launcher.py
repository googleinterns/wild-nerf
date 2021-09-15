# Lint as: python3
"""Launcher script for train/test jobs."""
import os
import pprint
import sys
import time
from typing import List

from absl import app
from absl import flags
from absl import logging
import yaml

# pylint: disable=g-import-not-at-top
if 'colab_kernel' not in os.environ['_']:
  # Skip if imported from colab.
  from google3.learning.deepmind.python.adhoc_import import binary_import
  from google3.learning.deepmind.xmanager import hyper
  import google3.learning.deepmind.xmanager2.client.google as xm
  from google3.learning.deepmind.xmanager2.contrib.tpu import xm_tpu

  with binary_import.AutoGoogle3():
    from google3.learning.brain.research.jax.xmanager import jax_xm

  FLAGS = flags.FLAGS
  flags.DEFINE_string("cell", "el", "Cell on which to launch the jobs.")
  flags.DEFINE_string("cell_eval", "lu",
                      "Cell on which to launch the evaluation jobs.")
  flags.DEFINE_integer("priority", 200, "Priority to which launch the job.")
  flags.DEFINE_integer("n_gpus", 1, "Number of gpus per train worker.")
  flags.DEFINE_integer("n_gpus_eval", 1, "Number of gpus per eval worker.")
  flags.DEFINE_string(
      "train_dir",
      None,  # Like: "/cns/{cell}-d/home/bydeng/projects/",
      "Experiment path.")
  flags.mark_flag_as_required("train_dir")
  flags.DEFINE_string(
      "data_dir",
      None,  # Like: "/cns/{cell}-d/home/bydeng/datasets/",
      "Data path.")
  flags.mark_flag_as_required("data_dir")
  flags.DEFINE_string("config", None,
                      "using config files to set hyperparameters.")
  flags.DEFINE_list("config_extra", None,
                    "using config files to set hyperparameters. "
                    "Comma separated and overwritten in order.")
  flags.DEFINE_bool("is_train", True, "The job is in the training mode.")
  flags.DEFINE_bool("use_tpu", True, "Whether to use tpu for training.")
  flags.DEFINE_enum("tpu_topology", "2x2", ["2x2", "4x2", "4x4", "8x8"],
                    "TPU topology for training.")
  flags.DEFINE_enum("tpu_platform", "jf", ["jf", "df"],
                    "TPU platform for training.")
  flags.DEFINE_bool("use_tpu_eval", False, "Whether to use tpu for evaluation.")
  flags.DEFINE_enum("tpu_topology_eval", "2x2", ["2x2", "4x2", "4x4", "8x8"],
                    "TPU topology for evaluation.")
  flags.DEFINE_enum("tpu_platform_eval", "jf", ["jf", "df"],
                    "TPU platform for evaluation.")
  flags.DEFINE_integer(
      "chunk", None, "the size of chunks for evaluation inferences, set to"
      "the value that fits your GPU/TPU memory.")
  flags.DEFINE_bool("batch_launch", False,
                    "launch jobs for all scenes in a dataset if set True.")
  flags.DEFINE_string("nametag", "jaxnerf", "A name to prepend to the xman name.")
  flags.DEFINE_multi_string("tags", None,
                            "A comma separated list of tags passed to the model. "
                            "Can be specified more than once.")
  flags.DEFINE_string("xm_tags", None,
                      "A comma separated list of experiment tags. Appears in "
                      "xmanager experiment description.")
else:
  # Arbitrary attributes can be assigned for debugging.
  FLAGS = type('', (), {})()


def encode_yaml(filename: str, exclude_comments=False) -> str:
  """Read a yaml file, validate, and return a hex encoded string."""
  with open(filename, "r") as fin:
    content = fin.read()
  try:
    yaml.load(content, Loader=yaml.SafeLoader)
  except Exception as ex:
    print("Could not parse config file {} with content:\n{}".format(
        filename, content), file=sys.stderr)
    raise ex
  if exclude_comments:
    content = "\n".join([line for line in content.splitlines()
                         if not line.lstrip().startswith("#")])
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


def get_hyperparameters(exp_dir):
  """Generate hyperparameters."""
  if FLAGS.batch_launch:
    # NOTE: --batch_launch is disabled for now.
    raise NotImplementedError("--batch_launch flag should not be used in our "
                              "experiments.")
    if "blender" in FLAGS.config:
      scenes = ("chair", "ship", "ficus", "mic", "hotdog", "drums", "materials",
                "lego")
    else: # Originally: elif "llff" in FLAGS.config:
      scenes = ("trex", "leaves", "room", "orchids", "horns", "fortress",
                "fern", "flower")
    return hyper.zipit([
        hyper.sweep(
            "data_dir",
            hyper.categorical(
                list(
                    os.path.join(FLAGS.data_dir.format(cell=FLAGS.cell), x)
                    for x in scenes))),
        hyper.sweep(
            "train_dir",
            hyper.categorical(list(os.path.join(exp_dir, x) for x in scenes))),
    ])
  else:
    return [{}]


def parse_tags(args: flags.FlagValues) -> List[str]:
  """Parse comma separated tags from --tags multi_string flags.

  Args:
    args: A FLAGS instance.

  Returns:
    A list of strings. Duplicates are removed and order is preserved.

  """
  tags = []
  if args.tags:
    for item in args.tags:
      tags.extend([tag.strip() for tag in item.split(",")])
  return list(dict.fromkeys(tags))


def get_train_dict(exp_dir):
  """Generate training job parameter dict."""
  params = {
      "config": FLAGS.config,
      "jax_tpu_async": 1,
      "tags": ",".join(parse_tags(FLAGS)),
  }
  if FLAGS.config_extra:
    encoded = ",".join([
        load_extra_config_encoded(filename) for filename in FLAGS.config_extra
    ])
    params["config_extra_encoded"] = encoded
  if not FLAGS.batch_launch:
    params["data_dir"] = FLAGS.data_dir.format(cell=FLAGS.cell)
    params["train_dir"] = exp_dir
  logging.info("Train params:\n %s", pprint.pformat(params))
  return params


def get_test_dict(exp_dir):
  """Generate testing job parameter dict."""
  params = {
      "config": FLAGS.config,
  }
  if FLAGS.config_extra:
    encoded = ",".join([
        load_extra_config_encoded(filename) for filename in FLAGS.config_extra
    ])
    params["config_extra_encoded"] = encoded
  if not FLAGS.batch_launch:
    params["data_dir"] = FLAGS.data_dir.format(cell=FLAGS.cell_eval)
    params["train_dir"] = exp_dir
  if FLAGS.is_train:
    params["eval_once"] = False
  if FLAGS.chunk is not None:
    params["chunk"] = FLAGS.chunk
  logging.info("Test params:\n %s", pprint.pformat(params))
  return params


def main(argv):
  del argv  # Unused.
  if FLAGS.is_train:
    FLAGS.train_dir = FLAGS.train_dir.format(cell=FLAGS.cell)
  exp_name = get_exp_name()

  if (not FLAGS.batch_launch) and FLAGS.is_train:
    exp_dir = os.path.join(
        FLAGS.train_dir, "%{0}%_%{1}%".format(xm.experiment.id,
                                              xm.experiment.work_unit_id))
  else:
    exp_dir = FLAGS.train_dir

  # Job: train
  executables = []

  # Construct training job executables
  if FLAGS.is_train:
    if FLAGS.use_tpu:
      platform = xm.Platform.from_str(FLAGS.tpu_platform)
      topology = xm.TpuTopology(FLAGS.tpu_topology)
      overrides, imports, build_target_args = jax_xm.tpu_configuration(
          platform, topology)
      requirements = jax_xm.tpu_default_requirements(platform, topology)
      mixins = [xm_tpu.JaxTpuMixin()]
      exec_train = xm.BuildTarget(
          "//experimental/users/daeyun/jaxnerf:train",
          runtime=xm.Borg(
              cell=FLAGS.cell,
              priority=FLAGS.priority,
              # Uncomment this line if you want others to be able to debug your
              # trainer by looking at your logs:
              # logs_read_access_roles=['all'],
              overrides=overrides,
              requirements=requirements,
              imports=imports),
          name="train_worker",
          args={
              **build_target_args,
              **get_train_dict(exp_dir)
          },
          platform=platform,
          mixins=mixins)
    else:
      train_require = xm.Requirements(
          gpu=FLAGS.n_gpus,
          gpu_types=[xm.GpuType.V100],
      )
      train_worker = xm.Borg(
          cell=FLAGS.cell,
          priority=FLAGS.priority,
          # Uncomment this line if you want others to be able to debug your
          # trainer by looking at your logs:
          # logs_read_access_roles=['all'],
          requirements=train_require)
      exec_train = xm.BuildTarget(
          "//experimental/users/daeyun/jaxnerf:train",
          name="train_worker",
          args=get_train_dict(exp_dir),
          platform=xm.Platform.GPU,
          runtime=train_worker)
    executables.append(exec_train)

  # Construct evaluation job executables
  if FLAGS.use_tpu_eval:
    platform = xm.Platform.from_str(FLAGS.tpu_platform_eval)
    topology = xm.TpuTopology(FLAGS.tpu_topology_eval)
    overrides, imports, build_target_args = jax_xm.tpu_configuration(
        platform, topology)
    requirements = jax_xm.tpu_default_requirements(platform, topology)
    mixins = [xm_tpu.JaxTpuMixin()]
    exec_eval = xm.BuildTarget(
        "//experimental/users/daeyun/jaxnerf:eval",
        runtime=xm.Borg(
            cell=FLAGS.cell_eval,
            priority=FLAGS.priority,
            # Uncomment this line if you want others to be able to debug your
            # trainer by looking at your logs:
            # logs_read_access_roles=['all'],
            overrides=overrides,
            requirements=requirements,
            imports=imports),
        name="eval_worker",
        args={
            **build_target_args,
            **get_test_dict(exp_dir)
        },
        platform=platform,
        mixins=mixins)
  else:
    eval_require = xm.Requirements(
        gpu=FLAGS.n_gpus_eval,
        gpu_types=[xm.GpuType.V100],
    )
    eval_worker = xm.Borg(
        cell=FLAGS.cell_eval,
        priority=FLAGS.priority,
        # Uncomment this line if you want others to be able to debug your
        # trainer by looking at your logs:
        # logs_read_access_roles=['all'],
        requirements=eval_require)
    exec_eval = xm.BuildTarget(
        "//experimental/users/daeyun/jaxnerf:eval",
        name="eval_worker",
        args=get_test_dict(exp_dir),
        platform=xm.Platform.GPU,
        runtime=eval_worker)
  executables.append(exec_eval)

  nametag = FLAGS.nametag.replace(os.sep, "_").lower()
  # Combine train and eval
  experiment = xm.ParallelExecutable(executables, name=nametag + "_service")

  hyper_parameters = get_hyperparameters(exp_dir)
  experiment = xm.ParameterSweep(experiment, hyper_parameters)
  if FLAGS.is_train:
    experiment = xm.WithTensorBoard(experiment, FLAGS.train_dir)

  if FLAGS.xm_tags:
    xm_tags = [item.strip() for item in FLAGS.xm_tags.split(",")]
  else:
    xm_tags = None

  # Launch experiments
  description = xm.ExperimentDescription(nametag + " " + exp_name, tags=xm_tags)
  xm.launch_experiment(description, experiment)


if __name__ == "__main__":
  app.run(main)
