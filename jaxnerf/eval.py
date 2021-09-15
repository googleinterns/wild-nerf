# Lint as: python3
"""Evaluation script for Nerf."""
import functools
from os import path
import time
import pprint

from absl import app
from absl import flags
from absl import logging
from absl.logging import DEBUG, INFO
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub

from google3.experimental.users.daeyun.jaxnerf.nerf import datasets
from google3.experimental.users.daeyun.jaxnerf.nerf import models
from google3.experimental.users.daeyun.jaxnerf.nerf import utils
from google3.experimental.users.daeyun.jaxnerf.nerf import utils_extra
# BEGIN GOOGLE-INTERNAL
import google3.learning.deepmind.xmanager2.client.google as xm  # pylint: disable=unused-import
# END GOOGLE-INTERNAL

FLAGS = flags.FLAGS

try:
  utils.define_flags()
  utils_extra.define_flags()
except flags._exceptions.DuplicateFlagError as ex:
  pass

LPIPS_TFHUB_PATH = "@neural-rendering/lpips/distance/1"


def compute_lpips(image1, image2, model):
  """Compute the LPIPS metric."""
  # The LPIPS model expects a batch dimension.
  return model(
      tf.convert_to_tensor(image1[None, ...]),
      tf.convert_to_tensor(image2[None, ...]))[0]


def main(unused_argv=None, update_flags=True):
  logging.log(INFO, "eval.main() started. Flags\n%s",
              pprint.pformat(utils_extra.flags_dict_by_module(FLAGS, "nerf")))
  # Hide the GPUs and TPUs from TF so it does not reserve memory on them for
  # LPIPS computation or dataset loading.
  try:
    tf.config.experimental.set_visible_devices([], "GPU")
    tf.config.experimental.set_visible_devices([], "TPU")
  except RuntimeError:
    # main() is called more than once, e.g. in a notebook.
    pass

  rng = random.PRNGKey(20200823)

  if update_flags:
    if FLAGS.config is not None:
      utils.update_flags(FLAGS)
    if FLAGS.config_extra_encoded is not None:
      # Overwrites existing args.
      utils_extra.update_extra_flags(FLAGS)
  if FLAGS.train_dir is None:
    raise ValueError("train_dir must be set. None set now.")
  if FLAGS.data_dir is None:
    raise ValueError("data_dir must be set. None set now.")

  dataset = datasets.get_dataset("test", FLAGS)
  utils.save_global_state('test_dataset', dataset)
  rng, key = random.split(rng)
  model, init_variables = models.get_model(key, dataset.peek(), FLAGS, dataset)
  optimizer = flax.optim.Adam(FLAGS.lr_init).create(init_variables)
  state = utils.TrainState(optimizer=optimizer)
  del optimizer, init_variables
  logging.info("Initialized model.")

  lpips_model = tf_hub.load(LPIPS_TFHUB_PATH)
  logging.info("Initialized lpips_model.")

  # Rendering is forced to be deterministic even if training was randomized, as
  # this eliminates "speckle" artifacts.
  # BARF alpha is always 1.0 for testing.
  def render_fn(variables, key_0, key_1, rays, camera_ids, ray_ids):
    return jax.lax.all_gather(
        model.apply(variables, key_0, key_1, rays, False, 1.0, camera_ids, ray_ids, True),
        axis_name="batch")

  # pmap over only the data input.
  render_pfn = jax.pmap(
      render_fn,
      in_axes=(None, None, None, 0, 0, 0),
      donate_argnums=(3, 4),
      axis_name="batch",
  )

  # Compiling to the CPU because it's faster and more accurate.
  ssim_fn = jax.jit(
      functools.partial(utils.compute_ssim, max_val=1.), backend="cpu")

  last_step = 0
  out_dir = path.join(FLAGS.train_dir,
                      "path_renders" if FLAGS.render_path else "test_preds")
  if not FLAGS.eval_once:
    summary_writer = tensorboard.SummaryWriter(
        path.join(FLAGS.train_dir, "eval"))
  while True:
    # Prevents polling too quickly while waiting for a checkpoint.
    time.sleep(0.5)

    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
    utils.save_global_state("state", state, verbose=False)
    step = int(state.optimizer.state.step)
    if step <= last_step:
      continue
    if FLAGS.save_output and (not utils.isdir(out_dir)):
      utils.makedirs(out_dir)
    psnr_values = []
    ssim_values = []
    lpips_values = []
    if not FLAGS.eval_once:
      showcase_index = np.random.randint(0, dataset.size)
    for idx in range(dataset.size):
      print(f"Evaluating {idx+1}/{dataset.size}")
      batch = next(dataset)
      pred_color, pred_disp, pred_acc = utils.render_image(
          functools.partial(render_pfn, state.optimizer.target),
          batch["rays"],
          batch["camera_ids"],
          batch["ray_ids"],
          rng,
          True,  # Originally, only true for --dataset=llff.
          chunk=FLAGS.chunk)
      if jax.process_index() != 0:  # Only record via host 0.
        continue
      if not FLAGS.eval_once and idx == showcase_index:
        showcase_color = pred_color
        showcase_disp = pred_disp
        showcase_acc = pred_acc
        if not FLAGS.render_path:
          showcase_gt = batch["pixels"]
      if not FLAGS.render_path:
        psnr = utils.compute_psnr(((pred_color - batch["pixels"])**2).mean())
        ssim = ssim_fn(pred_color, batch["pixels"])
        lpips = compute_lpips(pred_color, batch["pixels"], lpips_model)
        print(f"PSNR = {psnr:.4f}, SSIM = {ssim:.4f}")
        psnr_values.append(float(psnr))
        ssim_values.append(float(ssim))
        lpips_values.append(float(lpips))
      if FLAGS.save_output:
        utils.save_img(pred_color, path.join(
            out_dir, "rgb_{:03d}.png".format(idx)))
        utils.save_img(pred_disp[..., 0],
                       path.join(out_dir, "disp_{:03d}.png".format(idx)))
      if FLAGS.eval_matplotlib:
        utils_extra.show_pred_gt(pred_color, batch["pixels"])

    if (not FLAGS.eval_once) and (jax.process_index() == 0):
      summary_writer.image("pred_color", showcase_color, step)
      summary_writer.image("pred_disp", showcase_disp, step)
      summary_writer.image("pred_acc", showcase_acc, step)
      if not FLAGS.render_path:
        summary_writer.scalar("psnr", np.mean(np.array(psnr_values)), step)
        summary_writer.scalar("ssim", np.mean(np.array(ssim_values)), step)
        summary_writer.scalar("lpips", np.mean(np.array(lpips_values)), step)
        summary_writer.image("target", showcase_gt, step)
    if FLAGS.save_output and (not FLAGS.render_path) and (
        jax.process_index() == 0):
      with utils.open_file(path.join(out_dir, f"psnrs_{step}.txt"), "w") as f:
        f.write(" ".join([str(v) for v in psnr_values]))
      with utils.open_file(path.join(out_dir, f"ssims_{step}.txt"), "w") as f:
        f.write(" ".join([str(v) for v in ssim_values]))
      with utils.open_file(path.join(out_dir, f"lpips_{step}.txt"), "w") as f:
        f.write(" ".join([str(v) for v in lpips_values]))
      with utils.open_file(path.join(out_dir, "psnr.txt"), "w") as f:
        f.write("{}".format(np.mean(np.array(psnr_values))))
      with utils.open_file(path.join(out_dir, "ssim.txt"), "w") as f:
        f.write("{}".format(np.mean(np.array(ssim_values))))
      with utils.open_file(path.join(out_dir, "lpips.txt"), "w") as f:
        f.write("{}".format(np.mean(np.array(lpips_values))))
      print("PSNR", np.mean(np.array(psnr_values)))
      print("SSIM", np.mean(np.array(ssim_values)))
      print("LPIPS", np.mean(np.array(lpips_values)))
    if FLAGS.eval_once:
      break
    if int(step) >= FLAGS.max_steps:
      break
    last_step = step


if __name__ == "__main__":
  app.run(main)
