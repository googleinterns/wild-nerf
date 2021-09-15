# Lint as: python3
"""Training script for Nerf."""

import functools
import gc
import time
import copy
from absl import app
from absl import flags
from absl import logging
from absl.logging import INFO
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import config
from jax import random
import jax.numpy as jnp
import numpy as np
import pprint

from google3.experimental.users.daeyun.jaxnerf.nerf import datasets
from google3.experimental.users.daeyun.jaxnerf.nerf import models
from google3.experimental.users.daeyun.jaxnerf.nerf import utils
from google3.experimental.users.daeyun.jaxnerf.nerf import utils_extra

# BEGIN GOOGLE-INTERNAL
if __name__ == "__main__":
  import google3.learning.deepmind.xmanager2.client.google as xm  # pylint: disable=unused-import
# END GOOGLE-INTERNAL

FLAGS = flags.FLAGS

try:
  utils.define_flags()
  utils_extra.define_flags()
  config.parse_flags_with_absl()
except flags._exceptions.DuplicateFlagError as ex:
  pass


def barf_alpha(step):
  if FLAGS.barf_steps:
    ret = (jnp.clip(step / FLAGS.barf_steps, 0.0, 1.0)).astype(float)
  else:
    ret = 1.0
  return ret

def train_step(model, rng, state, batch, lr, step):
  """One optimization step.

  Args:
    model: The linen model.
    rng: jnp.ndarray, random number generator.
    state: utils.TrainState, state of the model/optimizer.
    batch: dict, a mini-batch of data for training.
    lr: float, real-time learning rate.
    step: int, global training step.

  Returns:
    new_state: utils.TrainState, new training state.
    stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
    rng: jnp.ndarray, updated random number generator.
  """
  rng, key_0, key_1 = random.split(rng, 3)

  def loss_fn(variables):
    alpha = barf_alpha(step)
    rays = batch["rays"]
    camera_ids = batch["camera_ids"]
    ray_ids = batch["ray_ids"]
    ret, ret_extra = model.apply(variables, key_0, key_1, rays, FLAGS.randomized, alpha, camera_ids, ray_ids, True)
    if len(ret) not in (1, 2):
      raise ValueError(
          "ret should contain either 1 set of output (coarse only), or 2 sets"
          "of output (coarse as ret[0] and fine as ret[1]).")
    # The main prediction is always at the end of the ret list.
    rgb, unused_disp, unused_acc = ret[-1]
    loss = ((rgb - batch["pixels"][..., :3])**2).mean()
    psnr = utils.compute_psnr(loss)
    if len(ret) > 1:
      # If there are both coarse and fine predictions, we compute the loss for
      # the coarse prediction (ret[0]) as well.
      rgb_c, unused_disp_c, unused_acc_c = ret[0]
      loss_c = ((rgb_c - batch["pixels"][..., :3])**2).mean()
      psnr_c = utils.compute_psnr(loss_c)
    else:
      loss_c = 0.
      psnr_c = 0.

    weight_variables = copy.deepcopy(variables).unfreeze()
    del weight_variables["params"]["delta_intrinsics"]
    del weight_variables["params"]["delta_pose"]

    def tree_sum_fn(fn):
      return jax.tree_util.tree_reduce(
          lambda x, y: x + fn(y), weight_variables, initializer=0)

    weight_l2 = (
        tree_sum_fn(lambda z: jnp.sum(z**2)) /
        tree_sum_fn(lambda z: jnp.prod(jnp.array(z.shape))))

    stats = utils.Stats(
        loss=loss, psnr=psnr, loss_c=loss_c, psnr_c=psnr_c, weight_l2=weight_l2)

    aux_output = {}
    aux_output["stats"] = stats
    aux_output["extra"] = ret_extra

    return loss + loss_c + FLAGS.weight_decay_mult * weight_l2, aux_output

  (_, aux_output), grad = (
      jax.value_and_grad(loss_fn, has_aux=True)(
          state.optimizer.target))
  stats = aux_output["stats"]
  grad = jax.lax.pmean(grad, axis_name="batch")
  stats = jax.lax.pmean(stats, axis_name="batch")

  # Clip the gradient by value.
  if FLAGS.grad_max_val > 0:
    clip_fn = lambda z: jnp.clip(z, -FLAGS.grad_max_val, FLAGS.grad_max_val)
    grad = jax.tree_util.tree_map(clip_fn, grad)

  # Clip the (possibly value-clipped) gradient by norm.
  if FLAGS.grad_max_norm > 0:
    grad_norm = jnp.sqrt(
        jax.tree_util.tree_reduce(
            lambda x, y: x + jnp.sum(y**2), grad, initializer=0))
    mult = jnp.minimum(1, FLAGS.grad_max_norm / (1e-7 + grad_norm))
    grad = jax.tree_util.tree_map(lambda z: mult * z, grad)

  new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=lr)

  new_state = state.replace(optimizer=new_optimizer)
  return new_state, stats, rng, aux_output


def init_datasets():
  dataset = datasets.get_dataset("train", FLAGS)
  test_dataset = datasets.get_dataset("test", FLAGS)
  return dataset, test_dataset


def main(unused_argv=None, update_flags=True):
  logging.log(INFO, "train.main() started. Flags\n%s",
              pprint.pformat(utils_extra.flags_dict_by_module(FLAGS, "nerf")))

  rng = random.PRNGKey(20200823)
  # Shift the numpy random seed by process_index() to shuffle data loaded by
  # different hosts.
  np.random.seed(20201473 + jax.process_index())

  if update_flags:
    if FLAGS.config is not None:
      utils.update_flags(FLAGS)
    if FLAGS.config_extra_encoded is not None:
      # Overwrites existing args.
      utils_extra.update_extra_flags(FLAGS)
  if FLAGS.batch_size % jax.device_count() != 0:
    raise ValueError("Batch size must be divisible by the number of devices.")
  if FLAGS.train_dir is None:
    raise ValueError("train_dir must be set. None set now.")
  if FLAGS.data_dir is None:
    raise ValueError("data_dir must be set. None set now.")

  dataset, test_dataset = init_datasets()
  # Make dataset objects accessible from notebook.
  utils.save_global_state('dataset', dataset)
  utils.save_global_state('test_dataset', test_dataset)

  rng, key = random.split(rng)
  # get_model sees a sample of training data.
  model, variables = models.get_model(key, dataset.peek(), FLAGS, dataset)
  optimizer = flax.optim.Adam(FLAGS.lr_init).create(variables)
  state = utils.TrainState(optimizer=optimizer)
  del optimizer, variables

  learning_rate_fn = functools.partial(
      utils.learning_rate_decay,
      lr_init=FLAGS.lr_init,
      lr_final=FLAGS.lr_final,
      max_steps=FLAGS.max_steps,
      lr_delay_steps=FLAGS.lr_delay_steps,
      lr_delay_mult=FLAGS.lr_delay_mult)

  train_pstep = jax.pmap(
      functools.partial(train_step, model),
      axis_name="batch",
      in_axes=(0, 0, 0, None, None),
      donate_argnums=(2,))

  def render_fn(step, variables, key_0, key_1, rays, camera_ids, ray_ids):
    alpha = barf_alpha(step)
    randomized = FLAGS.randomized
    if FLAGS.deterministic_rendering:
      randomized = False
    ret = model.apply(variables, key_0, key_1, rays, randomized, alpha,
                      camera_ids, ray_ids, True)
    # `ret` is a tuple ((rgb, disp, acc), ret_extra)
    return jax.lax.all_gather(ret, axis_name="batch")

  render_pfn = jax.pmap(
      render_fn,
      # Only distribute the data input.
      in_axes=(None, None, None, None, 0, 0, 0),
      donate_argnums=(3,),
      axis_name="batch",
  )

  # Compiling to the CPU because it's faster and more accurate.
  ssim_fn = jax.jit(
      functools.partial(utils.compute_ssim, max_val=1.), backend="cpu")

  if not utils.isdir(FLAGS.train_dir):
    utils.makedirs(FLAGS.train_dir)
  state = checkpoints.restore_checkpoint(FLAGS.train_dir, state,
                                         step=FLAGS.restored_step)
  # Resume training a the step of the last checkpoint.
  init_step = state.optimizer.state.step + 1
  state = flax.jax_utils.replicate(state)

  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(FLAGS.train_dir)
    summary_writer.hparams(utils_extra.flags_dict_by_module(FLAGS, "nerf"))

  # Prefetch_buffer_size = 3 x batch_size
  pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)
  n_local_devices = jax.local_device_count()
  rng = rng + jax.process_index()  # Make random seed separate across hosts.
  keys = random.split(rng, n_local_devices)  # For pmapping RNG keys.
  if FLAGS.gc_every > 0:
    gc.disable()  # Disable automatic garbage collection for efficiency.
  stats_trace = []
  reset_timer = True
  first_test_case = None
  render_steps_set = {
      int(item.strip())
      for item in FLAGS.render_steps_list.split(",")
      if item.strip()
  }
  for step, batch in zip(range(init_step, FLAGS.max_steps + 1), pdataset):
    if reset_timer:
      t_loop_start = time.time()
      reset_timer = False
    lr = learning_rate_fn(step)
    state, stats, keys, aux_output = train_pstep(keys, state, batch, lr, step)
    # TODO(daeyun): Add a flag to disable save_global_state calls used in dev.
    utils.save_global_state("state", state, verbose=False)
    utils.save_global_state("extra", aux_output["extra"], verbose=False)
    if jax.process_index() == 0:
      stats_trace.append(stats)
    if FLAGS.gc_every > 0 and step % FLAGS.gc_every == 0:
      gc.collect()

    # Log training summaries. This is put behind a process_index check because
    # in multi-host evaluation, all hosts need to run inference even though we
    # only use host 0 to record results.
    if jax.process_index() == 0:
      if step % FLAGS.print_every == 0:
        alpha = barf_alpha(step)
        summary_writer.scalar("train_loss", stats.loss[0], step)
        summary_writer.scalar("train_psnr", stats.psnr[0], step)
        summary_writer.scalar("train_loss_coarse", stats.loss_c[0], step)
        summary_writer.scalar("train_psnr_coarse", stats.psnr_c[0], step)
        summary_writer.scalar("weight_l2", stats.weight_l2[0], step)
        avg_loss = np.mean(np.concatenate([s.loss for s in stats_trace]))
        avg_psnr = np.mean(np.concatenate([s.psnr for s in stats_trace]))
        stats_trace = []
        summary_writer.scalar("train_avg_loss", avg_loss, step)
        summary_writer.scalar("train_avg_psnr", avg_psnr, step)
        summary_writer.scalar("learning_rate", lr, step)
        steps_per_sec = FLAGS.print_every / (time.time() - t_loop_start)
        reset_timer = True
        rays_per_sec = FLAGS.batch_size * steps_per_sec
        summary_writer.scalar("train_steps_per_sec", steps_per_sec, step)
        summary_writer.scalar("train_rays_per_sec", rays_per_sec, step)
        summary_writer.scalar("train_alpha", alpha, step)
        precision = int(np.ceil(np.log10(FLAGS.max_steps))) + 1
        print(("{:" + "{:d}".format(precision) + "d}").format(step) +
              f"/{FLAGS.max_steps:d}: " + f"i_loss={stats.loss[0]:0.4f}, " +
              f"avg_loss={avg_loss:0.4f}, " +
              f"weight_l2={stats.weight_l2[0]:0.2e}, " + f"lr={lr:0.2e}, " +
              f"{rays_per_sec:0.0f} rays/sec, barf_alpha={alpha:0.2f}, " +
              f"psnr={stats.psnr[0]:0.2f}")
      if step % FLAGS.save_every == 0:
        state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(
            FLAGS.train_dir, state_to_save, int(step), keep=100)
        logging.info("Saved checkpoint at step %d", step)

    # Test-set evaluation.
    if (FLAGS.render_every > 0 and
        step % FLAGS.render_every == 0) or step in render_steps_set:
      # We reuse the same random number generator from the optimization step
      # here on purpose so that the visualization matches what happened in
      # training.
      t_eval_start = time.time()
      eval_variables = jax.device_get(jax.tree_map(lambda x: x[0],
                                                   state)).optimizer.target

      def log_test_case(i_test, test_case, name="test"):
        pred_color, pred_disp, pred_acc = utils.render_image(
            functools.partial(render_pfn, step, eval_variables),
            test_case["rays"],
            test_case["camera_ids"],
            test_case["ray_ids"],
            keys[0],
            True,  # Originally, only true for --dataset=llff.
            chunk=FLAGS.chunk)

        if i_test == 0:
          utils.save_global_state("test_case", test_case)
          utils.save_global_state("pred_color", pred_color)

        # Log eval summaries on host 0.
        if jax.process_index() == 0:
          psnr = utils.compute_psnr(
              ((pred_color - test_case["pixels"])**2).mean())
          ssim = ssim_fn(pred_color, test_case["pixels"])
          alignment_im = np.zeros_like(test_case["pixels"])
          alignment_im[:,:,0] = test_case["pixels"].mean(-1)
          alignment_im[:,:,1] = pred_color.mean(-1)
          eval_time = time.time() - t_eval_start
          num_rays = jnp.prod(jnp.array(
              test_case["rays"].directions.shape[:-1]))
          rays_per_sec = num_rays / eval_time
          summary_writer.scalar(f"{name}_per_sec", rays_per_sec, step)
          print(f"Eval({name}) {step}: {eval_time:0.3f}s., "
                f"{rays_per_sec:0.0f} rays/sec")
          print(f"    PSNR: {psnr:0.4f}, SSIM: {ssim:0.4f}")
          summary_writer.scalar(f"{name}_psnr", psnr, step)
          summary_writer.scalar(f"{name}_ssim", ssim, step)
          summary_writer.image(f"{name}_pred_color", pred_color, step)
          summary_writer.image(f"{name}_pred_disp", pred_disp, step)
          summary_writer.image(f"{name}_pred_acc", pred_acc, step)
          summary_writer.image(f"{name}_target", test_case["pixels"], step)
          summary_writer.image(f"{name}_align", alignment_im, step)

          if FLAGS.train_matplotlib:
            utils_extra.show_pred_gt(pred_color, test_case["pixels"])

      for i_test in test_dataset.eval_indices:
        test_case = next(test_dataset)
        if first_test_case is None:
          first_test_case = test_case
        if i_test != test_case["example_id"]:
          raise RuntimeError("Evaluation is not in expected order.")
        log_test_case(i_test, test_case, f"eval {i_test:03d}")

      print(np.array(state.optimizer.target["params"]["delta_pose"][0]))
      print(np.array(state.optimizer.target["params"]["delta_intrinsics"][0]))

  if FLAGS.max_steps % FLAGS.save_every != 0:
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    checkpoints.save_checkpoint(
        FLAGS.train_dir, state, int(FLAGS.max_steps), keep=100)


if __name__ == "__main__":
  np.set_printoptions(4)
  app.run(main)
