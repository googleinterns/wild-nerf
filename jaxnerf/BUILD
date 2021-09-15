"BUILD file for jaxnerf."

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

# Files in configs/ directory will be built-in to the MPM package.
# Files in configs_extra/ directory will be hex encoded and passed in as a command line flag.

py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    data = glob([
        "configs/*.yaml",
    ]),
    python_version = "PY3",
    visibility = ["//experimental/users/daeyun/jaxnerf:__pkg__"],
    deps = [
        "//experimental/users/daeyun/jaxnerf/nerf",
        "//experimental/users/daeyun/jaxnerf/nerf:model_utils",
        "//experimental/users/daeyun/jaxnerf/nerf:utils_extra",
        "//learning/brain/research/jax:gpu_support",
        "//learning/brain/research/jax:tpu_support",
        "//learning/deepmind/analysis/tensortree",
        "//learning/deepmind/analysis/tensortree:hlog",
        "//learning/deepmind/xmanager2/client/google",
        "//third_party/google_research/google_research/jaxnerf/nerf",
        "//third_party/py/chex",
        "//third_party/py/flax",
        "//third_party/py/flax/metrics:tensorboard",
        "//third_party/py/flax/training",
        "//third_party/py/jax",
    ],
)

py_binary(
    name = "eval",
    srcs = [
        "eval.py",
    ],
    data = glob([
        "configs/*.yaml",
    ]),
    python_version = "PY3",
    visibility = ["//experimental/users/daeyun/jaxnerf:__pkg__"],
    deps = [
        "//experimental/users/daeyun/jaxnerf/nerf",
        "//experimental/users/daeyun/jaxnerf/nerf:model_utils",
        "//experimental/users/daeyun/jaxnerf/nerf:utils_extra",
        "//learning/brain/research/jax:gpu_support",
        "//learning/brain/research/jax:tpu_support",
        "//learning/deepmind/xmanager2/client/google",
        "//tech/env:envelope_loader",
        "//third_party/google_research/google_research/jaxnerf/nerf",
        "//third_party/py/chex",
        "//third_party/py/flax",
        "//third_party/py/flax/training",
        "//third_party/py/jax",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_hub",
    ],
)
