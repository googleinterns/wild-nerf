"""BUILD file for nerf library."""

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "nerf",
    srcs = [
        "datasets.py",
        "models.py",
        "utils.py",
    ],
    deps = [
        ":model_utils",
        ":transforms",
        ":utils_extra",
        "//learning/brain/research/jax:gpu_support",
        "//pyglib:gfile",
        "//pyglib/contrib/gpathlib:gpath",
        "//third_party/py/PIL:pil",
        "//third_party/py/chex",
        "//third_party/py/cvx2",
        "//third_party/py/flax",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/yaml",
    ],
)

py_library(
    name = "model_utils",
    srcs = [
        "model_utils.py",
    ],
    deps = [
        "//learning/brain/research/jax:gpu_support",
        "//third_party/py/chex",
        "//third_party/py/flax",
        "//third_party/py/jax",
    ],
)

py_test(
    name = "model_utils_test",
    srcs = [
        "model_utils_test.py",
    ],
    deps = [
        ":model_utils",
        ":nerf",
        "//testing/pybase",
    ],
)

py_library(
    name = "utils_extra",
    srcs = [
        "utils_extra.py",
    ],
    deps = [
        "//pyglib:gfile",
        "//third_party/py/PIL:pil",
        "//third_party/py/chex",
        "//third_party/py/cvx2",
        "//third_party/py/flax",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/yaml",
    ],
)

py_library(
    name = "transforms",
    srcs = [
        "transforms.py",
    ],
    deps = [
        "//third_party/py/jax",
    ],
)

py_test(
    name = "transforms_test",
    srcs = [
        "transforms_test.py",
    ],
    deps = [
        ":transforms",
        "//testing/pybase",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
    ],
)
