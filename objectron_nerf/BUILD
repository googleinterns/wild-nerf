py_library(
    name = "camera_utils",
    srcs = [
        "camera_utils.py",
    ],
    deps = [
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "io_utils",
    srcs = [
        "io_utils.py",
    ],
    deps = [
        "//third_party/py/cvx2",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "objectron_utils",
    srcs = [
        "objectron_utils.py",
    ],
    deps = [
        "//pyglib/contrib/gpathlib:gpath",
        "//third_party/mediapipe/modules/objectron/calculators:a_r_capture_metadata_py_pb2",
        "//third_party/mediapipe/modules/objectron/calculators:annotation_py_pb2",
        "//third_party/py/cvx2",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "video_utils",
    srcs = [
        "video_utils.py",
    ],
    deps = [
        "//experimental/users/daeyun/objectron_nerf:camera_utils",
        "//experimental/users/daeyun/objectron_nerf:io_utils",
        "//research/xeno/nnets/common:files",
        "//third_party/py/absl/logging",
        "//third_party/py/cvx2",
        "//third_party/py/numpy",
    ],
)
