"""3D visualization using plotly."""

from typing import List

import numpy as np
import plotly.graph_objects as go


def line_segments_from_cameras(
    mat34_list: List[np.ndarray], scale=0.1, alpha=1.0):
  """Convert camera parameters to drawable objects.

  Args:
    mat34_list: A list of 3 by 4 camera matrices.
    scale: How big the lines should be in the plot.

  Returns:
    A list of plotly graph objects.
  """
  ret = []
  for mat34 in mat34_list:
    colors = [
        f'rgba(255,0,0,{alpha:.2f})',
        f'rgba(0,255,0,{alpha:.2f})',
        f'rgba(0,0,255,{alpha:.2f})'
    ]
    text = ['X', 'Y', 'Z']
    for i in range(3):
      line = go.Scatter3d(
          x=[mat34[0, 3], mat34[0, 3] + mat34[0, i] * scale],
          y=[mat34[1, 3], mat34[1, 3] + mat34[1, i] * scale],
          z=[mat34[2, 3], mat34[2, 3] + mat34[2, i] * scale],
          mode='lines',
          text=text[i],
          marker=dict(size=4,),
          line=dict(color=colors[i], width=2))
      ret.append(line)
  return ret


def plot_llff_cameras(pose, cube_size=None):
  """Visualize cameras in LLFF format.

  See:
  https://github.com/Fyusion/LLFF#using-your-own-poses-without-running-colmap

  Args:
    pose: An array of shape (N, 17).
    cube_size: Size of the 3D axes to draw. If None, resized to fit the data.
  """
  # Cam-to-world.
  cameras = [pose[i, :-2].reshape([3, 5])[:, :4] for i in range(pose.shape[0])]
  plot_cameras(cameras, cube_size=cube_size)


def plot_cameras(cameras, cameras2=None, cube_size=None):
  # Cam-to-world.
  lines = line_segments_from_cameras(cameras)
  if cameras2 is not None:
    lines.extend(line_segments_from_cameras(cameras2, alpha=0.2))
  fig = go.Figure(data=lines)

  if cube_size is None:
    scene = dict(xaxis=dict(), yaxis=dict(), zaxis=dict(), aspectmode='data')
  else:
    scene = dict(
        xaxis=dict(
            nticks=4,
            range=[-cube_size, cube_size],
        ),
        yaxis=dict(
            nticks=4,
            range=[-cube_size, cube_size],
        ),
        zaxis=dict(
            nticks=4,
            range=[-cube_size, cube_size],
        ),
        aspectratio=dict(x=1, y=1, z=1),
    )

  fig.update_layout(
      autosize=True,
      height=900,
      scene_camera=dict(up=dict(x=0, y=0, z=1)),
      scene=scene,
  )
  fig.show()
