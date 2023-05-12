from typing import Tuple

import math
import numpy
from shapely import Polygon, Point


def get_rotation_matrix(angle: float):
    return numpy.array([
        [math.cos(angle), math.sin(angle)],
        [-1 * math.sin(angle), math.cos(angle)],
    ])


def make_rectangle(
    position: Tuple[float, float],
    size: Tuple[float, float],
    angle: float = 0.0
):
    corner_positions = numpy.array([
        [-1 * size[0] / 2, size[1] / 2],
        [size[0] / 2, size[1] / 2],
        [size[0] / 2, -1 * size[1] / 2],
        [-1 * size[0] / 2, -1 * size[1] / 2],
    ]).T
    rotation_matrix = get_rotation_matrix(angle)

    shell_points = (rotation_matrix @ corner_positions).T + position
    return Polygon(shell_points)


def make_circle(
    position: Tuple[float, float],
    radius: Tuple[float, float]
):
    return Point(position).buffer(radius)
