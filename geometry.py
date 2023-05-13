from typing import Tuple

import math
import numpy
from shapely import Polygon, Point, LineString, MultiPolygon


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


def make_ray_lines(
    start: Tuple[float, float],
    angle_offset: float,
    length: float,
    num_rays: int,
):
    print([
        make_line(start, length, (ray_i / num_rays) * (2 * math.pi) - angle_offset)
        for ray_i in range(num_rays)
    ])
    return [
        make_line(start, length, (ray_i / num_rays) * (2 * math.pi) - angle_offset)
        for ray_i in range(num_rays)
    ]


def make_line(
    start: Tuple[float, float],
    length: float,
    angle: float,
):
    end = numpy.array([length * math.cos(angle), length * math.sin(angle)]) + start

    return LineString([start, end])