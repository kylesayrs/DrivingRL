from typing import Tuple

import math
import numpy
from shapely import affinity, Polygon, Point, LineString, LinearRing

from src.utils import lerp


def make_rectangle(
    position: Tuple[float, float],
    size: Tuple[float, float],
    angle: float = 0.0
):
    corner_positions = numpy.array([  # facing right
        [     size[1],      size[0]],
        [     size[1], -1 * size[0]],
        [-1 * size[1], -1 * size[0]],
        [-1 * size[1],      size[0]],
    ]) / 2
    rectangle = Polygon(corner_positions)
    rectangle = affinity.rotate(rectangle, angle, "center", use_radians=True)
    rectangle = affinity.translate(rectangle, *position)

    return rectangle


def make_circle(
    position: Tuple[float, float],
    radius: Tuple[float, float]
):
    return Point(position).buffer(radius)


def make_box(
    size: Tuple[float, float],
    width: float
):
    return LinearRing([
        [-1 * width, -1 * width],
        [size[0] + width, -1 * width],
        [size[0] + width, size[1] + width],
        [-1 * width, size[1] + width],
    ]).buffer(width)


def make_ray_lines(
    start: Tuple[float, float],
    angle: float,
    length: float,
    sigmoid_density: float,
    num_rays: int,
):
    ray_angles = [
        (ray_i / num_rays) * (2 * math.pi)
        for ray_i in range(num_rays)
    ]

    # change density
    if sigmoid_density > 0.0:
        ray_angles = [
            lerp(
                math.atan(sigmoid_density * (ray_angle - math.pi)),
                math.atan(sigmoid_density * (0 - math.pi)),
                math.atan(sigmoid_density * (2 * math.pi - math.pi)),
                0,
                2 * math.pi,
                )
            for ray_angle in ray_angles
        ]

    return [
        make_line(start, length, ray_angle + angle)
        for ray_angle in ray_angles
    ]


def make_line(
    start: Tuple[float, float],
    length: float,
    angle: float,
):
    end = numpy.array([length * math.cos(angle), length * math.sin(angle)]) + start

    return LineString([start, end])


def affine_polygon(
    polygon: Polygon,
    displacement: Tuple[float, float],
    angular_displacement: float
):
    polygon = affinity.rotate(polygon, angular_displacement, "center", use_radians=True)
    polygon = affinity.translate(polygon, *displacement)

    return polygon
