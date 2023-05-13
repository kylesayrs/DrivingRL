import math
import numpy
import geopandas
import matplotlib.pyplot as plt
from shapely import MultiPolygon

from config import EnvironmentConfig
from geometry import make_rectangle, make_circle, make_ray_lines


def random_position(config: EnvironmentConfig):
    return (
        numpy.random.uniform(0.0, config.region_width),
        numpy.random.uniform(0.0, config.region_height)
    )


class DrivingEnvironment:
    def __init__(self, environment_config: EnvironmentConfig, device: str = "cpu") -> None:
        self.config = environment_config
        self.device = device

        (
            self.car_polygon,
            self.car_angle,
            self.car_protection_polygon,
            self.object_polygons,
            self.goal_polygon,
        ) = self._new_environment()
    

    def _get_random_position(self):
        return (
            numpy.random.uniform(0.0, self.config.region_width),
            numpy.random.uniform(0.0, self.config.region_height)
        )

    
    def _new_environment(self):
        car_angle = numpy.random.uniform(0.0, 2 * math.pi)

        car_polygon = make_rectangle(
            self._get_random_position(),
            (self.config.car_width, self.config.car_height),
            angle=car_angle
        )

        car_protection_polygon = car_polygon.buffer(self.config.car_protection_buffer)

        num_object_tries = numpy.random.randint(
            self.config.object_min_num,
            self.config.object_max_num
        )
        object_polygons = []
        for _ in range(num_object_tries):
            object_type = numpy.random.randint(0, 2)

            if object_type == 0:
                object_polygon = make_rectangle(
                    self._get_random_position(),
                    (
                        numpy.random.uniform(self.config.object_min_size, self.config.object_max_size),
                        numpy.random.uniform(self.config.object_min_size, self.config.object_max_size)
                    ),
                    angle=numpy.random.uniform(0.0, 2 * math.pi)
                )

            if object_type == 1:
                object_polygon = make_circle(
                    self._get_random_position(),
                    numpy.random.uniform(self.config.object_min_size, self.config.object_max_size)
                )

            if not object_polygon.intersects(car_protection_polygon):
                object_polygons.append(object_polygon)

        goal_polygon = make_circle(self._get_random_position(), self.config.goal_radius)
        
        return car_polygon, car_angle, car_protection_polygon, object_polygons, goal_polygon
    

    def render(self):
        ray_lines = make_ray_lines(
            numpy.array(self.car_polygon.centroid.coords)[0],  # TODO: cringe
            self.car_angle,
            self.config.ray_length,
            self.config.num_rays
        )

        objects_to_render = (
            [self.car_polygon] +
            [self.car_protection_polygon.exterior] +
            ray_lines + 
            self.object_polygons
        )
        colors = (
            ["blue"] +
            ["yellow"] +
            ["red"] * len(ray_lines) +
            ["black"] * len(self.object_polygons)
        )
        axes = geopandas.GeoSeries(objects_to_render).plot(color=colors)

        axes.set_xbound(0.0, self.config.region_width)
        axes.set_ybound(0.0, self.config.region_height)

        plt.show()
    

    def get_state(self):
        ray_lines = make_ray_lines(
            numpy.array(self.car_polygon.centroid.coords)[0],  # TODO: cringe
            self.car_angle,
            self.config.ray_length,
            self.config.num_rays
        )

        ray_distances = []
        for ray_line in self.ray_lines:
            intersections = []

            for object_polygon in self.object_polygons:
                intersections += ray_line.intersection(object_polygon.exterior)

            # note: rays always begin at car centroid
            intersection_distances = [
                numpy.linalg.norm(intersection - self.car_polygon.centroid.coords, p=2)
                for intersection in intersections
            ]

            first_intersection_index = numpy.argmin(intersection_distances)

            if first_intersection_index != -1:
                ray_distances.append(intersection_distances[first_intersection_index])

            else:
                ray_distances.append(self.config.ray_length * 2)

        goal_distance = numpy.linalg.norm(
            self.car_polygon.centroid.coords - self.goal_polygon.centroid.coords
        )
        goal_car_theta = self.car_polygon.centroid.coords @ self.goal_polygon.centroid.coords - self.car_angle
        goal_angle_cos = math.cos(goal_car_theta)
        goal_angle_sin = math.sin(goal_car_theta)
        
        state = ray_distances + [goal_distance, goal_angle_cos, goal_angle_sin]
        state = numpy.array(state)

        return state


    def perform_action(self, action):
        reward = None
        next_state = None
        is_finished = None
        
        return reward, next_state, is_finished
    

    def show_animation():
        raise NotImplementedError


if __name__ == "__main__":
    environment_config = EnvironmentConfig()
    environment = DrivingEnvironment(environment_config)
    environment.render()
