import math
import numpy
import geopandas
import matplotlib.pyplot as plt
from shapely import MultiPolygon

from config import EnvironmentConfig
from geometry import make_rectangle, make_circle, make_box, make_ray_lines, affine_polygon


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
            self.car_velocity,
            self.car_angle_velocity,
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
            (self.config.region_width / 2, self.config.region_height / 2),
            (self.config.car_width, self.config.car_height),
            head_angle=car_angle
        )

        car_velocity = numpy.array([0.0, 0.0])

        car_angle_velocity = 0.0

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
                    head_angle=numpy.random.uniform(0.0, 2 * math.pi)
                )

            if object_type == 1:
                object_polygon = make_circle(
                    self._get_random_position(),
                    numpy.random.uniform(self.config.object_min_size, self.config.object_max_size)
                )

            if not object_polygon.intersects(car_protection_polygon):
                object_polygons.append(object_polygon)

        boundary_object = make_box(
            (self.config.region_width, self.config.region_height),
            self.config.boundary_width
        )
        object_polygons.append(boundary_object)

        goal_polygon = make_circle(self._get_random_position(), self.config.goal_radius)
        
        return (
            car_polygon,
            car_angle,
            car_velocity,
            car_angle_velocity,
            car_protection_polygon,
            object_polygons,
            goal_polygon
        )
    

    def render(self):
        ray_lines = make_ray_lines(
            self.car_polygon.centroid.coords[0],
            self.car_angle,
            self.config.ray_length,
            self.config.num_rays
        )

        objects_to_render = (
            [self.car_polygon] +
            [self.car_protection_polygon.exterior] +
            [self.goal_polygon] +
            ray_lines + 
            self.object_polygons
        )
        colors = (
            ["blue"] +
            ["yellow"] +
            ["green"] +
            ["orange"] +
            ["red"] * (len(ray_lines) - 1) +
            ["black"] * len(self.object_polygons)
        )
        axes = geopandas.GeoSeries(objects_to_render).plot(color=colors)

        axes.set_xbound(
            -1 * self.config.boundary_width,
            self.config.region_width + self.config.boundary_width
        )
        axes.set_ybound(
            -1 * self.config.boundary_width,
            self.config.region_height + self.config.boundary_width
        )

        plt.show()
    

    def get_state(self):
        car_center = numpy.array(self.car_polygon.centroid.coords[0])
        goal_center = numpy.array(self.goal_polygon.centroid.coords[0])
        ray_lines = make_ray_lines(
            car_center,
            self.car_angle,
            self.config.ray_length,
            self.config.num_rays
        )

        ray_distances = []
        for ray_line in ray_lines:
            intersections = []

            for object_polygon in self.object_polygons:
                object_intersection = ray_line.intersection(object_polygon.exterior)

                if object_intersection.is_empty:
                    continue

                if (object_intersection.geom_type == "MultiPoint"):
                    intersections += [numpy.array(intersection.coords[0]) for intersection in object_intersection.geoms]

                else:
                    intersections.append(numpy.array(object_intersection.coords[0]))

            # note: rays always begin at car centroid
            intersection_distances = [
                numpy.linalg.norm(intersection - car_center)
                for intersection in intersections
            ]

            if len(intersection_distances) > 0:
                ray_distances.append(min(intersection_distances))
            else:
                ray_distances.append(self.config.ray_length * 2)

        goal_displacement = goal_center - car_center
        goal_distance = numpy.linalg.norm(goal_displacement)
        goal_angle_right = math.atan2(goal_displacement[1], goal_displacement[0])
        goal_angle_head = goal_angle_right - math.pi / 2
        
        state = ray_distances + [goal_angle_head, goal_distance]
        state = numpy.array(state)

        return state


    def perform_action(self, pos_acc: float, angle_acc: float):
        self.car_velocity += pos_acc
        self.car_angle_velocity += angle_acc

        self.car_polygon = affine_polygon(
            self.car_polygon,
            self.car_velocity,
            self.car_angle_velocity
        )
        self.car_angle += self.car_angle_velocity
        
        for object_polygon in self.object_polygons:
            if self.car_polygon.intersects(object_polygon):
                return self.config.collision_reward, True, None

        if self.car_polygon.intersects(self.goal_polygon):
            return self.config.goal_reward, True, None
        
        return 0.0, False, self.get_state()
    

    def show_animation():
        raise NotImplementedError


if __name__ == "__main__":
    environment_config = EnvironmentConfig()
    environment = DrivingEnvironment(environment_config)
    state = environment.get_state()
    print(state)
    environment.render()

    environment.perform_action(1.0, 0.0)
    state = environment.get_state()
    print(state)
    environment.render()

    environment.perform_action(1.0, 0.0)
    state = environment.get_state()
    print(state)
    environment.render()
