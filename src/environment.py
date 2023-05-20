
import math
import numpy
import geopandas
import matplotlib.pyplot as plt

from gym import Env, spaces

from src.config import EnvironmentConfig
from src.geometry import make_rectangle, make_circle, make_box, make_ray_lines, affine_polygon
from src.utils import lerp


class DrivingEnvironment(Env):
    def __init__(self, environment_config: EnvironmentConfig, device: str = "cpu"):
        super().__init__()
        
        self.config = environment_config
        self.device = device

        self.reset()

        self.action_space = spaces.Box(-1, 1, (2,))

        max_distance = math.sqrt(self.config.region_width ** 2 + self.config.region_height ** 2)
        self.observation_space = spaces.Dict(
            spaces={
                "car_velocity": spaces.Box(-1 * self.config.car_max_velocity, self.config.car_max_velocity, (1,)),
                "car_angle": spaces.Box(0.0, 2 * math.pi, (1,)),
                "car_angle_velocity": spaces.Box(-1 * self.config.car_max_angle_velocity, self.config.car_max_angle_velocity, (1,)),
                "visual": spaces.Box(0.0, self.config.ray_length, (self.config.num_rays,)),
                #"goal_angle": spaces.Box(0.0, 2 * math.pi, (1,)),
                "goal_angle": spaces.Box(0.0, 2 * math.pi, (2,)),
                "goal_distance": spaces.Box(0.0, max_distance, (1,)),
            }
        )


    def _get_random_position(self):
        return (
            numpy.random.uniform(0.0, self.config.region_width),
            numpy.random.uniform(0.0, self.config.region_height)
        )

    
    def reset(self):
        self.car_angle = numpy.random.uniform(0.0, 2 * math.pi)

        self.car_polygon = make_rectangle(
            (self.config.region_width / 2, self.config.region_height / 2),
            (self.config.car_width, self.config.car_height),
            angle=self.car_angle
        )

        self.car_velocity = numpy.array([0.0, 0.0])

        self.car_angle_velocity = 0.0

        self.car_protection_polygon = self.car_polygon.buffer(self.config.car_protection_buffer)

        num_object_tries = numpy.random.randint(
            self.config.object_min_num,
            self.config.object_max_num
        ) if  self.config.object_max_num > 0 else 0
        self.object_polygons = []
        for _ in range(num_object_tries):
            object_type = numpy.random.randint(0, 2)

            if object_type == 0:
                object_polygon = make_rectangle(
                    self._get_random_position(),
                    (
                        numpy.random.uniform(
                            self.config.object_min_size,
                            self.config.object_max_size
                        ),
                        numpy.random.uniform(
                            self.config.object_min_size,
                            self.config.object_max_size
                        )
                    ),
                    angle=numpy.random.uniform(0.0, 2 * math.pi)
                )

            if object_type == 1:
                object_polygon = make_circle(
                    self._get_random_position(),
                    numpy.random.uniform(
                        self.config.object_min_size,
                        self.config.object_max_size
                    )
                )

            if not object_polygon.intersects(self.car_protection_polygon):
                self.object_polygons.append(object_polygon)

        boundary_object = make_box(
            (self.config.region_width, self.config.region_height),
            self.config.boundary_width
        )
        self.object_polygons.append(boundary_object)

        self.goal_polygon = make_circle(
            self._get_random_position(),
            self.config.goal_radius
        )

        return self._get_observation()
    

    def render(self, mode="plot"):
        if mode == "console":
            raise NotImplementedError()
        
        elif mode == "plot":
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

        else:
            raise NotImplementedError()
    

    def _get_observation(self):
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
                    intersections += [
                        numpy.array(intersection.coords[0])
                        for intersection in object_intersection.geoms
                    ]

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
                ray_distances.append(self.config.ray_length)

        goal_displacement = goal_center - car_center
        goal_distance = numpy.linalg.norm(goal_displacement)
        goal_global_angle = math.atan2(goal_displacement[1], goal_displacement[0])
        goal_angle = (goal_global_angle - self.car_angle)
        goal_angle_cos = math.cos(goal_angle) % (2 * math.pi)
        goal_angle_sin = math.sin(goal_angle) % (2 * math.pi)

        return {
            "car_velocity": numpy.array([numpy.linalg.norm(self.car_velocity)]),
            "car_angle": numpy.array([self.car_angle]),
            "car_angle_velocity": numpy.array([self.car_angle_velocity]),
            "visual": numpy.array(ray_distances, dtype=numpy.float32),
            "goal_angle": numpy.array([goal_angle_cos, goal_angle_sin], dtype=numpy.float32),
            #"goal_angle": numpy.array([goal_angle], dtype=numpy.float32),
            "goal_distance": numpy.array([goal_distance], dtype=numpy.float32)
        }
    

    def _move_car(self, action: numpy.ndarray):
        forward_acc = lerp(action[0], -1, 1, self.config.car_min_acc, self.config.car_max_acc)
        angle_acc = lerp(action[1], -1, 1, -1 * self.config.car_max_angle_acc, self.config.car_max_angle_acc)

        # change angle
        self.car_angle_velocity += angle_acc
        self.car_angle_velocity = numpy.clip(
            self.car_angle_velocity,
            -1 * self.config.car_max_angle_velocity,
            self.config.car_max_angle_velocity
        )
        self.car_angle += self.car_angle_velocity
        self.car_angle %= 2 * math.pi

        # change position
        self.car_velocity += [
            math.cos(self.car_angle) * forward_acc,
            math.sin(self.car_angle) * forward_acc
        ]
        self.car_velocity = numpy.clip(
            self.car_velocity,
            -1 * self.config.car_max_velocity,
            self.config.car_max_velocity
        )
        self.car_polygon = affine_polygon(
            self.car_polygon,
            self.car_velocity,
            self.car_angle_velocity
        )


    def _car_is_out_of_bounds(self):
        return (
            self.car_polygon.centroid.coords[0][0] < 0.0 or
            self.car_polygon.centroid.coords[0][1] < 0.0 or
            self.car_polygon.centroid.coords[0][0] > self.config.region_width or
            self.car_polygon.centroid.coords[0][1] > self.config.region_height
        )


    def step(self, action: numpy.ndarray):
        """
        :param action[0]: car forward acceleration
        :param action[1]: car angle acceleration

        :return observation: current state
        :return reward: reward for performing this action
        :return done: True if the episode is finished
        :return info: optional information
        """
        self._move_car(action)

        if self._car_is_out_of_bounds():
            return self._get_observation(), self.config.collision_reward, True, {}
        
        for object_polygon in self.object_polygons:
            if self.car_polygon.intersects(object_polygon):
                return self._get_observation(), self.config.collision_reward, True, {}

        if self.car_polygon.intersects(self.goal_polygon):
            return self._get_observation(), self.config.goal_reward, True, {}
        
        return self._get_observation(), self.config.step_reward, False, {}
