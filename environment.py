import math
import numpy
import geopandas
import matplotlib.pyplot as plt

from config import EnvironmentConfig
from geometry import make_rectangle, make_circle


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
            self.car_protection_polygon,
            self.object_polygons
        ) = self._new_environment()
    

    def _get_random_position(self):
        return (
            numpy.random.uniform(0.0, self.config.region_width),
            numpy.random.uniform(0.0, self.config.region_height)
        )

    
    def _new_environment(self):
        car_polygon = make_rectangle(
            self._get_random_position(),
            (self.config.car_width, self.config.car_height),
            angle=numpy.random.uniform(0.0, 2 * math.pi)
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
        
        return car_polygon, car_protection_polygon, object_polygons
    

    def render(self):
        objects_to_render = (
            [self.car_polygon] +
            [self.car_protection_polygon.exterior] +
            self.object_polygons
        )
        colors = ["blue"] + ["yellow"] + ["black"] * len(self.object_polygons)
        axes = geopandas.GeoSeries(objects_to_render).plot(color=colors)

        # TODO: render input rays

        axes.set_xbound(0.0, self.config.region_width)
        axes.set_ybound(0.0, self.config.region_height)

        plt.show()
    

    def get_state(self):
        state = None

        return state


    def perform_action(self):
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
