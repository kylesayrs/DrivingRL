from src.config import EnvironmentConfig
from src.environment import DrivingEnvironment

if __name__ == "__main__":
    environment_config = EnvironmentConfig(
        goal_radius=100,
        object_max_num=0
    )
    environment = DrivingEnvironment(environment_config)

    print(environment.step([1, 0.1]))
    environment.render()
    print(environment.step([1, 0.0]))
    environment.render()
    print(environment.step([1, -0.3]))
    environment.render()
    print(environment.step([1, 0.0]))
    environment.render()
