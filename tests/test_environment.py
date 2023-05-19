from stable_baselines3.common.env_checker import check_env

from src.config import EnvironmentConfig
from src.environment import DrivingEnvironment


def test_check_env():
    environment_config = EnvironmentConfig()
    environment = DrivingEnvironment(environment_config)
    check_env(environment)


def test_slight_turn():
    environment_config = EnvironmentConfig()
    environment = DrivingEnvironment(environment_config)

    observation = environment._get_observation()
    print(observation)
    environment.render()

    environment.step([0.1, 0.1])
    observation = environment._get_observation()
    print(observation)
    environment.render()

    environment.step([0.1, 0.0])
    observation = environment._get_observation()
    print(observation)
    environment.render()

    environment.step([0.1, -0.3])
    observation = environment._get_observation()
    print(observation)
    environment.render()

    environment.step([0.1, 0.0])
    observation = environment._get_observation()
    print(observation)
    environment.render()
