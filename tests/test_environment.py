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

    environment.step([0.1, 0.1])
    environment.step([0.1, 0.0])
    environment.step([0.1, -0.3])
    environment.step([0.1, 0.0])
