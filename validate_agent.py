import sys
import argparse

from stable_baselines3 import PPO

from src.config import AgentConfig, EnvironmentConfig
from src.environment import DrivingEnvironment


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_path", help="Path to checkpoint zip")

def validate_agent(
    checkpoint_path: str,
    environment_config: EnvironmentConfig
):
    model = PPO.load(checkpoint_path)

    environment = DrivingEnvironment(environment_config)
    observation, reset_info = environment.reset()
    for i in range(environment_config.max_steps):
        action, _states = model.predict(observation)
        observation, rewards, dones, truncated, info = environment.step(action)
        environment.render()
        if dones:
            break

if __name__ == "__main__":
    args = parser.parse_args()

    environment_config = EnvironmentConfig()

    validate_agent(
        args.checkpoint_path,
        environment_config
    )
