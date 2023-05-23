import sys

from stable_baselines3 import PPO

from src.config import AgentConfig, EnvironmentConfig
from src.environment import DrivingEnvironment


def validate_agent(
    checkpoint_path: str,
    agent_config: AgentConfig,
    environment_config: EnvironmentConfig
):
    model = PPO.load(checkpoint_path)

    environment = DrivingEnvironment(environment_config)
    observation = environment.reset()
    print(observation)
    for i in range(agent_config.num_validation_steps):
        action, _states = model.predict(observation)
        print(action)
        observation, rewards, dones, info = environment.step(action)
        environment.render()
        if dones:
            break

if __name__ == "__main__":
    checkpoint_path = sys.argv[1]

    training_config = AgentConfig()
    environment_config = EnvironmentConfig()

    validate_agent(
        checkpoint_path,
        training_config,
        environment_config
    )
