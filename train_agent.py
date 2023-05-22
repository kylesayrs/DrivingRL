from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.config import AgentConfig, EnvironmentConfig
from src.environment import DrivingEnvironment

def train_agent(agent_config: AgentConfig, environment_config: EnvironmentConfig):
    environment = make_vec_env(
        DrivingEnvironment,
        env_kwargs={"environment_config": environment_config},
        n_envs=agent_config.n_envs
    )

    model = PPO(
        agent_config.policy,
        environment,
        learning_rate=agent_config.learning_rate,
        n_steps=agent_config.n_steps,
        batch_size=agent_config.batch_size,
        n_epochs=agent_config.n_epochs,
        gamma=agent_config.gamma,
        gae_lambda=agent_config.gae_lambda,
        clip_range=agent_config.clip_range,
        verbose=agent_config.verbosity
    )
    model.learn(
        total_timesteps=training_config.total_timesteps,
        progress_bar=training_config.progress_bar
    )
    now_string = str(datetime.now()).replace(" ", "_")
    model.save(f"checkpoints/{now_string}.zip")


if __name__ == "__main__":
    training_config = AgentConfig()
    environment_config = EnvironmentConfig(
        num_rays=8,
        goal_radius=3,
        object_max_num=20
    )

    train_agent(training_config, environment_config)
