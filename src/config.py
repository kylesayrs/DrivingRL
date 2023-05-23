from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    n_envs: int = Field(default=2)
    total_timesteps: float = Field(default=500_000)

    policy: str = Field(default="MultiInputPolicy")
    learning_rate: float = Field(default=0.001)
    n_steps: float = Field(default=1024, description="The number of steps to run for each environment per update")
    batch_size: int = Field(default=64)
    n_epochs: int = Field(default=15)

    gamma: float = Field(default=0.999)
    gae_lambda: float = Field(default=0.95)
    clip_range: float = Field(default=0.2)

    num_validation_steps: int = Field(default=300)
    progress_bar: bool = Field(default=False)
    verbosity: int = Field(default=1)
    device: str = Field(default="cpu")


class EnvironmentConfig(BaseModel):
    region_width: float = Field(default=50.0)
    region_height: float = Field(default=50.0)
    boundary_width: float = Field(default=1.0)
    
    car_width: float = Field(default=1.0)
    car_height: float = Field(default=3.0)
    car_protection_buffer: float = Field(default=2.0)

    car_max_velocity: float = Field(default=10.0)
    car_max_angle_velocity: float = Field(default=0.5)

    car_min_acc: float = Field(default=-0.5)
    car_max_acc: float = Field(default=1.0)
    car_max_angle_acc: float = Field(default=0.05)

    object_min_num: int = Field(default=5)
    object_max_num: int = Field(default=10)
    object_min_size: float = Field(default=0.5)
    object_max_size: float = Field(default=3.0)

    num_rays: int = Field(default=6)
    ray_length: float = Field(default=15.0)
    goal_radius: float = Field(default=3.0)

    step_reward: float = Field(default=0.0)
    collision_reward: float = Field(default=-1.0)
    goal_reward: float = Field(default=1.0)
    max_steps: int = Field(default=500)
