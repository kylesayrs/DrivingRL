from pydantic import BaseModel, Field


class EnvironmentConfig(BaseModel):
    region_width: float = Field(default=20.0)
    region_height: float = Field(default=20.0)
    boundary_width: float = Field(default=1.0)
    
    car_width: float = Field(default=1.0)
    car_height: float = Field(default=3.0)
    car_protection_buffer: float = Field(default=2.0)

    car_min_acc: float = Field(default=-3.0)
    car_max_acc: float = Field(default=3.0)
    car_min_angle_acc: float = Field(default=-3.0)
    car_max_angle_acc: float = Field(default=3.0)

    object_min_num: int = Field(default=1)
    object_max_num: int = Field(default=8)
    object_min_size: float = Field(default=0.5)
    object_max_size: float = Field(default=3.0)

    num_rays: int = Field(default=12)
    ray_length: float = Field(default=10.0)
    goal_radius: float = Field(default=1.0)

    collision_reward: float = Field(default=0.0)
    goal_reward: float = Field(default=1.0)
