from src.config import AgentConfig, EnvironmentConfig
from src.environment import DrivingEnvironment

if __name__ == "__main__":
    agent_config = AgentConfig()
    environment_config = EnvironmentConfig()
    environment = DrivingEnvironment(environment_config)

    while True:
        try:
            input_string = input("[acc_speed, acc_angle]: ")
        except EOFError:
            break

        acc_speed, acc_angle = input_string.split(" ")
        acc_speed = float(acc_speed)
        acc_angle = float(acc_angle)

        environment.step([acc_speed, acc_angle])
        environment.render(mode="plot")
