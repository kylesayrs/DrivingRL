def lerp(x: float, x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min
