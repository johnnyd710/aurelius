import numpy as np

class Variable:
  def __init__(self, ambient, variance) -> None:
    self.ambient = ambient
    self.variance = variance
    self.current_value = ambient

  def step(self) -> int:
    """
    each timestep apply noise from the normal distribution
    """
    self.current_value = np.random.normal(self.ambient, self.variance)
    return self.current_value

  def __str__(self) -> str:
    return str(self.current_value)

  def __add__(self, other):
    if (type(other) is float):
      return self.current_value + other
    return self.current_value + other.current_value

  def __sub__(self, other):
    if (type(other) is float):
      return self.current_value - other
    return self.current_value - other.current_value

  def __truediv__(self, other):
    if (type(other) is float):
      return self.current_value / other
    return self.current_value / other.current_value

  def __mul__(self, other):
    if (type(other) is float):
      return self.current_value * other
    return self.current_value * other.current_value
