import numpy as np

from variable import Variable


class Temperature(Variable):
  def __init__(self, ambient, variance=1) -> None:
    super().__init__(ambient, variance)
