import numpy as np
from temperature import Temperature
from flow import Flow

class HeatExchanger:
  def __init__(self,
    heatCapacity,
    area,
    HSInTemp = [],
    HSOutTemp = [],
    CSInTemp = [],
    CSOutTemp = [],
    HSFlow = [],
    ) -> None:
    self.C = heatCapacity # kK/kg.K
    self.area = area
    # input validation
    assert(all([len(x) == 2 for x in [HSInTemp, HSOutTemp, CSInTemp, CSOutTemp, HSFlow]]))
    self.HSInTemp = Temperature(HSInTemp[0], HSInTemp[1])
    self.HSOutTemp = Temperature(HSOutTemp[0], HSOutTemp[1])
    self.CSInTemp = Temperature(CSInTemp[0], CSInTemp[1])
    self.CSOutTemp = Temperature(CSOutTemp[0], CSOutTemp[1])
    self.HSFlow = Flow(HSFlow[0], HSFlow[1])

  def logMeanTemp(self):
    lmt = (self.HSInTemp - self.CSOutTemp) - (self.HSOutTemp - self.CSInTemp)
    lmt = lmt / np.log((self.HSInTemp - self.CSOutTemp) / (self.HSOutTemp - self.CSInTemp))
    return lmt

  def heatFlux(self):
    return self.HSFlow * self.C * (self.HSInTemp - self.HSOutTemp)

  def U(self):
    return self.heatFlux() / (self.area * self.logMeanTemp())

  def step(self) -> int:
    self.HSInTemp.step()
    self.HSOutTemp.step()
    self.CSInTemp.step()
    self.CSOutTemp.step()
    self.HSFlow.step()

  def __str__(self) -> str:
    return f"""
    Hot Side In Temperature: {self.HSInTemp} \n
    Hot Side Out Temperature: {self.HSOutTemp} \n
    Cold Side Out Temperature: {self.CSOutTemp} \n
    Cold Side In Temperature: {self.CSInTemp} \n
    Flow: {self.HSFlow} \n
    Area: {self.area} \n
    Heat Capacity: {self.C} \n
    Heat Flux: {self.heatFlux()} \n
    LMT: {self.logMeanTemp()} \n
    U: {self.U()} \n
    """


if __name__ == "__main__":
  heatExchanger = HeatExchanger(
    1.8,
    29.02,
    [85.0, 1.5],
    [60.0, 1],
    [18.0, 0.5],
    [44.27, 1],
    [24906.0, 100],
  )
  print(heatExchanger)
  heatExchanger.step()
  print(heatExchanger)
  heatExchanger.step()
  print(heatExchanger)