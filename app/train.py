import torch
import numpy as np
from model import AutoEncoder

TIMESTEPS = 3
HIDDEN_DIM = 64

class MyDataset(torch.utils.data.Dataset):
  """Some Information about MyDataset"""
  def __init__(self, timeseries: np.array, timesteps: int):
    super(MyDataset, self).__init__()
    x, y = self.temporalize(
      X = timeseries,
      y = np.zeros(len(timeseries)),
      lookback=timesteps
    )
    x = np.array(x)
    x = x.reshape(x.shape[0], 1, timesteps)
    self.X = torch.Tensor(x)

  def __getitem__(self, index):
    return self.X[index]

  def __len__(self):
    return self.X.shape[0]

  def temporalize(self, X, y, lookback):
    output_X = []
    output_y = []
    for i in range(len(X)-lookback-1):
      t = []
      for j in range(1,lookback+1):
          # Gather past records upto the lookback period
          t.append(X[[(i+j+1)], :])
      output_X.append(t)
      output_y.append(y[i+lookback+1])
    return output_X, output_y

timeseries = np.array([
  [0.1**3, 0.2**3, 0.3**3, 0.4**3, 0.5**3, 0.6**3, 0.7**3, 0.8**3, 0.9**3, 1.0**3, 1.1**3, 1.2**3, 1.3**3, 1.4**3, 1.5**3, 1.6**3, 1.7**3, 1.8**3, 1.9**3, 2.0**3],
]).transpose()

model = AutoEncoder(TIMESTEPS, HIDDEN_DIM, 2)

dataset = MyDataset(timeseries, TIMESTEPS)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

criterion = torch.nn.MSELoss()

device = torch.device('cpu')

# loop over the dataset multiple timese
for epoch in range(500):
  running_loss = 0.0
  for i, data in enumerate(dataloader):
    inputs = data
    inputs = inputs.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    # print(outputs[0][0][0], inputs[0][0][0])
    loss = criterion(outputs, inputs)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
  
  print("Loss: {}".format(running_loss))