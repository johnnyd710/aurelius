# third party
import torch
import numpy as np
import matplotlib.pyplot as plt
# local imports
from model import AutoEncoder
from utils import MinMaxScaler, sine_data_generation

TIMESTEPS = 100
ENCODING_DIM = 7
HIDDEN_DIM = 64
device = torch.device('cpu')

class MyDataset(torch.utils.data.Dataset):
  """Some Information about MyDataset"""
  def __init__(self, timeseries: np.array, timesteps: int):
    super(MyDataset, self).__init__()
    timeseries = MinMaxScaler(timeseries)
    self.X = torch.Tensor(timeseries).squeeze()

  def __getitem__(self, index):
    return self.X[index]

  def __len__(self):
    return self.X.shape[0]

  def temporalize(self, X, lookback):
    X = X.reshape(X.shape[0], 1)
    output_X = np.array([])
    for i in range(len(X)-lookback+1):
        t = []
        for j in range(0,lookback):
            # Gather past records upto the lookback period
            t.append(X[[(i+j)], :])
        output_X = np.append(output_X, t)
    return output_X.reshape(X.shape[0] - lookback + 1, 1, lookback)

# exponential = [(2*x + (x * random.randint(-15, 15)/100)) for x in range(1000)]
sine = sine_data_generation(1000, TIMESTEPS, 1)
timeseries = np.array(sine)

model = AutoEncoder(TIMESTEPS, ENCODING_DIM, [HIDDEN_DIM])
model.to(device)

dataset = MyDataset(timeseries, TIMESTEPS)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.L1Loss(reduction='sum')

# loop over the dataset multiple timese
for epoch in range(100):
  running_loss = 0.0
  for i, data in enumerate(dataloader):
    inputs = data
    inputs: torch.Tensor = inputs.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    # inputs = inputs.squeeze(0)
    outputs = model(inputs)
    outputs = outputs.squeeze()
    loss = criterion(outputs, inputs)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
  
  print("Loss: {}".format(running_loss))


model.eval()

more_sines = sine_data_generation(20, TIMESTEPS, 1)
more_sines = MinMaxScaler(more_sines)
for sin in more_sines:
  sin = torch.tensor(sin).transpose(0,1).float()
  out: torch.Tensor = model.forward(sin)
  print("\n\ntest = ", sin, "\nOut = ", out)
  plt.plot(sin.squeeze().data, label = 'original')
  plt.plot(out.squeeze().data, label = 'model out')
  plt.legend()
  plt.show()