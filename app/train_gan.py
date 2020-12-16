import torch
import numpy as np
from model import AutoEncoder
import random

SEQ_LEN = 60
HIDDEN_DIM = 64
BATCH_SIZE = 16
N_FEATURES = 1

    batch_size = 16
    seq_len = 32
    noise_dim = 100
    seq_dim = 4

device = torch.device('cpu')

def normalize(x: np.array, _max: int, _min: int):
  return (x - _min) / (_max - _min)

class MyDataset(torch.utils.data.Dataset):
  """Some Information about MyDataset"""
  def __init__(self, timeseries: np.array, timesteps: int):
    super(MyDataset, self).__init__()
    self._min = timeseries.min()
    self._max = timeseries.max()
    timeseries = normalize(timeseries, self._max, self._min)
    X = self.temporalize(
      X = timeseries,
      lookback=timesteps
    )
    self.X = torch.Tensor(X)

  def __getitem__(self, index):
    return self.X[index]

  def getMax(self):
    return self._max

  def getMin(self):
    return self._min

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

exponential = [(x**2/100 + (x * random.randint(-15, 15)/100)) for x in range(1000)]
timeseries = np.array(
  exponential
)

# Create generator and discriminator models
netD = LSTMDiscriminator(in_dim=in_dim, device=device).to(device)
netG = LSTMGenerator(in_dim=in_dim, out_dim=in_dim, device=device).to(device)

dataset = MyDataset(timeseries, TIMESTEPS)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.L1Loss(reduction='sum')

# loop over the dataset multiple timese
for epoch in range(100):
  running_loss = 0.0
  for i, data in enumerate(dataloader):
    inputs = data
    inputs = inputs.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    outputs = outputs.view(1,1,TIMESTEPS)
    loss = criterion(outputs, inputs)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
  
  print("Loss: {}".format(running_loss))


model.eval()

test = np.array([20**2/100, 21**2/100, 22**2/100])
test = normalize(test, dataset.getMax(), dataset.getMin())
test = torch.Tensor(test).view(1,1,TIMESTEPS)
print("test = ", test, "Out = ", model.forward(test))