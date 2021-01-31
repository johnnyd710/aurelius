# third party
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
# local imports
from autoencoder import VarationalAutoencoder
from utils import MinMaxScaler, sine_data_generation, SoftDTW

RESTORE = False
EPOCHS = 10
TIMESTEPS = 100
ENCODING_DIM = 7
HIDDEN_DIM = 64
LATENT_SIZE = 7
PI = 3.14
if torch.cuda.is_available():
  print('using gpu...')
  device = "cuda:0" 
else:  
  print('using cpu...')
  device = "cpu"  

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
sine = sine_data_generation(10000,
  TIMESTEPS,
  1,
  frequency=[0, 0.1],
  phase=[0, 7])
timeseries = np.array(sine)

model = VarationalAutoencoder(TIMESTEPS, ENCODING_DIM, LATENT_SIZE, [HIDDEN_DIM])
model.to(device)

dataset = MyDataset(timeseries, TIMESTEPS)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.MSELoss()
criterion = SoftDTW(gamma=1.0, normalize=False)

if RESTORE:
  model.load_state_dict(torch.load('checkpoint.pt'))
  optimizer.load_state_dict(torch.load('optimizer-checkpoint.pt'))

# loop over the dataset multiple timese
for epoch in range(EPOCHS):
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
  
  print("Epoch {:d} complete out of {:d}".format(epoch+1, EPOCHS))
  print("Loss: {}".format(running_loss))
  torch.save(model.state_dict(), 'checkpoint.pt')
  torch.save(optimizer.state_dict(), 'optimizer-checkpoint.pt')


model.eval()

more_sines = sine_data_generation(5, TIMESTEPS, 1)
more_sines = MinMaxScaler(more_sines)
for sin in more_sines:
  sin = torch.tensor(sin).transpose(0,1).float()
  out: torch.Tensor = model.forward(sin)
  print("\n\ntest = ", sin, "\nOut = ", out)
  plt.plot(sin.squeeze().data, label = 'original')
  plt.plot(out.squeeze().data, label = 'model out')
  plt.legend()
  plt.show()