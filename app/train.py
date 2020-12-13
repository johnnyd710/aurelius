import torch
import numpy as np
from model import AutoEncoder

TIMESTEPS = 3
HIDDEN_DIM = 64
LAYERS = 1

def normalize(x: np.array):
  return (x - x.min()) / (x.max() - x.min())

class MyDataset(torch.utils.data.Dataset):
  """Some Information about MyDataset"""
  def __init__(self, timeseries: np.array, timesteps: int):
    super(MyDataset, self).__init__()
    timeseries = normalize(timeseries)
    X = self.temporalize(
      X = timeseries,
      lookback=timesteps
    )
    self.X = torch.Tensor(X)

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

timeseries = np.array([
  0.1**3, 0.2**3, 0.3**3, 0.4**3, 0.5**3, 0.6**3, 0.7**3, 0.8**3, 0.9**3, 1.0**3,
  1.1**3, 1.2**3, 1.3**3, 1.4**3, 1.5**3, 1.6**3, 1.7**3, 1.8**3, 1.9**3, 2.0**3,
])

model = AutoEncoder(TIMESTEPS, HIDDEN_DIM, LAYERS)

dataset = MyDataset(timeseries, TIMESTEPS)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

criterion = torch.nn.MSELoss()

device = torch.device('cpu')

# loop over the dataset multiple timese
for epoch in range(5):
  running_loss = 0.0
  for i, data in enumerate(dataloader):
    inputs = data
    inputs = inputs.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, inputs)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
  
  print("Loss: {}".format(running_loss))


model.eval()

test = torch.Tensor([1**3,2**3,3**3]).view(1,1,3)
test = normalize(test)
print("test = ", test, "Out = ", model.forward(test))