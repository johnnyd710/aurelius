import torch

class AutoEncoder(torch.nn.Module):
    def __init__(self, timesteps: int, hidden_dim: int, layer_dim: int):
        super(AutoEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.timesteps = timesteps
        self.encoder = torch.nn.LSTM(timesteps, hidden_dim, layer_dim, batch_first=True)
        self.decoder = torch.nn.LSTM(hidden_dim, timesteps, layer_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # hidden state
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        latent, (hn, cn) = self.encoder(x, (h0.detach(), c0.detach()))
        latent = self.relu(latent)
        # hidden state
        h0 = torch.zeros(self.layer_dim, x.size(0), self.timesteps).requires_grad_()
        # cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.timesteps).requires_grad_()
        out, (hn, cn) = self.decoder(latent, (h0.detach(), c0.detach()))
        return out

input_dim = 3
hidden_dim = 64
layer_dim = 2
model = AutoEncoder(input_dim, hidden_dim, layer_dim)