import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ, out_activ):
        super(Encoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True
            )
            self.layers.append(layer)

        self.h_activ, self.out_activ = h_activ, out_activ

    def forward(self, x):
        x = x.unsqueeze(0)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)
            elif self.out_activ and index == self.num_layers - 1:
                return self.out_activ(h_n).squeeze()

        return h_n.squeeze()


class Decoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ):
        super(Decoder, self).__init__()
        layer_dims = [input_dim] + h_dims + [h_dims[-1]]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True
            )
            self.layers.append(layer)

        self.h_activ = h_activ
        self.dense_matrix = nn.Parameter(
            torch.rand((layer_dims[-1], out_dim), dtype=torch.float),
            requires_grad=True
        )

    def forward(self, x, seq_len):
        x = x.repeat(seq_len, 1).unsqueeze(0)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)
        return torch.mm(x.squeeze(1), self.dense_matrix)


class VarationalAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, latent_size, h_dims=[], h_activ=nn.Sigmoid(),
                 out_activ=nn.Tanh(),
                 batch_size=1):
        super(VarationalAutoencoder, self).__init__()

        self.encoder = Encoder(input_dim, encoding_dim, h_dims, h_activ,
                               out_activ)
        self.decoder = Decoder(encoding_dim, input_dim, h_dims[::-1],
                               h_activ)
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.encoding_to_mu = nn.Linear(encoding_dim, latent_size)
        self.encoding_to_var = nn.Linear(encoding_dim, latent_size)
        self.latent_to_encoding = nn.Linear(latent_size, encoding_dim)

    def forward(self, x):
        seq_len = x.shape[0]
        x = self.encoder(x)
        mean = self.encoding_to_mu(x)
        logvar = self.encoding_to_var(x)
        std = torch.exp(0.5 * logvar)
        z = torch.randn([self.batch_size, self.latent_size])
        z = z * std + mean
        x = self.latent_to_encoding(z)
        x = self.decoder(x, seq_len)

        return x