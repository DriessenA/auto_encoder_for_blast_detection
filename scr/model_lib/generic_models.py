from typing import List
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_experiments.models.utils import setup_optimizer


class FlexibleVariationalEncoder(nn.Module):
    def __init__(
        self,
        layers: List[List[int]],
        activation: nn.Module = torch.nn.ReLU(),
    ):
        super(FlexibleVariationalEncoder, self).__init__()
        self.first_part = nn.ModuleList()

        self._activation_cls = getattr(nn, activation)
        assert isinstance(
            self._activation_cls(), nn.Module
        ), "Activation should be a torch.nn.modules.Module."

        for i, dimensions in enumerate(layers):
            in_dim, out_dim = dimensions
            if i < len(layers) - 1:
                self.first_part.append(nn.Linear(in_dim, out_dim))
                self.first_part.append(self._activation_cls())

            elif i == len(layers) - 1:
                self.linear2 = nn.Linear(in_dim, out_dim)
                self.linear3 = nn.Linear(in_dim, out_dim)
            else:
                Exception("Error in defining layers of the variational encoder")

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward_module_list(self, x):
        input_data = x
        for layer in self.first_part:
            input_data = layer(input_data)
        return input_data

    def forward(self, x):
        x = self.forward_module_list(x.float())
        mu = self.linear2(x.float())
        sigma = torch.exp(0.5 * self.linear3(x.float()))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = torch.mean(
            -0.5 * torch.sum(1 + sigma - mu**2 - sigma.exp(), dim=1), dim=0
        )

        return z


class FlexibleDecoder(nn.Module):
    def __init__(
        self,
        layers: List[List[int]],
        activation: nn.Module = torch.nn.ReLU(),
    ):
        super(FlexibleDecoder, self).__init__()
        self.decoder = nn.ModuleList()

        self._activation_cls = getattr(nn, activation)
        assert isinstance(
            self._activation_cls(), nn.Module
        ), "Activation should be a torch.nn.modules.Module."

        for i, dimensions in enumerate(layers):
            in_dim, out_dim = dimensions
            if i < len(layers) - 1:
                self.decoder.append(nn.Linear(in_dim, out_dim))
                self.decoder.append(self._activation_cls())
            elif i == len(layers) - 1:
                self.decoder.append(nn.Linear(in_dim, out_dim))
                self.decoder.append(nn.Sigmoid())
            else:
                Exception("Error in defining layers of the variational decoder")

    def forward_module_list(self, x):
        input_data = x
        for layer in self.decoder:
            input_data = layer(input_data)
        return input_data

    def forward(self, x):
        input_data = x.float()
        for layer in self.decoder:
            input_data = layer(input_data.float())
        return input_data


class FlexibleVAE(nn.Module):
    def __init__(
        self,
        enc_layers: List[List[int]],
        dec_layers: List[List[int]],
        activation: str = "ReLU",
    ):
        super(FlexibleVAE, self).__init__()

        self.encoder = FlexibleVariationalEncoder(
            layers=enc_layers,
            activation=activation,
        )
        self.decoder = FlexibleDecoder(
            layers=dec_layers,
            activation=activation,
        )

    def forward(self, x):
        z = self.encoder(x.float())
        pred = self.decoder(z)
        return pred

    def get_latent_space(self, x):

        with torch.no_grad():
            z = self.encoder(x.float())
        return z


class FlexibleEncoder(nn.Module):
    def __init__(
        self,
        layers: List[List[int]],
        activation: nn.Module = torch.nn.ReLU(),
    ):
        super(FlexibleEncoder, self).__init__()
        self.encoder = nn.ModuleList()

        self._activation_cls = getattr(nn, activation)
        assert isinstance(
            self._activation_cls(), nn.Module
        ), "Activation should be a torch.nn.modules.Module."

        for i, dimensions in enumerate(layers):
            in_dim, out_dim = dimensions
            if i < len(layers) - 1:
                self.encoder.append(nn.Linear(in_dim, out_dim))
                self.encoder.append(self._activation_cls())
            elif i == len(layers) - 1:
                self.encoder.append(nn.Linear(in_dim, out_dim))
            else:
                Exception("Error in defining layers of the encoder")

    def forward_module_list(self, x):
        input_data = x
        for layer in self.encoder:
            input_data = layer(input_data)
        return input_data

    def forward(self, x):
        input_data = x.float()
        for layer in self.encoder:
            input_data = layer(input_data.float())
        return input_data


class FlexibleAE(nn.Module):
    def __init__(
        self,
        enc_layers: List[List[int]],
        dec_layers: List[List[int]],
        activation: str = "ReLU",
    ):
        super(FlexibleAE, self).__init__()

        self.encoder = FlexibleEncoder(
            layers=enc_layers,
            activation=activation,
        )
        self.decoder = FlexibleDecoder(
            layers=dec_layers,
            activation=activation,
        )
        self.encoder.kl = 0

    def forward(self, x):
        enc = self.encoder(x.float())
        pred = self.decoder(enc)
        return pred

    def get_latent_space(self, x):

        with torch.no_grad():
            enc = self.encoder(x.float())
        return enc

    def additional_loss_func(self):
        return 0.0


class vAELightning(pl.LightningModule):
    def __init__(self, encoder_decoder, optimization):
        super(vAELightning, self).__init__()

        self.endec = encoder_decoder
        self.softmax = nn.Softmax(dim=1)
        self.optim_config = optimization

    def configure_optimizers(self):
        return setup_optimizer(self, self.optim_config)

    def forward(self, x):
        return self.endec(x.float())

    def predict(self, x):
        pred = self.endec(x.float())
        return pred

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x_hat = self(x.float())
        loss_func = torch.nn.MSELoss()
        loss = loss_func(x_hat, x) + self.endec.encoder.kl
        self.log("train/loss", loss)
        self.log("train/MSE", loss_func(x_hat, x))
        self.log("train/kl", self.endec.encoder.kl)
        return loss

    def _eval(self, batch, batch_idx, set_name: str):
        x, y = batch
        x_hat = self(x.float())
        loss_func = torch.nn.MSELoss()
        loss = loss_func(x_hat, x) + self.endec.encoder.kl
        self.log("{}/loss".format(set_name), loss)
        self.log("{}/MSE".format(set_name), loss_func(x_hat, x))
        self.log("{}/kl".format(set_name), self.endec.encoder.kl)

    def validation_step(self, valid_batch, batch_idx):
        self._eval(valid_batch, batch_idx, "valid")

    def test_step(self, test_batch, batch_idx):
        self._eval(test_batch, batch_idx, "test")
