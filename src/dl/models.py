"""LSTM backbone and Lightning training heads."""

from __future__ import annotations

from collections.abc import Callable

import pytorch_lightning as pl
import torch
from torch import nn


class SimpleRNN(nn.Module):
    """RNN / LSTM / GRU module with packed-sequence support."""

    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        num_layers=1,
        bidirectional=False,
        pre_rnn_linear_size=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if pre_rnn_linear_size is not None:
            self.pre_rnn_linear = nn.Linear(input_size, pre_rnn_linear_size)
            self.input_size = pre_rnn_linear_size
        else:
            self.pre_rnn_linear = None
            self.input_size = input_size

        if rnn_type == "RNN":
            self.rnn = nn.RNN(
                self.input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                self.input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                self.input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(
                f"Unsupported rnn_type: {rnn_type}. Choose from 'RNN', 'LSTM', 'GRU'."
            )

    def forward(self, sequences, lengths, return_all=False):
        x = sequences
        if self.pre_rnn_linear is not None:
            x = self.pre_rnn_linear(x)

        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.rnn(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        if return_all:
            return out, hidden

        if self.bidirectional:
            if isinstance(hidden, tuple):
                hidden = torch.cat((hidden[0][-2], hidden[0][-1]), dim=1)
            else:
                hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            if isinstance(hidden, tuple):
                hidden = hidden[0][-1]
            else:
                hidden = hidden[-1]
        return hidden


class DictWrapperRNN(nn.Module):
    """Wrapper so the Lightning head can pass dict batches to the RNN."""

    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn

    def forward(self, sequences=None, lengths=None, **kwargs):
        if sequences is None:
            sequences = kwargs.get("sequences")
        if lengths is None:
            lengths = kwargs.get("lengths")
        if lengths is not None and hasattr(lengths, "is_cuda") and lengths.is_cuda:
            lengths = lengths.cpu()
        return self.rnn(sequences, lengths)


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.ModuleList | nn.Module,
        output_size,
        hidden_layers: tuple = (),
        activation=nn.ReLU(),
        learning_rate: float = 1e-3,
        optimizer_class: Callable = torch.optim.AdamW,
        optimizer_params: dict | None = None,
        scheduler_class: Callable | None = None,
        scheduler_params: dict | None = None,
        loss_fn: Callable | None = None,
    ):
        super().__init__()
        self.backbone = backbone

        modules = []
        for hidden_units in hidden_layers:
            modules.append(nn.LazyLinear(hidden_units))
            modules.append(activation)
        modules.append(nn.LazyLinear(output_size))
        self.head = nn.ModuleList(modules)

        self.loss_fn = loss_fn
        self.LR = learning_rate
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params if scheduler_params is not None else {}
        self.flat = nn.Flatten()

    def forward(self, x):
        if isinstance(x, dict):
            if isinstance(self.backbone, nn.Sequential):
                if len(x) == 1:
                    x = self.backbone(next(iter(x.values())))
                else:
                    x = self.backbone(*x.values())
            else:
                x = self.backbone(**x)
        else:
            x = self.backbone(x)

        if len(x.size()) == 4:
            x = self.flat(x)
        for layer in self.head:
            x = layer(x)
        return x

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(), lr=self.LR, **self.optimizer_params
        )
        if self.scheduler_class is not None:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_params)
            return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        y = batch.pop("y")
        out = self(batch)
        loss = self.loss_fn(out, y)
        if getattr(self, "_trainer", None) is not None:
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch.pop("y")
        out = self(batch)
        loss = self.loss_fn(out, y)
        if getattr(self, "_trainer", None) is not None:
            self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


class Classifier(BaseModel):
    def __init__(
        self,
        backbone: nn.ModuleList | nn.Module,
        n_classes,
        classifier_hidden_layers=(),
        classifier_activation=nn.ReLU(),
        learning_rate=1e-3,
        optimizer_class: Callable = torch.optim.AdamW,
        optimizer_params: dict | None = None,
        scheduler_class: Callable | None = None,
        scheduler_params: dict | None = None,
    ):
        super().__init__(
            backbone=backbone,
            output_size=n_classes,
            hidden_layers=classifier_hidden_layers,
            activation=classifier_activation,
            learning_rate=learning_rate,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
            loss_fn=nn.CrossEntropyLoss(),
        )


class Regressor(BaseModel):
    def __init__(
        self,
        backbone: nn.ModuleList | nn.Module,
        output_dim,
        regressor_hidden_layers=(),
        regressor_activation=nn.ReLU(),
        learning_rate=1e-3,
        optimizer_class: Callable = torch.optim.AdamW,
        optimizer_params: dict | None = None,
        scheduler_class: Callable | None = None,
        scheduler_params: dict | None = None,
    ):
        super().__init__(
            backbone=backbone,
            output_size=output_dim,
            hidden_layers=regressor_hidden_layers,
            activation=regressor_activation,
            learning_rate=learning_rate,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
            loss_fn=nn.MSELoss(),
        )
