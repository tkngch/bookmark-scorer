#!/usr/bin/env python3
"""Parameter-free averaging model."""


from pathlib import Path

import torch

from ml.domain.data import DailyVisitCounts
from ml.domain.loss import Losses
from ml.domain.model import HyperParameters, Model, ModelMetaData


class Averaging(torch.nn.Module):
    """Averaging model to be used as a baseline."""

    def __init__(self, n_days: int) -> None:
        super().__init__()
        self.ones = torch.ones(1, n_days) / n_days

    def forward(self, x):
        return torch.mm(x, self.ones.T)


def define_model(hyper_parameters: HyperParameters) -> Model:
    """Define a model."""
    metadata = ModelMetaData(
        model_name=Path(__file__).stem,
        hyper_parameters=hyper_parameters,
        is_yhat_logged=False,
    )
    return Model(
        model=Averaging(metadata.hyper_parameters.n_days),
        loss_function=None,
        optimiser=None,
        metadata=metadata,
    )


if __name__ == "__main__":
    model = define_model(HyperParameters(n_days=10))
    data = DailyVisitCounts.load(model.metadata.hyper_parameters.n_days)
    losses = Losses.calculate(
        data, model.model, is_yhat_logged=model.metadata.is_yhat_logged
    )
    print(losses)
