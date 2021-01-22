#!/usr/bin/env python3
"""Poisson regression model.

As the visit counts are usually sparse (i.e., many bookmarks are not visited on a given
day), the visit counts are filtered out if they contain only zero counts during the past
few days. An alternative is to train the zero-inflated Poisson model.

"""

from pathlib import Path

import torch

from ml.domain.data import DailyVisitCounts
from ml.domain.loss import Losses
from ml.domain.model import HyperParameters, Model, ModelMetaData


def define_model(hyper_parameters: HyperParameters) -> Model:
    """Define a Poisson regression model."""
    metadata = ModelMetaData(
        model_name=Path(__file__).stem,
        hyper_parameters=hyper_parameters,
        is_yhat_logged=True,
    )

    torch_model = torch.nn.Linear(hyper_parameters.n_days, 1)
    loss_function = torch.nn.PoissonNLLLoss()
    # Use Adadelta or Adam, to easily include L2 penality (weight_decay).
    optimiser = torch.optim.Adam(
        torch_model.parameters(), weight_decay=hyper_parameters.l2_coef
    )
    return Model(
        model=torch_model,
        loss_function=loss_function,
        optimiser=optimiser,
        metadata=metadata,
    )


if __name__ == "__main__":
    model = define_model(HyperParameters(n_days=10, l2_coef=1.0))
    data = DailyVisitCounts.load(model.metadata.hyper_parameters.n_days)
    model.train(data.training, n_epochs=10000)
    losses = Losses.calculate(
        data, model.model, is_yhat_logged=model.metadata.is_yhat_logged
    )
    print(losses)
