#!/usr/bin/env python3
"""Temporally-discounted averaging model.

In this model, a score f is given by

    f = \sum_{t = 1}^{m} n(t) w(t)
      = [n(1) n(2) n(3) ... n(m)] . [w(1) w(2) w(3) ... w(m)]^T

where n(t) is the visit counts from t days ago. m is a hyper-parameter, and w(t) is a
weight which satisfies

    w(t) > 0 for all t,    \sum_{t = 1}^{m} w(t) = 1,    and    w(t) > w(t + 1).

The last condition constrains the model as "temporally-discounting", such that recent
visits are given greater weights than distant visits.  We parameterise the weight

    w(t) =  \theta^t / \sum_{i = 1}^{m} \theta^i

and

    \theta = 1 / (1 + \exp(-\beta))

Then, \beta is a parameter to be estimated.

In the implementation, we use the following derivation:

    [\theta \theta^2 \theta^3 ... \theta^m]
    = \exp \log [\theta \theta^2 \theta^3 ... \theta^m]
    = \exp { [1 2 3 ... m] \log \theta }

"""

from pathlib import Path

import torch

from ml.domain.data import DailyVisitCounts
from ml.domain.loss import Losses
from ml.domain.model import HyperParameters, Model, ModelMetaData


class TemporallyDiscountedAveraging(torch.nn.Module):
    """Temporally-discounted averaging model."""

    def __init__(self, n_days: int) -> None:
        super().__init__()
        self.ones = torch.ones(1, n_days)
        self.beta = torch.nn.Parameter(torch.zeros(1, 1))
        self.days = torch.tensor([i + 1 for i in range(n_days)])

    def forward(self, x):
        theta = torch.sigmoid(self.beta)
        unnormalised_coefficient = torch.exp(torch.log(theta) * self.days)
        normalising_factor = torch.mm(unnormalised_coefficient, self.ones.T)
        coefficient = torch.div(unnormalised_coefficient, normalising_factor)
        return torch.mm(x, coefficient.T)


def define_model(hyper_parameters: HyperParameters) -> Model:
    """Train a model."""
    metadata = ModelMetaData(
        model_name=Path(__file__).stem,
        hyper_parameters=hyper_parameters,
        is_yhat_logged=False,
    )
    torch_model = TemporallyDiscountedAveraging(metadata.hyper_parameters.n_days)
    loss_function = torch.nn.MSELoss()
    optimiser = torch.optim.LBFGS(torch_model.parameters())

    return Model(
        model=torch_model,
        loss_function=loss_function,
        optimiser=optimiser,
        metadata=metadata,
    )


if __name__ == "__main__":
    model = define_model(HyperParameters(n_days=10))
    data = DailyVisitCounts.load(model.metadata.hyper_parameters.n_days)
    model.train(data.training, n_epochs=1000)
    losses = Losses.calculate(
        data, model.model, is_yhat_logged=model.metadata.is_yhat_logged
    )
    print(losses)
