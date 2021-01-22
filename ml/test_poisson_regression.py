#!/usr/bin/env python3
"""Test poisson regression model."""

from pytest import approx
from torch import tensor
from torch.nn import PoissonNLLLoss

from ml.domain.data import DataSet
from ml.domain.loss import Loss
from ml.domain.model import HyperParameters
from ml.poisson_regression import define_model


def test_overfit():
    """Test overfitting the model.

    The aim is to test if the model-training is correctly specified. With an erroneous
    specification, a model may not overfit.

    """
    n_dimensions = 4
    dataset = DataSet(
        x=tensor([[1.0] * n_dimensions, [2.0] * n_dimensions]), y=tensor([[1.0], [2.0]])
    )
    model = define_model(HyperParameters(n_days=n_dimensions, l2_coef=0.0))
    model.train(dataset, n_epochs=10000)

    actual = Loss.calculate(dataset, model.model, is_yhat_logged=True)
    # Expected loss is when inference equals the outcomes.
    expected = PoissonNLLLoss(full=True, log_input=False)(dataset.y, dataset.y).item()

    assert actual.poisson_nll == approx(expected)


def test_underfit():
    """Test underfitting the model.

    The aim is to test if the regularisation parameter (`l2_coef`) is set up in a way
    that a larger value makes the weight approach zero.

    """
    n_dimensions = 4
    dataset = DataSet(
        x=tensor([[1.0] * n_dimensions, [2.0] * n_dimensions]), y=tensor([[1.0], [2.0]])
    )
    model = define_model(HyperParameters(n_days=n_dimensions, l2_coef=1e5))
    model.train(dataset, n_epochs=10000)

    parameter = [x for name, x in model.model.named_parameters() if name == "weight"]
    assert len(parameter) == 1
    norm = parameter[0].norm()
    assert norm < 1e-4
