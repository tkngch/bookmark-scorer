#!/usr/bin/env python3

from pytest import approx
from torch import tensor
from torch.nn import PoissonNLLLoss

from ml.poisson_regression import (DataSet, fit_poisson_regression,
                                   get_poisson_loss)


def test_overfit():
    """Test overfitting the model.

    The aim is to test if the model-training is correctly specified. With an erroneous
    specification, a model may not overfit.

    """
    dataset = DataSet(x=tensor([[1., 1., 1., 1.], [2., 2., 2., 2.]]), y=tensor([[1.], [2.]]))
    model = fit_poisson_regression(dataset, l2_coef=0.0)

    actual = get_poisson_loss(dataset, model)
    # Expected loss is when inference equals the outcomes.
    expected = PoissonNLLLoss(full=True, log_input=False)(dataset.y, dataset.y).item()

    assert actual == approx(expected)


def test_underfit():
    """Test underfitting the model.

    The aim is to test if the regularisation parameter (`l2_coef`) is set up in a way
    that a larger value makes the weight approach zero.

    """
    dataset = DataSet(x=tensor([[1., 1., 1., 1.], [2., 2., 2., 2.]]), y=tensor([[1.], [2.]]))
    model = fit_poisson_regression(dataset, l2_coef=1e5)

    parameter = [x for name, x in model.named_parameters() if name == "weight"]
    assert len(parameter) == 1
    norm = parameter[0].norm()
    assert norm < 1e-4
