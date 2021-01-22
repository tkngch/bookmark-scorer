#!/usr/bin/env python3
"""Domain models for Loss."""

from dataclasses import dataclass
from typing import Optional, Union

import torch

from ml.domain.data import DataSet, DataSets


@dataclass(frozen=True)
class Loss:
    """Loss values."""

    mse: Optional[float]
    poisson_nll: Optional[float]

    @staticmethod
    def calculate(
        data: DataSet, model: torch.jit.ScriptModule, is_yhat_logged: bool = False
    ) -> "Loss":
        """Calculate the loss for a given model and data."""
        yhat = model(data.x)
        if is_yhat_logged:
            outputs = torch.exp(yhat)
        else:
            outputs = yhat

        mse = torch.nn.MSELoss()(outputs, data.y)
        poisson_nll = torch.nn.PoissonNLLLoss(log_input=False, full=True)(
            outputs, data.y
        )

        return Loss(mse=mse.item(), poisson_nll=poisson_nll.item())

    def __str__(self) -> str:
        output = ""
        if self.mse:
            output += f"MSE: {round(self.mse, 2)}; "
        if self.poisson_nll:
            output += f"PoissonNLL: {round(self.poisson_nll, 2)}; "

        return output

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class Losses:
    training: Loss
    validation: Loss

    @staticmethod
    def calculate(
        data: DataSets,
        model: Union[torch.nn.Module, torch.jit.ScriptModule],
        is_yhat_logged: bool,
    ) -> "Losses":
        return Losses(
            training=Loss.calculate(data.training, model, is_yhat_logged),
            validation=Loss.calculate(data.validation, model, is_yhat_logged),
        )

    def __str__(self) -> str:
        return f"Training: {self.training} Validation: {self.validation}"

    def __repr__(self) -> str:
        return self.__str__()
