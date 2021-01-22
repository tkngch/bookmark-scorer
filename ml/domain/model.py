#!/usr/bin/env python3
"""Domain model for machine-learning or statistical models."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import torch
from tqdm import tqdm

from ml.domain.data import DataSet
from ml.domain.loss import Loss


@dataclass(frozen=True)
class HyperParameters:
    """Model hyper-parameters."""

    n_days: int
    l2_coef: Optional[float] = None


@dataclass(frozen=True)
class ModelMetaData:
    """Model metadata."""

    model_name: str
    hyper_parameters: HyperParameters
    is_yhat_logged: bool

    def to_dict(self) -> Dict[str, Any]:
        """Represent this class as dict."""
        return {
            "model_name": self.model_name,
            "hyper_parameters": asdict(self.hyper_parameters),
            "is_yhat_logged": self.is_yhat_logged,
        }

    @staticmethod
    def from_dict(input_dict: Dict[str, Any]) -> "ModelMetaData":
        """Instantiate ModelMetaData with dict."""
        return ModelMetaData(
            model_name=input_dict["model_name"],
            hyper_parameters=HyperParameters(**input_dict["hyper_parameters"]),
            is_yhat_logged=input_dict["is_yhat_logged"],
        )


@dataclass(frozen=True)
class Model:
    """Functions for training a model."""

    model: Union[torch.nn.Module, torch.jit.ScriptModule]
    loss_function: Optional[torch.nn.Module]
    optimiser: Optional[torch.optim.Optimizer]
    metadata: ModelMetaData

    def train(self, data: DataSet, n_epochs: int) -> None:
        """Fit a model with gradient descent."""
        if self.loss_function is not None and self.optimiser is not None:
            self._train(data, n_epochs)

    def _train(self, data: DataSet, n_epochs: int) -> None:
        for _ in tqdm(range(n_epochs), f"Training {self.metadata.model_name}"):

            def _evaluate():
                self.optimiser.zero_grad()
                outputs = self.model(data.x)
                loss = self.loss_function(outputs, data.y)
                loss.backward()
                return loss

            self.optimiser.step(_evaluate)

    def loss(self, data: DataSet, is_yhat_logged: bool) -> Loss:
        """Calculate the loss."""
        return Loss.calculate(data, self.model, is_yhat_logged)
