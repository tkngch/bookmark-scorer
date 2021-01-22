#!/usr/bin/env python3
"""Entrypoint."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Tuple

import torch

from ml import averaging, poisson_regression, temporally_discounted_averaging
from ml.domain.data import DailyVisitCounts
from ml.domain.loss import Losses
from ml.domain.model import HyperParameters, Model, ModelMetaData


def main() -> None:
    """Find the best model and if necessary, replace the production model."""
    resource_path = Path(__file__).parent.parent.joinpath("src", "main", "resources")
    assert resource_path.exists()

    candidate_model, candidate_losses = _get_candidate()
    is_candidate_better = True

    model_path = resource_path.joinpath("model.pt")
    metadata_path = resource_path.joinpath("model_metadata.json")
    if model_path.exists() and metadata_path.exists():
        _, losses = _load_production(model_path, metadata_path)
        is_candidate_better = candidate_losses.validation.mse < losses.validation.mse

    if is_candidate_better:
        print("Storing the candidate model as the production model.")
        torch.jit.script(candidate_model.model).save(model_path.as_posix())
        with open(metadata_path, "w") as handler:
            json.dump(candidate_model.metadata.to_dict(), handler)


def _load_production(model_path: Path, metadata_path: Path) -> Tuple[Model, Losses]:
    with open(metadata_path, "r") as handler:
        metadata = ModelMetaData.from_dict(json.load(handler))

    torch_model = torch.jit.load(model_path.as_posix())
    model = Model(
        model=torch_model, loss_function=None, optimiser=None, metadata=metadata
    )

    data = DailyVisitCounts.load(metadata.hyper_parameters.n_days)
    losses = Losses.calculate(data, model.model, is_yhat_logged=metadata.is_yhat_logged)

    print(f"Production {metadata.model_name}\t{losses}")
    return model, losses


def _get_candidate() -> Tuple[Model, Losses]:
    hyper_parameters = HyperParameters(n_days=10)
    data = DailyVisitCounts.load(hyper_parameters.n_days)

    baseline_model = averaging.define_model(hyper_parameters)
    baseline_losses = Losses.calculate(
        data,
        baseline_model.model,
        is_yhat_logged=baseline_model.metadata.is_yhat_logged,
    )

    poisson_model = poisson_regression.define_model(
        replace(hyper_parameters, l2_coef=1.0)
    )
    poisson_model.train(data.training, n_epochs=10000)
    poisson_losses = Losses.calculate(
        data, poisson_model.model, is_yhat_logged=poisson_model.metadata.is_yhat_logged
    )

    temporal_model = temporally_discounted_averaging.define_model(hyper_parameters)
    temporal_model.train(data.training, n_epochs=1000)
    temporal_losses = Losses.calculate(
        data,
        temporal_model.model,
        is_yhat_logged=temporal_model.metadata.is_yhat_logged,
    )

    n_chars = 20
    print("")
    print(
        f"{baseline_model.metadata.model_name.ljust(n_chars)[:n_chars]}\t{baseline_losses}"
    )
    print(
        f"{temporal_model.metadata.model_name.ljust(n_chars)[:n_chars]}\t{temporal_losses}"
    )
    print(
        f"{poisson_model.metadata.model_name.ljust(n_chars)[:n_chars]}\t{poisson_losses}"
    )
    print("")

    candidate = sorted(
        [
            (baseline_model, baseline_losses),
            (poisson_model, poisson_losses),
            (temporal_model, temporal_losses),
        ],
        key=lambda entry: entry[1].validation.mse,
    )[0]
    print(f"{candidate[0].metadata.model_name} performs the best.")
    print("")

    return candidate


if __name__ == "__main__":
    main()
