#!/usr/bin/env python3
"""Poisson regression model.

As the visit counts are usually sparse (i.e., many bookmarks are not visited on a given
day), the visit counts are filtered out if they contain only zero counts during the past
few days. An alternative is to train the zero-inflated Poisson model.

"""

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from getpass import getuser
from pathlib import Path
from random import Random
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm


@dataclass(frozen=True)
class DataSet:
    x: torch.tensor
    y: torch.tensor


@dataclass(frozen=True)
class Data:
    training: DataSet
    validation: DataSet

    @staticmethod
    def load(n_days: int) -> "Data":
        path = Path.home().joinpath(
            ".local", "share", "bookmark-manager", "data.sqlite3"
        )
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(
            "WITH date_log AS ( "
            "SELECT bookmarkId, SUBSTR(visitedAt, 1, 10) AS date "
            "FROM bookmarkVisitLog WHERE username = ? "
            ") "
            "SELECT bookmarkId, date, COUNT(*) AS n "
            "FROM date_log GROUP BY bookmarkId, date ",
            (getuser(),),
        )
        all_counts = {}
        for row in cursor.fetchall():
            all_counts.update({(row[0], date.fromisoformat(row[1])): row[2]})
        conn.close()

        bookmark_ids = sorted({key[0] for key in all_counts})
        Random(12345).shuffle(bookmark_ids)
        data = Data._wrangle_data(bookmark_ids, all_counts, n_days)

        n_training = len(data["y"]) * 3 // 4

        training_data = DataSet(
            x=torch.tensor(data["x"][:n_training]),
            y=torch.tensor(data["y"][:n_training]),
        )
        validation_data = DataSet(
            x=torch.tensor(data["x"][n_training:]),
            y=torch.tensor(data["y"][n_training:]),
        )

        assert training_data.x.shape == (n_training, n_days)
        assert training_data.y.shape == (n_training, 1)
        assert validation_data.x.shape == (len(data["y"]) - n_training, n_days)
        assert validation_data.y.shape == (len(data["y"]) - n_training, 1)

        return Data(training=training_data, validation=validation_data)

    @staticmethod
    def _wrangle_data(
        bookmark_ids: List[str], all_counts: Dict[Tuple[str, date], int], n_days: int
    ) -> Dict[str, List[List[float]]]:
        today = date.today()
        xs, ys = [], []
        for bid in bookmark_ids:
            x = [
                float(all_counts.get((bid, today - timedelta(days=n)), 0))
                for n in range(2, n_days + 2)
            ]
            if sum(x) > 0:
                xs.append(x)
                ys.append([float(all_counts.get((bid, today - timedelta(days=1)), 0))])
        return {"x": xs, "y": ys}


def fit_poisson_regression(
        data: DataSet, l2_coef: float = 1.0, n_epochs: int = 10000
) -> torch.jit.ScriptModule:
    model = torch.nn.Linear(data.x.shape[1], data.y.shape[1])
    loss_function = torch.nn.PoissonNLLLoss()
    # Use Adadelta or Adam, to easily include L2 penality (weight_decay).
    optimiser = torch.optim.Adam(model.parameters(), weight_decay=l2_coef)

    for _ in tqdm(range(n_epochs), "model-training"):

        def _evaluate():
            optimiser.zero_grad()
            outputs = model(data.x)
            loss = loss_function(outputs, data.y)
            loss.backward()
            return loss

        optimiser.step(_evaluate)

    return torch.jit.script(model)


def get_poisson_loss(data: DataSet, model: torch.jit.ScriptModule) -> float:
    loss_function = torch.nn.PoissonNLLLoss(full=True)
    outputs = model(data.x)
    loss = loss_function(outputs, data.y)
    return loss.item()


def load(src: Path) -> torch.jit.ScriptModule:
    return torch.jit.load(src.as_posix())


def save(model: torch.jit.ScriptModule, dest: Path) -> None:
    model.save(dest.as_posix())


def main():
    n_dimensions = 10
    modelname = "poisson_regression"
    model_path = Path(__file__).parent.parent.joinpath(
        "src", "main", "resources", f"{modelname}.pt"
    )
    assert model_path.parent.exists()

    data = Data.load(n_dimensions)
    print("# Poisson Regression")
    print("")
    model = fit_poisson_regression(data.training)
    error = get_poisson_loss(data.validation, model)
    print("## Candidate")
    print(f"Training error: {get_poisson_loss(data.training, model)}")
    print(f"Validation error: {error}")
    print("")

    is_model_better = True
    if model_path.exists():
        current = load(model_path)
        error_with_current = get_poisson_loss(data.validation, current)
        print("## Current")
        print(f"Training error: {get_poisson_loss(data.training, current)}")
        print(f"Validation error: {error_with_current}")
        print("")
        is_model_better = error < error_with_current

    if is_model_better:
        print("Replacing the production model.")
        save(model, model_path)
