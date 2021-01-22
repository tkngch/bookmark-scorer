#!/usr/bin/env python3
"""Domain models to represent data."""

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from getpass import getuser
from pathlib import Path
from random import Random
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class DataSet:
    """One set of data points."""

    x: torch.tensor
    y: torch.tensor


@dataclass(frozen=True)
class DataSets:
    """Multiple sets of data points."""

    training: DataSet
    validation: DataSet


@dataclass(frozen=True)
class DailyVisitCounts(DataSets):
    """The number of times a bookmark is visited per day.

    A row represents a bookmark, and a column represents a day: the first column stores
    the visit counts from 1 day ago, the second column stores the visit counts from 2
    days ago, and so on.

    """

    @staticmethod
    def load(n_days: int) -> "DailyVisitCounts":
        """Load data from the database."""
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
        Random(1234).shuffle(bookmark_ids)
        data = DailyVisitCounts._wrangle_data(bookmark_ids, all_counts, n_days)

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

        return DailyVisitCounts(training=training_data, validation=validation_data)

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
