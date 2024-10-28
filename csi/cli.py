# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (C) 2024 - Centreon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

from enum import Enum
from typing import Annotated, Callable

import numpy as np
import pandas as pd
import typer
from Levenshtein import distance as levenshtein_distance
from sklearn.cluster import DBSCAN  # type: ignore

app = typer.Typer()


class Algorithm(str, Enum):
    LEVENSHTEIN = "levenshtein"
    JACARD = "jacard"


def jacard_distance(status1: str, status2: str) -> float:
    set1 = set(status1)
    set2 = set(status2)
    return len(set1.intersection(set2)) / len(set1.union(set2))


def compute_distance_matrix(
    service_data: dict[str, str], algorithm: Algorithm, id: str | None = None
) -> np.ndarray | dict[str, float]:
    """Compute pairwise distances using Levenshtein"""
    assert (
        id is None or service_data.get(id) is not None
    ), f"'{id}' not found in the input data."

    n = len(service_data)
    distance_matrix = np.zeros((n, n))
    labels = list(service_data.keys())

    fun_algorithm: Callable[[str, str], float]
    match algorithm:
        case Algorithm.LEVENSHTEIN:
            fun_algorithm = levenshtein_distance
        case Algorithm.JACARD:
            fun_algorithm = jacard_distance
        case _:
            raise ValueError(f"Unsupported algorithm: {algorithm.value}")

    n = len(labels)
    for i in range(n):
        for j in range(i):
            status_i = service_data[labels[i]]
            status_j = service_data[labels[j]]
            distance = fun_algorithm(status_i, status_j)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    if id is not None:
        i = labels.index(id)
        distances = {label: distance_matrix[i, j] for j, label in enumerate(labels)}
        return dict(sorted(distances.items(), key=lambda item: item[1]))

    return distance_matrix


def read_csv(input: str, min_element: int = 10) -> dict[str, str]:
    symbols = {"0": "O", "1": "W", "2": "C", "3": "U"}

    # Load data in a df and replace status id with their symbol
    df = pd.read_csv(
        input,
        names=["ctime", "id", "status"],
        converters={"status": lambda status: symbols.get(status, "")},
    )

    assert len(df) > 0, "Empty CSV file"

    # Join status symbols by service ordered by time
    df = (
        df.sort_values(by=["id", "ctime"])[["id", "status"]]
        .groupby("id")
        .agg({"status": lambda status: "".join(status)})
        .query("status.str.len() >= @min_element")
    )

    return df.to_dict()["status"]


@app.command()
def distance(
    input: Annotated[str, typer.Argument(help="Input CSV file.")],
    id: Annotated[str, typer.Argument(help="ID to compare.")],
    compare_names: Annotated[
        bool,
        typer.Option(
            "--compare-names",
            help="Compare names instead of status sequencess.",
            show_default=False,
        ),
    ] = False,
    threshold: Annotated[
        float, typer.Option("--threshold", "-t", help="Distance threshold.")
    ] = float("inf"),
    algorithm: Annotated[
        Algorithm,
        typer.Option("--algorithm", "-a", help="Distance algorithm."),
    ] = Algorithm.LEVENSHTEIN,
):
    """
    Compute pairwise distances for a given ID and return other IDs below a threshold,
    sorted by closest to furthest.
    """
    try:
        service_data = read_csv(input)
    except AssertionError as e:
        typer.echo(e)
        raise typer.Exit(code=1)

    if id not in service_data:
        typer.echo(f"ID {id} not found in the input file.")
        raise typer.Exit(code=1)

    if compare_names:
        service_data = {k: k for k in service_data.keys()}

    try:
        result_ids = compute_distance_matrix(service_data, algorithm, id)
        assert isinstance(result_ids, dict)
    except AssertionError as e:
        typer.echo(e)
        raise typer.Exit(code=1)

    typer.echo(f"Distances for ID {id} above {threshold} using {algorithm.value}:\n")
    result_ids = {k: v for k, v in result_ids.items() if v <= threshold}
    max_length = max(len(name) for name in result_ids.keys())

    for name, distance in result_ids.items():
        typer.echo(f"{name:<{max_length}}: {distance}")


@app.command()
def cluster(
    input: Annotated[str, typer.Argument(help="Input CSV file.")],
    algorithm: Annotated[
        Algorithm, typer.Option(help="Distance algorithm.")
    ] = Algorithm.LEVENSHTEIN,
    compare_names: Annotated[
        bool,
        typer.Option(
            "--compare-names",
            help="Compare names instead of status sequencess.",
            show_default=False,
        ),
    ] = False,
    eps: Annotated[float, typer.Option(help="DBSCAN eps value.")] = 3,
    min_samples: Annotated[int, typer.Option(help="DBSCAN min_samples value.")] = 2,
):
    """
    Perform clustering using DBSCAN based on the chosen distance algorithm.
    """
    try:
        service_data = read_csv(input)
    except AssertionError as e:
        typer.echo(e)
        raise typer.Exit(code=1)

    if compare_names:
        service_data = {k: k for k in service_data.keys()}

    distance_matrix = compute_distance_matrix(service_data, algorithm)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = dbscan.fit_predict(distance_matrix)

    for cluster in np.unique(labels):
        if cluster == -1:
            n = len([label for label in labels if label == -1])
            typer.echo(f"Cluster {cluster}: {n} services with no cluster\n")
            continue
        typer.echo(f"Cluster {cluster}:")
        for i, k in enumerate(service_data.keys()):
            if labels[i] == cluster:
                typer.echo(f"  {k}")
        typer.echo("")


if __name__ == "__main__":
    app()
