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
    levenshtein = "levenshtein"
    jacard = "jacard"

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
    if algorithm.value == "levenshtein":
        fun_algorithm = levenshtein_distance
    elif algorithm.value == "jacard":
        fun_algorithm = jacard_distance
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm.value}")

    for i, name1 in enumerate(labels):
        for j, name2 in enumerate(labels):
            status1 = service_data[name1]
            status2 = service_data[name2]
            dist = fun_algorithm(status1, status2)
            distance_matrix[i, j] = dist

    if id is not None:
        index_id = list(service_data.keys()).index(id)
        distances = distance_matrix[index_id]
        keys = sorted(labels, key=lambda x: distances[labels.index(x)])
        distances = sorted(distances)
        return dict(zip(keys, distances))

    return distance_matrix


def read_csv(input: str, min_element: int = 10) -> dict[str, str]:
    df = pd.read_csv(input, names=["ctime", "id", "status"])
    assert len(df) > 0, "Empty CSV file"

    status_map = {
        0: "O",  # OK
        1: "W",  # WARNING
        2: "C",  # CRITICAL
        3: "U",  # UNKNOWN
    }

    service_data: dict[str, str] = {}
    total_service = 0
    for _, row in df.iterrows():
        name = row["id"]
        status = row["status"]
        ctime = row["ctime"]
        if name not in service_data:
            service_data[name] = ""
            total_service += 1
        if status in status_map:
            service_data[name] += status_map[status]
        else:
            print(
                f"Warning: Unrecognized status code '{status}' for service '{name}' at time '{ctime}'"  # noqa E501
            )

    for name in list(service_data.keys()):
        if len(service_data[name]) <= min_element or len(set(service_data[name])) == 1:
            del service_data[name]
            total_service -= 1
    return service_data


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
    ] = Algorithm.levenshtein,
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

    if compare_names:
        service_data = {k: k for k in service_data.keys()}

    target_row = service_data.get(id)
    if target_row is None:
        typer.echo(f"ID {id} not found in the input file.")
        raise typer.Exit(code=1)

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
    ] = Algorithm.levenshtein,
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
