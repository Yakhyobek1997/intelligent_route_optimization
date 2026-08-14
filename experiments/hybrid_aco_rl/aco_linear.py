"""Ant Colony Optimization for the Traveling Salesman Problem.

This is a Python 3 rewrite of the original 2017 experiment. Pheromone values
are updated linearly after every iteration: first they evaporate, then every
ant deposits an amount inversely proportional to the length of its tour.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


Point = tuple[float, float]


def load_cities(path: Path) -> list[Point]:
    """Load one ``x,y`` city coordinate per line."""
    cities: list[Point] = []
    with path.open("r", encoding="utf-8") as data_file:
        for line_number, raw_line in enumerate(data_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            values = [value.strip() for value in line.split(",")]
            if len(values) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected 'x,y', got {raw_line.rstrip()!r}"
                )

            try:
                cities.append((float(values[0]), float(values[1])))
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: coordinates must be numbers"
                ) from error

    if len(cities) < 2:
        raise ValueError(f"{path}: at least two cities are required")
    if len(set(cities)) != len(cities):
        raise ValueError(f"{path}: duplicate city coordinates are not supported")
    return cities


def euclidean_distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


@dataclass(frozen=True)
class Solution:
    tour: list[int]
    length: float
    history: list[float]
    elapsed_seconds: float


class LinearPheromoneACO:
    """Solve a TSP instance using ant colony optimization."""

    def __init__(
        self,
        cities: Sequence[Point],
        *,
        ant_count: int | None = None,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation: float = 0.5,
        deposit_strength: float = 100.0,
        random_move_probability: float = 0.01,
        seed: int = 42,
    ) -> None:
        if not cities:
            raise ValueError("cities cannot be empty")
        if ant_count is not None and ant_count < 1:
            raise ValueError("ant_count must be at least 1")
        if not 0.0 < evaporation < 1.0:
            raise ValueError("evaporation must be between 0 and 1")
        if not 0.0 <= random_move_probability <= 1.0:
            raise ValueError("random_move_probability must be between 0 and 1")

        self.cities = list(cities)
        self.city_count = len(cities)
        self.ant_count = ant_count or max(1, int(self.city_count * 0.8))
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.deposit_strength = deposit_strength
        self.random_move_probability = random_move_probability
        self.random = random.Random(seed)

        self.distances = [
            [euclidean_distance(first, second) for second in self.cities]
            for first in self.cities
        ]
        self.pheromones = [
            [1.0 for _ in range(self.city_count)] for _ in range(self.city_count)
        ]

    def tour_length(self, tour: Sequence[int]) -> float:
        return sum(
            self.distances[tour[index - 1]][tour[index]]
            for index in range(len(tour))
        )

    def _choose_next_city(self, current: int, unvisited: set[int]) -> int:
        if self.random.random() < self.random_move_probability:
            return self.random.choice(tuple(unvisited))

        choices = sorted(unvisited)
        weights = []
        for city in choices:
            trail = self.pheromones[current][city] ** self.alpha
            heuristic = (1.0 / self.distances[current][city]) ** self.beta
            weights.append(trail * heuristic)

        total = sum(weights)
        if total <= 0.0 or not math.isfinite(total):
            return self.random.choice(choices)
        return self.random.choices(choices, weights=weights, k=1)[0]

    def _build_tour(self) -> list[int]:
        start = self.random.randrange(self.city_count)
        tour = [start]
        unvisited = set(range(self.city_count))
        unvisited.remove(start)

        while unvisited:
            next_city = self._choose_next_city(tour[-1], unvisited)
            tour.append(next_city)
            unvisited.remove(next_city)
        return tour

    def _update_pheromones(self, tours: Sequence[tuple[list[int], float]]) -> None:
        retention = 1.0 - self.evaporation
        minimum_pheromone = 1e-12
        for first in range(self.city_count):
            for second in range(self.city_count):
                self.pheromones[first][second] = max(
                    minimum_pheromone,
                    self.pheromones[first][second] * retention,
                )

        for tour, length in tours:
            deposit = self.deposit_strength / length
            for index in range(len(tour)):
                first = tour[index - 1]
                second = tour[index]
                self.pheromones[first][second] += deposit
                self.pheromones[second][first] += deposit

    def solve(self, iterations: int = 200) -> Solution:
        if iterations < 1:
            raise ValueError("iterations must be at least 1")

        best_tour: list[int] = []
        best_length = math.inf
        history: list[float] = []
        started_at = time.perf_counter()

        for _ in range(iterations):
            tours = []
            for _ in range(self.ant_count):
                tour = self._build_tour()
                length = self.tour_length(tour)
                tours.append((tour, length))
                if length < best_length:
                    best_tour = tour.copy()
                    best_length = length

            self._update_pheromones(tours)
            history.append(best_length)

        return Solution(
            tour=best_tour,
            length=best_length,
            history=history,
            elapsed_seconds=time.perf_counter() - started_at,
        )


def save_result_plot(
    cities: Sequence[Point], solution: Solution, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, (route_axis, convergence_axis) = plt.subplots(1, 2, figsize=(13, 5))
    closed_tour = solution.tour + [solution.tour[0]]
    route_x = [cities[index][0] for index in closed_tour]
    route_y = [cities[index][1] for index in closed_tour]

    route_axis.plot(route_x, route_y, "o-", linewidth=1.2, markersize=4)
    route_axis.set_title(f"Best route (length: {solution.length:.3f})")
    route_axis.set_xlabel("x")
    route_axis.set_ylabel("y")
    route_axis.grid(alpha=0.25)

    convergence_axis.plot(range(1, len(solution.history) + 1), solution.history)
    convergence_axis.set_title("ACO convergence")
    convergence_axis.set_xlabel("Iteration")
    convergence_axis.set_ylabel("Best tour length")
    convergence_axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)


def parse_arguments() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Solve a coordinate-based TSP using Ant Colony Optimization."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=script_directory / "_data1.txt",
        help="input file containing one x,y coordinate per line",
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--ants", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=script_directory / "results" / "aco_linear_result.png",
        help="path where the route and convergence plot will be saved",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also open the result plot in a window",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    cities = load_cities(arguments.data.resolve())
    solver = LinearPheromoneACO(
        cities,
        ant_count=arguments.ants,
        seed=arguments.seed,
    )
    solution = solver.solve(arguments.iterations)
    output_path = arguments.output.resolve()
    save_result_plot(cities, solution, output_path)

    print(f"Cities: {len(cities)}")
    print(f"Ants: {solver.ant_count}")
    print(f"Iterations: {arguments.iterations}")
    print(f"Time: {solution.elapsed_seconds:.3f} seconds")
    print(f"Best tour length: {solution.length:.6f}")
    print("Best tour: " + " -> ".join(map(str, solution.tour + [solution.tour[0]])))
    print(f"Plot saved to: {output_path}")

    if arguments.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
