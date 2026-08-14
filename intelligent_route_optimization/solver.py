"""Command-line orchestration for Intelligent Route Optimization."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .ant_colony import (
    DEFAULT_OUTPUT,
    PROJECT_ROOT,
    LinearPheromoneACO,
    Solution,
    load_cities,
    parse_arguments,
)
from .visualization import save_result_plot


DEFAULT_RESULT = PROJECT_ROOT / "results" / "route_optimization_result.png"
__all__ = ["LinearPheromoneACO", "Solution", "load_cities", "main"]


def main() -> None:
    arguments = parse_arguments()
    cities = load_cities(arguments.data.resolve())
    solver = LinearPheromoneACO(
        cities,
        ant_count=arguments.ants,
        seed=arguments.seed,
    )
    solution = solver.solve(arguments.iterations)

    requested_output = arguments.output.resolve()
    output_path = (
        DEFAULT_RESULT.resolve()
        if requested_output == DEFAULT_OUTPUT.resolve()
        else requested_output
    )
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
