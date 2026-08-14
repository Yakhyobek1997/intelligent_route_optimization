"""Tests for the current ACO solver."""

import unittest
from pathlib import Path

from intelligent_route_optimization.solver import LinearPheromoneACO, load_cities


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SolverTests(unittest.TestCase):
    def test_included_dataset_is_valid(self) -> None:
        cities = load_cities(PROJECT_ROOT / "data" / "cities_30.txt")
        self.assertEqual(len(cities), 30)

    def test_solver_returns_a_complete_circular_tour(self) -> None:
        cities = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        result = LinearPheromoneACO(cities, ant_count=8, seed=42).solve(20)

        self.assertEqual(len(result.tour), len(cities))
        self.assertEqual(set(result.tour), set(range(len(cities))))
        self.assertEqual(len(result.history), 20)
        self.assertTrue(
            all(
                previous >= current
                for previous, current in zip(result.history, result.history[1:])
            )
        )
        self.assertAlmostEqual(result.length, 4.0)


if __name__ == "__main__":
    unittest.main()
