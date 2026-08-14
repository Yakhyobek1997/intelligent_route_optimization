"""Professional result visualization for route optimization experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .ant_colony import Point, Solution


BACKGROUND = "#f8fafc"
PANEL = "#ffffff"
TEXT = "#0f172a"
MUTED = "#64748b"
GRID = "#cbd5e1"
ACCENT = "#7c3aed"
SUCCESS = "#059669"
ROUTE_COLORS = ["#06b6d4", "#2563eb", "#7c3aed", "#db2777"]


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(color=GRID, alpha=0.45, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, labelsize=8.5, length=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)


def _draw_route(
    figure: plt.Figure,
    axis: plt.Axes,
    cities: Sequence[Point],
    solution: Solution,
) -> None:
    route_cmap = LinearSegmentedColormap.from_list("route_progress", ROUTE_COLORS)
    closed_tour = solution.tour + [solution.tour[0]]
    route_points = [cities[index] for index in closed_tour]
    route_segments = [
        [route_points[index], route_points[index + 1]]
        for index in range(len(route_points) - 1)
    ]
    progress = list(range(len(route_segments)))
    route_norm = Normalize(vmin=0, vmax=max(1, len(route_segments) - 1))

    colored_route = LineCollection(
        route_segments,
        cmap=route_cmap,
        norm=route_norm,
        linewidths=2.4,
        alpha=0.9,
        zorder=2,
    )
    colored_route.set_array(progress)
    axis.add_collection(colored_route)

    city_x = [point[0] for point in route_points[:-1]]
    city_y = [point[1] for point in route_points[:-1]]
    axis.scatter(
        city_x,
        city_y,
        c=progress,
        cmap=route_cmap,
        norm=route_norm,
        s=58,
        edgecolors="white",
        linewidths=1.3,
        zorder=4,
    )

    start_city = solution.tour[0]
    start_x, start_y = cities[start_city]
    axis.scatter(
        [start_x],
        [start_y],
        marker="*",
        s=260,
        color="#f59e0b",
        edgecolors="white",
        linewidths=1.6,
        label=f"Start / finish: city {start_city}",
        zorder=6,
    )

    for city_index in solution.tour:
        x_coordinate, y_coordinate = cities[city_index]
        axis.annotate(
            str(city_index),
            (x_coordinate, y_coordinate),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            fontweight="semibold",
            color=TEXT,
            zorder=7,
        )

    arrow_step = max(1, len(route_segments) // 8)
    for index in range(0, len(route_segments), arrow_step):
        (first_x, first_y), (second_x, second_y) = route_segments[index]
        arrow_start = (
            first_x + (second_x - first_x) * 0.42,
            first_y + (second_y - first_y) * 0.42,
        )
        arrow_end = (
            first_x + (second_x - first_x) * 0.60,
            first_y + (second_y - first_y) * 0.60,
        )
        axis.annotate(
            "",
            xy=arrow_end,
            xytext=arrow_start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": route_cmap(route_norm(index)),
                "lw": 1.8,
                "mutation_scale": 11,
            },
            zorder=3,
        )

    axis.set_title(
        "Best route",
        loc="left",
        pad=14,
        fontsize=14,
        fontweight="bold",
        color=TEXT,
    )
    axis.text(
        1.0,
        1.035,
        f"Total distance  {solution.length:,.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="semibold",
        color=ACCENT,
    )
    axis.set_xlabel("X coordinate", color=MUTED, labelpad=8)
    axis.set_ylabel("Y coordinate", color=MUTED, labelpad=8)
    axis.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor=GRID,
        fontsize=8.5,
    )
    axis.margins(0.08)

    colorbar = figure.colorbar(
        colored_route,
        ax=axis,
        orientation="horizontal",
        fraction=0.045,
        pad=0.12,
        aspect=35,
    )
    colorbar.set_label("Route progress", color=MUTED, fontsize=8.5)
    colorbar.set_ticks([0, max(1, len(route_segments) - 1)])
    colorbar.set_ticklabels(["Start", "Finish"])
    colorbar.ax.tick_params(labelsize=8, colors=MUTED, length=0)
    colorbar.outline.set_visible(False)


def _draw_convergence(axis: plt.Axes, solution: Solution) -> None:
    iterations = list(range(1, len(solution.history) + 1))
    history_min = min(solution.history)
    history_max = max(solution.history)
    history_span = max(1.0, history_max - history_min)
    chart_floor = history_min - history_span * 0.08

    axis.fill_between(
        iterations,
        solution.history,
        chart_floor,
        step="post",
        color="#ddd6fe",
        alpha=0.55,
        zorder=1,
    )
    axis.plot(
        iterations,
        solution.history,
        drawstyle="steps-post",
        color=ACCENT,
        linewidth=2.3,
        zorder=3,
    )

    improvement_indexes = [
        index
        for index in range(1, len(solution.history))
        if solution.history[index] < solution.history[index - 1]
    ]
    axis.scatter(
        [iterations[index] for index in improvement_indexes],
        [solution.history[index] for index in improvement_indexes],
        s=28,
        color="#ec4899",
        edgecolors="white",
        linewidths=0.8,
        label=f"{len(improvement_indexes)} improvements",
        zorder=4,
    )
    axis.scatter(
        [iterations[-1]],
        [solution.history[-1]],
        s=85,
        color=SUCCESS,
        edgecolors="white",
        linewidths=1.5,
        zorder=5,
    )

    axis.annotate(
        f"Start  {solution.history[0]:,.1f}",
        xy=(iterations[0], solution.history[0]),
        xytext=(14, -6),
        textcoords="offset points",
        fontsize=9,
        color=MUTED,
    )
    axis.annotate(
        f"Best  {solution.history[-1]:,.3f}",
        xy=(iterations[-1], solution.history[-1]),
        xytext=(-12, 16),
        textcoords="offset points",
        ha="right",
        fontsize=9.5,
        fontweight="bold",
        color=SUCCESS,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#ecfdf5",
            "edgecolor": "#a7f3d0",
        },
        arrowprops={"arrowstyle": "->", "color": SUCCESS, "lw": 1.2},
    )

    axis.set_title(
        "Convergence",
        loc="left",
        pad=14,
        fontsize=14,
        fontweight="bold",
        color=TEXT,
    )
    axis.text(
        1.0,
        1.035,
        f"{len(solution.history):,} iterations",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=MUTED,
    )
    axis.set_xlabel("Iteration", color=MUTED, labelpad=8)
    axis.set_ylabel("Best route length", color=MUTED, labelpad=8)
    axis.set_ylim(chart_floor, history_max + history_span * 0.12)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=MUTED)


def save_result_plot(
    cities: Sequence[Point], solution: Solution, output_path: Path
) -> None:
    """Save a portfolio-ready route and convergence dashboard."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(15, 6.6), facecolor=BACKGROUND)
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.08, 0.92),
        left=0.055,
        right=0.975,
        bottom=0.13,
        top=0.82,
        wspace=0.18,
    )
    route_axis = figure.add_subplot(grid[0, 0], facecolor=PANEL)
    convergence_axis = figure.add_subplot(grid[0, 1], facecolor=PANEL)

    figure.suptitle(
        "Intelligent Route Optimization",
        x=0.055,
        y=0.96,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
    )
    figure.text(
        0.055,
        0.905,
        "Ant Colony Optimization result dashboard",
        ha="left",
        fontsize=11.5,
        color=MUTED,
    )

    _draw_route(figure, route_axis, cities, solution)
    _draw_convergence(convergence_axis, solution)
    _style_axis(route_axis)
    _style_axis(convergence_axis)

    improvement = (solution.history[0] - solution.history[-1]) / solution.history[0]
    figure.text(
        0.975,
        0.045,
        (
            f"{len(cities)} cities   •   {len(solution.history)} iterations   •   "
            f"{improvement:.1%} improvement   •   {solution.elapsed_seconds:.3f}s"
        ),
        ha="right",
        fontsize=9,
        color=MUTED,
    )

    figure.savefig(
        output_path,
        dpi=180,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
