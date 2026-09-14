"""Saving and loading run results.

Every run writes a CSV. This matters for two reasons. The runs take time, so we do not
want to repeat them every time a plot needs adjusting. And a CSV of the raw per tick
numbers is the evidence behind the figures, which is worth having in the repository.
"""

from __future__ import annotations

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

COLUMNS = [
    "tick", "population", "births", "deaths",
    "mean_energy", "mean_age", "mean_neighbours", "density", "food", "nests_free",
    "died_starving", "died_old", "died_neglected", "dependents", "mean_impairment",
]


def save_history(result: dict, results_dir: Path | None = None) -> Path:
    """Write one run's per tick history to results/<name>_history.csv."""
    directory = Path(results_dir or RESULTS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result['name']}_history.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in result["history"]:
            writer.writerow({k: row[k] for k in COLUMNS})
    return path


def load_history(name: str, results_dir: Path | None = None) -> list[dict]:
    """Read back a saved history, converting the numbers out of their text form."""
    directory = Path(results_dir or RESULTS_DIR)
    path = directory / f"{name}_history.csv"
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                integer = key in (
                "tick", "population", "births", "deaths", "food", "nests_free",
                "died_starving", "died_old", "died_neglected", "dependents",
            )
            row[key] = int(value) if integer else float(value)
            rows.append(row)
    return rows


def summarise(result: dict) -> dict:
    """A few headline numbers describing one run.

    peak_population and the tick it happened at are the pair that matter for the research
    question. A population that peaks early and then falls is the collapse pattern we are
    looking for. A population that rises and stays flat is not.
    """
    history = result["history"]
    if not history:
        return {}
    peak = max(history, key=lambda r: r["population"])
    final = history[-1]
    late = history[len(history) // 2:]
    return {
        "name": result["name"],
        "ticks_run": len(history),
        "peak_population": peak["population"],
        "peak_tick": peak["tick"],
        "final_population": final["population"],
        "collapse_ratio": final["population"] / peak["population"] if peak["population"] else 0.0,
        "mean_neighbours_late": sum(r["mean_neighbours"] for r in late) / len(late),
        "total_births": sum(r["births"] for r in history),
        "total_deaths": sum(r["deaths"] for r in history),
        "died_starving": sum(r.get("died_starving", 0) for r in history),
        "died_old": sum(r.get("died_old", 0) for r in history),
        "died_neglected": sum(r.get("died_neglected", 0) for r in history),
        "extinct_at": result.get("extinct_at"),
    }
