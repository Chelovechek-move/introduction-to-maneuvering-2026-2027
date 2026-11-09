#!/usr/bin/env python3
"""Reproduce DiPrinzio's weights for the vector error model, §§11.1.5–6.

Run from the course root: python3 8/images/maneuver_weights.py
The initial required velocity vector has unit magnitude. Each maneuver has
a 3 degree pointing error and zero magnitude error. After each maneuver the
remaining required vector is repartitioned using the remaining weights.
The cost includes an exact correction after the last imperfect maneuver.
This model does not propagate an orbit or optimize the lecture's GEO transfer.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

BASE = Path(__file__).resolve().parent
ANGLE = np.deg2rad(3.0)
MAGNITUDE_ERROR = 0.0


def sequence(weights):
    remaining = 1.0
    executed = []
    for j, weight in enumerate(weights):
        planned = weight / np.sum(weights[j:]) * remaining
        actual = (1.0 + MAGNITUDE_ERROR) * planned
        # Stable form of the cosine rule, including small residuals.
        remaining = float(np.hypot(
            remaining - actual,
            2.0 * np.sqrt(remaining * actual) * np.sin(ANGLE / 2.0),
        ))
        executed.append(float(actual))
    return float(sum(executed) + remaining), executed, remaining


def main():
    results = {
        "model": "sequential vector correction; no orbital propagation",
        "pointing_error_deg": 3.0,
        "relative_magnitude_error": MAGNITUDE_ERROR,
        "initial_required_vector_magnitude": 1.0,
        "objective": "sum of executed magnitudes plus exact final correction",
        "solver": "SLSQP, sum(weights)=1, weights in [1e-8,1], ftol=1e-13",
        "groups": [],
    }
    reference = {3: [.7052, .2473, .0475],
                 4: [.5848, .2928, .1027, .0197]}
    for count in (3, 4):
        equal = np.ones(count) / count
        solution = minimize(
            lambda w: sequence(w)[0], equal, method="SLSQP",
            bounds=[(1e-8, 1.0)] * count,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            options={"ftol": 1e-13, "maxiter": 1000},
        )
        assert solution.success, solution.message
        assert np.max(np.abs(solution.x - reference[count])) < 5e-5
        cost, executed, correction = sequence(solution.x)
        results["groups"].append({
            "count": count, "weights": solution.x.tolist(),
            "book_rounded_weights": reference[count],
            "cost": cost, "equal_weight_cost": sequence(equal)[0],
            "executed_magnitudes": executed,
            "final_exact_correction": correction,
            "iterations": int(solution.nit),
        })
    path = BASE / "maneuver-weights-results.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
