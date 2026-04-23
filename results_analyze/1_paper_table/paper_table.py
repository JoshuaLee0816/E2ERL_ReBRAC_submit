"""
paper_table.py
==============
Generate a E2E_ReBRAC big table results from wandb_results_E2EReBRAC.csv. #results_analyze/1_paper_table/data/

Outputs:
  paper_table.csv  
  paper_table.tex   
"""

import csv
import os
import re

# 1. Baseline data (Last Scores) from CORL README

BASELINES = {
    # ── Locomotion ──────────────────────────────────────────────────────────
    "halfcheetah-medium-v2": {
        "BC":       "42.40 ± 0.19",
        "10%BC":    "42.46 ± 0.70",
        "TD3+BC":   "48.10 ± 0.18",
        "AWAC":     "49.46 ± 0.62",
        "CQL":      "47.04 ± 0.22",
        "IQL":      "48.31 ± 0.22",
        "ReBRAC":   "64.04 ± 0.68",
        "SAC-N":    "68.20 ± 1.28",
        "EDAC":     "67.70 ± 1.04",
        "DT":       "42.20 ± 0.26",
    },
    "halfcheetah-medium-replay-v2": {
        "BC":       "35.66 ± 2.33",
        "10%BC":    "23.59 ± 6.95",
        "TD3+BC":   "44.84 ± 0.59",
        "AWAC":     "44.70 ± 0.69",
        "CQL":      "45.04 ± 0.27",
        "IQL":      "44.46 ± 0.22",
        "ReBRAC":   "51.18 ± 0.31",
        "SAC-N":    "60.70 ± 1.01",
        "EDAC":     "62.06 ± 1.10",
        "DT":       "38.91 ± 0.50",
    },
    "halfcheetah-medium-expert-v2": {
        "BC":       "55.95 ± 7.35",
        "10%BC":    "90.10 ± 2.45",
        "TD3+BC":   "90.78 ± 6.04",
        "AWAC":     "93.62 ± 0.41",
        "CQL":      "95.63 ± 0.42",
        "IQL":      "94.74 ± 0.52",
        "ReBRAC":   "103.80 ± 2.95",
        "SAC-N":    "98.96 ± 9.31",
        "EDAC":     "104.76 ± 0.64",
        "DT":       "91.55 ± 0.95",
    },
    "hopper-medium-v2": {
        "BC":       "53.51 ± 1.76",
        "10%BC":    "55.48 ± 7.30",
        "TD3+BC":   "60.37 ± 3.49",
        "AWAC":     "74.45 ± 9.14",
        "CQL":      "59.08 ± 3.77",
        "IQL":      "67.53 ± 3.78",
        "ReBRAC":   "102.29 ± 0.17",
        "SAC-N":    "40.82 ± 9.91",
        "EDAC":     "101.70 ± 0.28",
        "DT":       "65.10 ± 1.61",
    },
    "hopper-medium-replay-v2": {
        "BC":       "29.81 ± 2.07",
        "10%BC":    "70.42 ± 8.66",
        "TD3+BC":   "64.42 ± 21.52",
        "AWAC":     "96.39 ± 5.28",
        "CQL":      "95.11 ± 5.27",
        "IQL":      "97.43 ± 6.39",
        "ReBRAC":   "94.98 ± 6.53",
        "SAC-N":    "100.33 ± 0.78",
        "EDAC":     "99.66 ± 0.81",
        "DT":       "81.77 ± 6.87",
    },
    "hopper-medium-expert-v2": {
        "BC":       "52.30 ± 4.01",
        "10%BC":    "111.16 ± 1.03",
        "TD3+BC":   "101.17 ± 9.07",
        "AWAC":     "52.73 ± 37.47",
        "CQL":      "99.26 ± 10.91",
        "IQL":      "107.42 ± 7.80",
        "ReBRAC":   "109.45 ± 2.34",
        "SAC-N":    "101.31 ± 11.63",
        "EDAC":     "105.19 ± 10.08",
        "DT":       "110.44 ± 0.33",
    },
    "walker2d-medium-v2": {
        "BC":       "63.23 ± 16.24",
        "10%BC":    "67.34 ± 5.17",
        "TD3+BC":   "82.71 ± 4.78",
        "AWAC":     "66.53 ± 26.04",
        "CQL":      "80.75 ± 3.28",
        "IQL":      "80.91 ± 3.17",
        "ReBRAC":   "85.82 ± 0.77",
        "SAC-N":    "87.47 ± 0.66",
        "EDAC":     "93.36 ± 1.38",
        "DT":       "67.63 ± 2.54",
    },
    "walker2d-medium-replay-v2": {
        "BC":       "21.80 ± 10.15",
        "10%BC":    "54.35 ± 6.34",
        "TD3+BC":   "85.62 ± 4.01",
        "AWAC":     "82.20 ± 1.05",
        "CQL":      "73.09 ± 13.22",
        "IQL":      "82.15 ± 3.03",
        "ReBRAC":   "84.25 ± 2.25",
        "SAC-N":    "78.99 ± 0.50",
        "EDAC":     "87.10 ± 2.78",
        "DT":       "59.86 ± 2.73",
    },
    "walker2d-medium-expert-v2": {
        "BC":       "98.96 ± 15.98",
        "10%BC":    "108.70 ± 0.25",
        "TD3+BC":   "110.03 ± 0.36",
        "AWAC":     "49.41 ± 38.16",
        "CQL":      "109.56 ± 0.39",
        "IQL":      "111.72 ± 0.86",
        "ReBRAC":   "111.86 ± 0.43",
        "SAC-N":    "114.93 ± 0.41",
        "EDAC":     "114.75 ± 0.74",
        "DT":       "107.11 ± 0.96",
    },
    # ── AntMaze ─────────────────────────────────────────────────────────────
    "antmaze-umaze-v2": {
        "BC":       "55.25 ± 4.15",
        "10%BC":    "65.75 ± 5.26",
        "TD3+BC":   "70.75 ± 39.18",
        "AWAC":     "57.75 ± 10.28",
        "CQL":      "92.75 ± 1.92",
        "IQL":      "77.00 ± 5.52",
        "ReBRAC":   "97.75 ± 1.48",
        "SAC-N":    "0.00 ± 0.00",
        "EDAC":     "0.00 ± 0.00",
        "DT":       "57.00 ± 9.82",
    },
    "antmaze-umaze-diverse-v2": {
        "BC":       "47.25 ± 4.09",
        "10%BC":    "44.00 ± 1.00",
        "TD3+BC":   "44.75 ± 11.61",
        "AWAC":     "58.00 ± 7.68",
        "CQL":      "37.25 ± 3.70",
        "IQL":      "54.25 ± 5.54",
        "ReBRAC":   "83.50 ± 7.02",
        "SAC-N":    "0.00 ± 0.00",
        "EDAC":     "0.00 ± 0.00",
        "DT":       "51.75 ± 0.43",
    },
    "antmaze-medium-play-v2": {
        "BC":       "0.00 ± 0.00",
        "10%BC":    "2.00 ± 0.71",
        "TD3+BC":   "0.25 ± 0.43",
        "AWAC":     "0.00 ± 0.00",
        "CQL":      "65.75 ± 11.61",
        "IQL":      "65.75 ± 11.71",
        "ReBRAC":   "89.50 ± 3.35",
        "SAC-N":    "0.00 ± 0.00",
        "EDAC":     "0.00 ± 0.00",
        "DT":       "0.00 ± 0.00",
    },
    "antmaze-medium-diverse-v2": {
        "BC":       "0.75 ± 0.83",
        "10%BC":    "5.75 ± 9.39",
        "TD3+BC":   "0.25 ± 0.43",
        "AWAC":     "0.00 ± 0.00",
        "CQL":      "67.25 ± 3.56",
        "IQL":      "73.75 ± 5.45",
        "ReBRAC":   "83.50 ± 8.20",
        "SAC-N":    "0.00 ± 0.00",
        "EDAC":     "0.00 ± 0.00",
        "DT":       "0.00 ± 0.00",
    },
    "antmaze-large-play-v2": {
        "BC":       "0.00 ± 0.00",
        "10%BC":    "0.00 ± 0.00",
        "TD3+BC":   "0.00 ± 0.00",
        "AWAC":     "0.00 ± 0.00",
        "CQL":      "20.75 ± 7.26",
        "IQL":      "42.00 ± 4.53",
        "ReBRAC":   "52.25 ± 29.01",
        "SAC-N":    "0.00 ± 0.00",
        "EDAC":     "0.00 ± 0.00",
        "DT":       "0.00 ± 0.00",
    },
    "antmaze-large-diverse-v2": {
        "BC":       "0.00 ± 0.00",
        "10%BC":    "0.75 ± 0.83",
        "TD3+BC":   "0.00 ± 0.00",
        "AWAC":     "0.00 ± 0.00",
        "CQL":      "20.50 ± 13.24",
        "IQL":      "30.25 ± 3.63",
        "ReBRAC":   "64.00 ± 5.43",
        "SAC-N":    "0.00 ± 0.00",
        "EDAC":     "0.00 ± 0.00",
        "DT":       "0.00 ± 0.00",
    },
    # ── Maze2D ──────────────────────────────────────────────────────────────
    "maze2d-umaze-v1": {
        "BC":       "0.36 ± 8.69",
        "10%BC":    "12.18 ± 4.29",
        "TD3+BC":   "29.41 ± 12.31",
        "AWAC":     "82.67 ± 28.30",
        "CQL":      "-8.90 ± 6.11",
        "IQL":      "42.11 ± 0.58",
        "ReBRAC":   "106.87 ± 22.16",
        "SAC-N":    "130.59 ± 16.52",
        "EDAC":     "95.26 ± 6.39",
        "DT":       "18.08 ± 25.42",
    },
    "maze2d-medium-v1": {
        "BC":       "0.79 ± 3.25",
        "10%BC":    "14.25 ± 2.33",
        "TD3+BC":   "59.45 ± 36.25",
        "AWAC":     "52.88 ± 55.12",
        "CQL":      "86.11 ± 9.68",
        "IQL":      "34.85 ± 2.72",
        "ReBRAC":   "105.11 ± 31.67",
        "SAC-N":    "88.61 ± 18.72",
        "EDAC":     "57.04 ± 3.45",
        "DT":       "31.71 ± 26.33",
    },
    "maze2d-large-v1": {
        "BC":       "2.26 ± 4.39",
        "10%BC":    "11.32 ± 5.10",
        "TD3+BC":   "97.10 ± 25.41",
        "AWAC":     "209.13 ± 8.19",
        "CQL":      "23.75 ± 36.70",
        "IQL":      "61.72 ± 3.50",
        "ReBRAC":   "78.33 ± 61.77",
        "SAC-N":    "204.76 ± 1.19",
        "EDAC":     "95.60 ± 22.92",
        "DT":       "35.66 ± 28.20",
    },
    # ── Adroit ──────────────────────────────────────────────────────────────
    "pen-human-v1": {
        "BC":       "71.03 ± 6.26",
        "10%BC":    "26.99 ± 9.60",
        "TD3+BC":   "-3.88 ± 0.21",
        "AWAC":     "81.12 ± 13.47",
        "CQL":      "13.71 ± 16.98",
        "IQL":      "78.49 ± 8.21",
        "ReBRAC":   "103.16 ± 8.49",
        "SAC-N":    "6.86 ± 5.93",
        "EDAC":     "5.07 ± 6.16",
        "DT":       "67.68 ± 5.48",
    },
    "pen-cloned-v1": {
        "BC":       "51.92 ± 15.15",
        "10%BC":    "46.67 ± 14.25",
        "TD3+BC":   "5.13 ± 5.28",
        "AWAC":     "89.56 ± 15.57",
        "CQL":      "1.04 ± 6.62",
        "IQL":      "83.42 ± 8.19",
        "ReBRAC":   "102.79 ± 7.84",
        "SAC-N":    "31.35 ± 2.14",
        "EDAC":     "12.02 ± 1.75",
        "DT":       "64.43 ± 1.43",
    },
    "pen-expert-v1": {
        "BC":       "109.65 ± 7.28",
        "10%BC":    "114.96 ± 2.96",
        "TD3+BC":   "122.53 ± 21.27",
        "AWAC":     "160.37 ± 1.21",
        "CQL":      "-1.41 ± 2.34",
        "IQL":      "128.05 ± 9.21",
        "ReBRAC":   "152.16 ± 6.33",
        "SAC-N":    "87.11 ± 48.95",
        "EDAC":     "-1.55 ± 0.81",
        "DT":       "116.38 ± 1.27",
    },
    "hammer-human-v1": {
        "BC":       "3.03 ± 3.39",
        "10%BC":    "-0.19 ± 0.02",
        "TD3+BC":   "1.02 ± 0.24",
        "AWAC":     "3.37 ± 1.93",
        "CQL":      "0.14 ± 0.11",
        "IQL":      "1.79 ± 0.80",
        "ReBRAC":   "0.24 ± 0.24",
        "SAC-N":    "0.24 ± 0.00",
        "EDAC":     "0.28 ± 0.18",
        "DT":       "1.28 ± 0.15",
    },
    "hammer-cloned-v1": {
        "BC":       "0.55 ± 0.16",
        "10%BC":    "0.12 ± 0.08",
        "TD3+BC":   "0.25 ± 0.01",
        "AWAC":     "0.21 ± 0.24",
        "CQL":      "0.30 ± 0.01",
        "IQL":      "1.50 ± 0.69",
        "ReBRAC":   "5.00 ± 3.75",
        "SAC-N":    "0.14 ± 0.09",
        "EDAC":     "0.19 ± 0.07",
        "DT":       "1.82 ± 0.55",
    },
    "hammer-expert-v1": {
        "BC":       "126.78 ± 0.64",
        "10%BC":    "121.75 ± 7.67",
        "TD3+BC":   "3.11 ± 0.03",
        "AWAC":     "127.06 ± 0.29",
        "CQL":      "0.26 ± 0.01",
        "IQL":      "128.68 ± 0.33",
        "ReBRAC":   "133.62 ± 0.27",
        "SAC-N":    "25.13 ± 43.25",
        "EDAC":     "28.52 ± 49.00",
        "DT":       "117.45 ± 6.65",
    },
    "door-human-v1": {
        "BC":       "2.34 ± 4.00",
        "10%BC":    "-0.13 ± 0.07",
        "TD3+BC":   "-0.33 ± 0.01",
        "AWAC":     "4.60 ± 1.90",
        "CQL":      "5.53 ± 1.31",
        "IQL":      "3.26 ± 1.83",
        "ReBRAC":   "-0.10 ± 0.01",
        "SAC-N":    "-0.38 ± 0.00",
        "EDAC":     "-0.12 ± 0.13",
        "DT":       "4.44 ± 0.87",
    },
    "door-cloned-v1": {
        "BC":       "-0.09 ± 0.03",
        "10%BC":    "0.29 ± 0.59",
        "TD3+BC":   "-0.34 ± 0.01",
        "AWAC":     "0.93 ± 1.66",
        "CQL":      "-0.33 ± 0.01",
        "IQL":      "3.07 ± 1.75",
        "ReBRAC":   "0.06 ± 0.05",
        "SAC-N":    "-0.33 ± 0.00",
        "EDAC":     "2.66 ± 2.31",
        "DT":       "7.64 ± 3.26",
    },
    "door-expert-v1": {
        "BC":       "105.35 ± 0.09",
        "10%BC":    "104.04 ± 1.46",
        "TD3+BC":   "-0.33 ± 0.01",
        "AWAC":     "104.85 ± 0.24",
        "CQL":      "-0.32 ± 0.02",
        "IQL":      "106.65 ± 0.25",
        "ReBRAC":   "106.37 ± 0.29",
        "SAC-N":    "-0.33 ± 0.00",
        "EDAC":     "106.29 ± 1.73",
        "DT":       "104.87 ± 0.39",
    },
    "relocate-human-v1": {
        "BC":       "0.04 ± 0.03",
        "10%BC":    "-0.14 ± 0.08",
        "TD3+BC":   "-0.29 ± 0.01",
        "AWAC":     "0.05 ± 0.03",
        "CQL":      "0.06 ± 0.03",
        "IQL":      "0.12 ± 0.04",
        "ReBRAC":   "0.16 ± 0.30",
        "SAC-N":    "-0.31 ± 0.01",
        "EDAC":     "-0.17 ± 0.17",
        "DT":       "0.05 ± 0.01",
    },
    "relocate-cloned-v1": {
        "BC":       "-0.06 ± 0.01",
        "10%BC":    "-0.00 ± 0.02",
        "TD3+BC":   "-0.30 ± 0.01",
        "AWAC":     "-0.04 ± 0.04",
        "CQL":      "-0.29 ± 0.01",
        "IQL":      "0.04 ± 0.01",
        "ReBRAC":   "1.66 ± 2.59",
        "SAC-N":    "-0.01 ± 0.10",
        "EDAC":     "0.17 ± 0.35",
        "DT":       "0.16 ± 0.09",
    },
    "relocate-expert-v1": {
        "BC":       "107.58 ± 1.20",
        "10%BC":    "97.90 ± 5.21",
        "TD3+BC":   "-1.73 ± 0.96",
        "AWAC":     "108.87 ± 0.85",
        "CQL":      "-0.30 ± 0.02",
        "IQL":      "106.11 ± 4.02",
        "ReBRAC":   "107.52 ± 2.28",
        "SAC-N":    "-0.36 ± 0.00",
        "EDAC":     "71.94 ± 18.37",
        "DT":       "104.28 ± 0.42",
    },
}

# Canonical display order of environments grouped by domain
DOMAIN_ENVS = {
    "Locomotion": [
        "halfcheetah-medium-v2",
        "halfcheetah-medium-replay-v2",
        "halfcheetah-medium-expert-v2",
        "hopper-medium-v2",
        "hopper-medium-replay-v2",
        "hopper-medium-expert-v2",
        "walker2d-medium-v2",
        "walker2d-medium-replay-v2",
        "walker2d-medium-expert-v2",
    ],
    "Maze2D": [
        "maze2d-umaze-v1",
        "maze2d-medium-v1",
        "maze2d-large-v1",
    ],
    "AntMaze": [
        "antmaze-umaze-v2",
        "antmaze-umaze-diverse-v2",
        "antmaze-medium-play-v2",
        "antmaze-medium-diverse-v2",
        "antmaze-large-play-v2",
        "antmaze-large-diverse-v2",
    ],
    "Adroit": [
        "pen-human-v1",
        "pen-cloned-v1",
        "pen-expert-v1",
        "door-human-v1",
        "door-cloned-v1",
        "door-expert-v1",
        "hammer-human-v1",
        "hammer-cloned-v1",
        "hammer-expert-v1",
        "relocate-human-v1",
        "relocate-cloned-v1",
        "relocate-expert-v1",
    ],
}

BASELINE_COLS = ["BC", "10%BC", "TD3+BC", "AWAC", "CQL", "IQL", "ReBRAC", "SAC-N", "EDAC", "DT"]
E2ERL_COLS    = ["E2ERL(β=0.001)", "E2ERL(β=0.0025)", "E2ERL(β=0.005)", "E2ERL(β=0.0075)", "E2ERL(β=0.01)", "E2ERL(β=0.05)", "E2ERL(β=0.1)", "E2ERL(β=0.5)"]
"""
You can adjust the E2ERL_COLS list if you want to include fewer or more β values in the table.
"""


# 2. Load E2ERL results

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
E2ERL_CSV     = os.path.join(SCRIPT_DIR, "data", "wandb_results_E2EReBRAC.csv")
OUTPUT_CSV    = os.path.join(SCRIPT_DIR, "outputs", "E2EReBRAC_table.csv")
OUTPUT_TEX    = os.path.join(SCRIPT_DIR, "outputs", "E2EReBRAC_table.tex")

def load_e2erl(path: str) -> dict:
    """Return dict: env -> {beta_col: 'mean ± std'}"""
    results = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            env = row["Environment"]
            results[env] = {
                "E2ERL(β=0.001)": row.get("Mean±Std(β=0.001)", "N/A").strip() or "N/A",
                "E2ERL(β=0.0025)": row.get("Mean±Std(β=0.0025)", "N/A").strip() or "N/A",
                "E2ERL(β=0.005)": row.get("Mean±Std(β=0.005)", "N/A").strip() or "N/A",
                "E2ERL(β=0.0075)": row.get("Mean±Std(β=0.0075)", "N/A").strip() or "N/A",
                "E2ERL(β=0.01)":  row.get("Mean±Std(β=0.01)",  "N/A").strip() or "N/A",
                "E2ERL(β=0.05)":  row.get("Mean±Std(β=0.05)",  "N/A").strip() or "N/A",
                "E2ERL(β=0.1)":   row.get("Mean±Std(β=0.1)",   "N/A").strip() or "N/A",
                "E2ERL(β=0.5)":   row.get("Mean±Std(β=0.5)",   "N/A").strip() or "N/A",
            }
    return results

# 3. Compute per-domain averages (mean only, ignoring N/A)

def mean_only(val: str) -> float | None:
    """Extract the mean from 'mean ± std' string."""
    if val.strip() == "N/A" or val.strip() == "":
        return None
    m = re.match(r"^\s*(-?[\d.]+)", val)
    return float(m.group(1)) if m else None

def domain_avg(rows: list[dict], cols: list[str]) -> dict:
    sums   = {c: 0.0 for c in cols}
    counts = {c: 0   for c in cols}
    for r in rows:
        for c in cols:
            v = mean_only(r.get(c, "N/A"))
            if v is not None:
                sums[c]   += v
                counts[c] += 1
    avgs = {}
    for c in cols:
        avgs[c] = f"{sums[c] / counts[c]:.2f}" if counts[c] > 0 else "N/A"
    return avgs


# 4. Build row data (shared between CSV and LaTeX writers)

def build_rows(e2erl: dict) -> list[tuple[str, list[dict]]]:
    """Return list of (domain, rows) where each row is a plain dict."""
    result = []
    for domain, envs in DOMAIN_ENVS.items():
        domain_rows = []
        for env in envs:
            row = {"Task-Name": env}
            for col in BASELINE_COLS:
                row[col] = BASELINES.get(env, {}).get(col, "N/A")
            e_data = e2erl.get(env, {})
            for col in E2ERL_COLS:
                row[col] = e_data.get(col, "N/A")
            domain_rows.append(row)
        result.append((domain, domain_rows))
    return result


# 5. CSV writer

def write_csv(domain_data: list[tuple[str, list[dict]]]):
    all_cols = ["Task-Name"] + BASELINE_COLS + E2ERL_COLS
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        for domain, rows in domain_data:
            for row in rows:
                writer.writerow(row)
            avg = domain_avg(rows, BASELINE_COLS + E2ERL_COLS)
            avg_row = {"Task-Name": f"[{domain} average]"}
            avg_row.update(avg)
            writer.writerow(avg_row)
            writer.writerow({c: "" for c in all_cols})
    print(f"CSV saved  → {OUTPUT_CSV}")


# 6. LaTeX writer

def latex_cell(val: str, highlight: bool) -> str:
    """Format a table cell value; wrap in red if highlight=True."""
    safe = val.replace("±", r"$\pm$")   # convert ± to math mode
    if highlight:
        return r"\textcolor{red}{" + safe + r"}"
    return safe


def write_latex(domain_data: list[tuple[str, list[dict]]]):
    all_data_cols = BASELINE_COLS + E2ERL_COLS
    n_cols = 1 + len(all_data_cols)          # Task-Name + data

    # column spec: l for task name, c for each value
    col_spec = "l" + "c" * len(all_data_cols)

    header_cells = ["Task-Name"] + [c.replace("%", r"\%") for c in BASELINE_COLS + E2ERL_COLS]

    lines = []
    lines.append(r"\documentclass[a4paper]{article}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{xcolor}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{geometry}")
    lines.append(r"\geometry{margin=1cm}")
    lines.append(r"\begin{document}")
    lines.append(r"")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Offline RL benchmark results (Last Scores).}")
    lines.append(r"  \label{tab:offline_results}")
    lines.append(r"  \resizebox{\textwidth}{!}{")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")
    lines.append("    " + " & ".join(header_cells) + r" \\")
    lines.append(r"    \midrule")

    for domain, rows in domain_data:
        # domain separator comment
        lines.append(f"    % ── {domain} ──")

        for row in rows:
            rebrac_mean = mean_only(row.get("ReBRAC", "N/A"))

            cells = [row["Task-Name"].replace("_", r"\_")]
            for col in all_data_cols:
                val = row.get(col, "N/A")
                # highlight only E2ERL columns that beat ReBRAC
                if col in E2ERL_COLS and rebrac_mean is not None:
                    e_mean = mean_only(val)
                    highlight = (e_mean is not None) and (e_mean > rebrac_mean)
                else:
                    highlight = False
                cells.append(latex_cell(val, highlight))

            lines.append("    " + " & ".join(cells) + r" \\")

        # average row
        avg = domain_avg(rows, all_data_cols)
        avg_cells = [f"\\textit{{{domain} avg}}"]
        for col in all_data_cols:
            avg_cells.append(avg.get(col, "N/A"))
        lines.append(r"    \midrule")
        lines.append("    " + " & ".join(avg_cells) + r" \\")
        lines.append(r"    \midrule")

    # remove the last extra \midrule and replace with \bottomrule
    lines[-1] = r"    \bottomrule"

    lines.append(r"  \end{tabular}")
    lines.append(r"  }")   # close resizebox
    lines.append(r"\end{table*}")
    lines.append(r"")
    lines.append(r"\end{document}")

    os.makedirs(os.path.dirname(OUTPUT_TEX), exist_ok=True)
    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX saved → {OUTPUT_TEX}")


def main():
    e2erl       = load_e2erl(E2ERL_CSV)
    domain_data = build_rows(e2erl)
    write_csv(domain_data)
    write_latex(domain_data)

if __name__ == "__main__":
    main()
