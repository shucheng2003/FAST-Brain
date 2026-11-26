import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .config import project_root


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def read_records(root):
    records = []
    for path in root.rglob("metrics.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["dataset"], record["variant"])].append(record)
    rows = []
    for (dataset, variant), vals in sorted(groups.items()):
        row = {"dataset": dataset, "variant": variant, "runs": len(vals)}
        numeric = sorted({k for v in vals for k, val in v.items() if isinstance(val, (int, float)) and k != "seed"})
        for key in numeric:
            arr = np.array([v[key] for v in vals if key in v and v[key] is not None], dtype=float)
            if arr.size:
                row[f"{key}_mean"] = float(arr.mean())
                row[f"{key}_std"] = float(arr.std(ddof=0))
        rows.append(row)
    return rows


def main():
    args = parse_args()
    root = project_root() / args.results_dir
    rows = summarize(read_records(root))
    out = root / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
