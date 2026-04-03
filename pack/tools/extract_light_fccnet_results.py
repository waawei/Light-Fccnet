"""Extract only Light-FCCNet run summaries from checkpoint directories.

Usage:
    python -m pack.tools.extract_light_fccnet_results --root /root/checkpoints
    python -m pack.tools.extract_light_fccnet_results --root /root/checkpoints --format csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from .extract_best_results import collect_results, render_table, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Light-FCCNet checkpoint results")
    parser.add_argument("--root", required=True, help="Root directory containing run folders")
    parser.add_argument(
        "--format",
        choices=("table", "csv", "json"),
        default="table",
        help="Output format",
    )
    parser.add_argument("--output", default="", help="Optional output file path")
    return parser.parse_args()


def filter_light_rows(rows):
    return [row for row in rows if row.get("model_name") == "light_fccnet"]


def main() -> None:
    args = parse_args()
    rows = filter_light_rows(collect_results(args.root))

    if args.format == "json":
        content = json.dumps(rows, ensure_ascii=False, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            print(content)
        return

    if args.format == "csv":
        if args.output:
            write_csv(rows, args.output)
        else:
            writer = csv.DictWriter(
                os.sys.stdout,
                fieldnames=["run", "dataset", "variant", "epoch", "best_mae", "best_mse", "best_mape", "model_name", "input_size", "use_p1", "use_p2", "use_p3", "checkpoint"],
            )
            writer.writeheader()
            writer.writerows(rows)
        return

    content = render_table(rows)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        print(content)


if __name__ == "__main__":
    main()
