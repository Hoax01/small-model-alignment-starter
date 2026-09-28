from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SFT/DPO training curves from metrics.jsonl files.")
    parser.add_argument("--metrics_path", action="append", required=True, help="Path to a metrics.jsonl file. Can be passed multiple times.")
    parser.add_argument("--output_dir", default="outputs/plots")
    parser.add_argument("--prefix", default="training")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "metric"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    series: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    loaded_rows = 0
    for metrics_path_raw in args.metrics_path:
        metrics_path = Path(metrics_path_raw)
        rows = read_jsonl(metrics_path)
        loaded_rows += len(rows)
        run_name = metrics_path.parent.name or metrics_path.stem
        for row in rows:
            step = row.get("step")
            if not is_number(step):
                continue
            phase = row.get("phase", "metrics")
            label = f"{run_name}:{phase}"
            for key, value in row.items():
                if key in {"step", "epoch"} or not is_number(value):
                    continue
                series[key].append((int(step), float(value), label))

    if not series:
        raise SystemExit(f"No plottable numeric metrics found in {args.metrics_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for metric_name, points in sorted(series.items()):
        by_label: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for step, value, label in points:
            by_label[label].append((step, value))

        plt.figure(figsize=(9, 5))
        for label, label_points in sorted(by_label.items()):
            label_points.sort()
            steps = [point[0] for point in label_points]
            values = [point[1] for point in label_points]
            plt.plot(steps, values, marker="o", linewidth=1.8, markersize=3, label=label)

        plt.xlabel("optimizer step")
        plt.ylabel(metric_name)
        plt.title(metric_name.replace("_", " "))
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        out_path = output_dir / f"{safe_name(args.prefix)}_{safe_name(metric_name)}.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        written.append(str(out_path))

    summary_path = output_dir / f"{safe_name(args.prefix)}_plot_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(
            {
                "metrics_paths": args.metrics_path,
                "rows_loaded": loaded_rows,
                "metrics_plotted": sorted(series),
                "plots": written,
            },
            fout,
            indent=2,
        )

    print(f"Loaded {loaded_rows} metric rows.")
    for path in written:
        print(f"Wrote {path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
