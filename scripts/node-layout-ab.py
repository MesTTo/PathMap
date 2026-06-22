#!/usr/bin/env python3
"""Summarize node-layout benchmark output as A/B comparisons."""

import argparse
import importlib.util
import pathlib
import subprocess
import sys


REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benches"))

from divan_fmt import parse_divan_output  # noqa: E402


def load_table_module():
    table_path = REPO / "scripts" / "node-layout-table.py"
    spec = importlib.util.spec_from_file_location("node_layout_table", table_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {table_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TABLE = load_table_module()


def run_benchmark(sample_count: int) -> str:
    cmd = [
        "cargo",
        "+nightly",
        "bench",
        "--bench",
        "node_layout",
        "--features",
        "counters",
        "--",
        "--sample-count",
        str(sample_count),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def fmt_ns(ns: float) -> str:
    return TABLE.fmt_ns(ns)


def fmt_ratio(ratio: float) -> str:
    if ratio >= 1:
        return f"{ratio:.2f}x faster"
    return f"{1 / ratio:.2f}x slower"


def row_index(rows):
    return {
        (row["op"], row["repr"], row["fixture"], row["case"]): row
        for row in rows
    }


def compare_rows(label: str, base, other):
    base_ns = base["median_ns"]
    other_ns = other["median_ns"]
    if base_ns == 0 or other_ns == 0:
        ratio = 1.0
    else:
        ratio = base_ns / other_ns
    return {
        "label": label,
        "base": base,
        "other": other,
        "ratio": ratio,
        "delta_ns": other_ns - base_ns,
    }


def old_vs_current(rows):
    indexed = row_index(rows)
    comparisons = []
    for row in rows:
        op = row["op"]
        if not op.startswith("old_"):
            continue
        current_op = op.removeprefix("old_")
        current = indexed.get((current_op, row["repr"], row["fixture"], row["case"]))
        if current is None:
            continue
        label = (
            f"{current_op} {row['repr']} {row['fixture']} "
            f"case={row['case']}"
        )
        comparisons.append(compare_rows(label, row, current))
    return sorted(
        comparisons,
        key=lambda item: (
            item["base"]["fixture"],
            item["base"]["repr"],
            item["base"]["op"],
            item["base"]["case"],
        ),
    )


def nested_repr_ab(rows):
    indexed = row_index(rows)
    comparisons = []
    ops = [
        "get",
        "path_exists",
        "update",
        "join",
        "get_or_set",
        "remove_no_prune",
        "remove_branches_no_prune",
        "create_path",
    ]
    depths = sorted(
        {
            row["case"]
            for row in rows
            if row["fixture"] == "nested" and row["repr"] == "line-list"
        }
    )
    for op in ops:
        for depth in depths:
            line = indexed.get((op, "line-list", "nested", depth))
            dense = indexed.get((op, "dense", "nested", depth))
            if line is None or dense is None:
                continue
            label = f"{op} nested depth={depth}"
            comparisons.append(compare_rows(label, line, dense))
    return comparisons


def root_threshold_ab(rows):
    indexed = row_index(rows)
    comparisons = []
    ops = ["get", "path_exists", "update", "join", "get_or_set", "remove_no_prune"]
    for op in ops:
        line = indexed.get((op, "line-list", "root", 2))
        if line is None:
            continue
        for dense_fanout in [3, 4]:
            dense = indexed.get((op, "dense", "root", dense_fanout))
            if dense is None:
                continue
            label = f"{op} root line-list(2) vs dense({dense_fanout})"
            comparisons.append(compare_rows(label, line, dense))
    return comparisons


def node_kernel_ab(rows):
    indexed = row_index(rows)
    comparisons = []
    fanouts = sorted(
        {
            row["case"]
            for row in rows
            if row["fixture"] == "node-kernel" and row["repr"] == "linear-scan"
        }
    )
    for fanout in fanouts:
        linear = indexed.get(("child_hit", "linear-scan", "node-kernel", fanout))
        if linear is None:
            continue
        for repr_name in ["mask-rank", "binary-search", "branchless-linear"]:
            candidate = indexed.get(("child_hit", repr_name, "node-kernel", fanout))
            if candidate is None:
                continue
            label = f"child hit fanout={fanout} linear vs {repr_name}"
            comparisons.append(compare_rows(label, linear, candidate))
    return comparisons


def indexed_bit_direction_ab(rows):
    indexed = row_index(rows)
    comparisons = []
    fanouts = sorted(
        {
            row["case"]
            for row in rows
            if row["fixture"] == "node-kernel"
            and row["repr"] == "mask-select-forward"
        }
    )
    for fanout in fanouts:
        forward = indexed.get(
            ("child_select", "mask-select-forward", "node-kernel", fanout)
        )
        backward = indexed.get(
            ("child_select", "mask-select-backward", "node-kernel", fanout)
        )
        if forward is None or backward is None:
            continue
        label = f"indexed_bit fanout={fanout} forward vs backward"
        comparisons.append(compare_rows(label, forward, backward))
    return comparisons


def indexed_bit_legacy_ab(rows, direction: str):
    indexed = row_index(rows)
    comparisons = []
    fanouts = sorted(
        {
            row["case"]
            for row in rows
            if row["fixture"] == "node-kernel"
            and row["repr"] == f"legacy-mask-select-{direction}"
        }
    )
    for fanout in fanouts:
        legacy = indexed.get(
            (
                "child_select",
                f"legacy-mask-select-{direction}",
                "node-kernel",
                fanout,
            )
        )
        current = indexed.get(
            ("child_select", f"mask-select-{direction}", "node-kernel", fanout)
        )
        if legacy is None or current is None:
            continue
        label = f"indexed_bit {direction} fanout={fanout} legacy vs current"
        comparisons.append(compare_rows(label, legacy, current))
    return comparisons


def render_section(title: str, base_name: str, other_name: str, comparisons) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"| comparison | {base_name} | {other_name} | delta | ratio | samples |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cmp in comparisons:
        base = cmp["base"]
        other = cmp["other"]
        samples = min(base["samples"], other["samples"])
        lines.append(
            "| {label} | {base} | {other} | {delta} | {ratio} | {samples} |".format(
                label=cmp["label"],
                base=fmt_ns(base["median_ns"]),
                other=fmt_ns(other["median_ns"]),
                delta=fmt_ns(cmp["delta_ns"]),
                ratio=fmt_ratio(cmp["ratio"]),
                samples=samples,
            )
        )
    lines.append("")
    return lines


def render_ab(rows) -> str:
    lines = [
        "# Node Layout A/B Summary",
        "",
        "Ratios use median time. A ratio above 1 means the right-hand candidate is faster.",
        "",
    ]
    lines.extend(
        render_section(
            "Current Direct Operation Versus Old Zipper-Shaped Operation",
            "old median",
            "current median",
            old_vs_current(rows),
        )
    )
    lines.extend(
        render_section(
            "Nested Line-List Versus Dense Walks",
            "line-list median",
            "dense median",
            nested_repr_ab(rows),
        )
    )
    lines.extend(
        render_section(
            "Root Threshold-Adjacent Walks",
            "line-list fanout 2",
            "dense fanout 3/4",
            root_threshold_ab(rows),
        )
    )
    lines.extend(
        render_section(
            "Node-Local Child Hit Kernel",
            "linear scan median",
            "candidate median",
            node_kernel_ab(rows),
        )
    )
    lines.extend(
        render_section(
            "ByteMask Indexed-Bit Direction Kernel",
            "forward median",
            "backward median",
            indexed_bit_direction_ab(rows),
        )
    )
    lines.extend(
        render_section(
            "ByteMask Indexed-Bit Forward Legacy Versus Current",
            "legacy median",
            "current median",
            indexed_bit_legacy_ab(rows, "forward"),
        )
    )
    lines.extend(
        render_section(
            "ByteMask Indexed-Bit Backward Legacy Versus Current",
            "legacy median",
            "current median",
            indexed_bit_legacy_ab(rows, "backward"),
        )
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-file", type=pathlib.Path)
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()

    if args.from_file:
        text = args.from_file.read_text()
    else:
        text = run_benchmark(args.sample_count)

    parsed = parse_divan_output(text)
    rows = TABLE.rows(parsed)
    if not rows:
        print("no node_layout benchmark rows found", file=sys.stderr)
        return 1
    print(render_ab(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
