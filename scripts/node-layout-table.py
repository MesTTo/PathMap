#!/usr/bin/env python3
"""Run or parse the node-layout Divan benchmark as a compact Markdown table."""

import argparse
import pathlib
import subprocess
import sys


REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benches"))

from divan_fmt import parse_divan_output  # noqa: E402


CASE_META = {
    "node_layout_kernel_linear_child_hit": {
        "op": "child_hit",
        "repr": "linear-scan",
        "fixture": "node-kernel",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_kernel_mask_rank_hit": {
        "op": "child_hit",
        "repr": "mask-rank",
        "fixture": "node-kernel",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_kernel_binary_child_hit": {
        "op": "child_hit",
        "repr": "binary-search",
        "fixture": "node-kernel",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_kernel_branchless_linear_child_hit": {
        "op": "child_hit",
        "repr": "branchless-linear",
        "fixture": "node-kernel",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_kernel_indexed_bit_forward": {
        "op": "child_select",
        "repr": "mask-select-forward",
        "fixture": "node-kernel",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_kernel_indexed_bit_backward": {
        "op": "child_select",
        "repr": "mask-select-backward",
        "fixture": "node-kernel",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_kernel_legacy_indexed_bit_forward": {
        "op": "child_select",
        "repr": "legacy-mask-select-forward",
        "fixture": "node-kernel",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_kernel_legacy_indexed_bit_backward": {
        "op": "child_select",
        "repr": "legacy-mask-select-backward",
        "fixture": "node-kernel",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_line_list_root_lookup": {
        "op": "get",
        "repr": "line-list",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_line_list_root_update": {
        "op": "update",
        "repr": "line-list",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_line_list_root_join": {
        "op": "join",
        "repr": "line-list",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_line_list_root_old_join": {
        "op": "old_join",
        "repr": "line-list",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_line_list_root_get_or_set": {
        "op": "get_or_set",
        "repr": "line-list",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_line_list_root_old_get_or_set": {
        "op": "old_get_or_set",
        "repr": "line-list",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_line_list_root_remove_no_prune": {
        "op": "remove_no_prune",
        "repr": "line-list",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_line_list_root_old_remove_no_prune": {
        "op": "old_remove_no_prune",
        "repr": "line-list",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_dense_root_lookup": {
        "op": "get",
        "repr": "dense",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_dense_root_update": {
        "op": "update",
        "repr": "dense",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_dense_root_join": {
        "op": "join",
        "repr": "dense",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_dense_root_old_join": {
        "op": "old_join",
        "repr": "dense",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_dense_root_get_or_set": {
        "op": "get_or_set",
        "repr": "dense",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_dense_root_old_get_or_set": {
        "op": "old_get_or_set",
        "repr": "dense",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_dense_root_remove_no_prune": {
        "op": "remove_no_prune",
        "repr": "dense",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_dense_root_old_remove_no_prune": {
        "op": "old_remove_no_prune",
        "repr": "dense",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_compressed_key_lookup": {
        "op": "get",
        "repr": "compressed",
        "fixture": "single-key",
        "child_items": lambda _case: 1,
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_compressed_key_update": {
        "op": "update",
        "repr": "compressed",
        "fixture": "single-key",
        "child_items": lambda _case: 1,
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_compressed_key_join": {
        "op": "join",
        "repr": "compressed",
        "fixture": "single-key",
        "child_items": lambda _case: 1,
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_compressed_key_old_join": {
        "op": "old_join",
        "repr": "compressed",
        "fixture": "single-key",
        "child_items": lambda _case: 1,
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_compressed_key_get_or_set": {
        "op": "get_or_set",
        "repr": "compressed",
        "fixture": "single-key",
        "child_items": lambda _case: 1,
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_compressed_key_old_get_or_set": {
        "op": "old_get_or_set",
        "repr": "compressed",
        "fixture": "single-key",
        "child_items": lambda _case: 1,
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_compressed_key_remove_no_prune": {
        "op": "remove_no_prune",
        "repr": "compressed",
        "fixture": "single-key",
        "child_items": lambda _case: 1,
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_compressed_key_old_remove_no_prune": {
        "op": "old_remove_no_prune",
        "repr": "compressed",
        "fixture": "single-key",
        "child_items": lambda _case: 1,
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_compressed_key_remove_branches_no_prune": {
        "op": "remove_branches_no_prune",
        "repr": "compressed",
        "fixture": "branch-prefix",
        "child_items": lambda _case: 1,
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_compressed_key_old_remove_branches_no_prune": {
        "op": "old_remove_branches_no_prune",
        "repr": "compressed",
        "fixture": "branch-prefix",
        "child_items": lambda _case: 1,
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_compressed_key_create_path": {
        "op": "create_path",
        "repr": "compressed",
        "fixture": "missing-path",
        "child_items": lambda _case: 1,
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_compressed_key_old_create_path": {
        "op": "old_create_path",
        "repr": "compressed",
        "fixture": "missing-path",
        "child_items": lambda _case: 1,
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_line_list_lookup": {
        "op": "get",
        "repr": "line-list",
        "fixture": "nested",
        "child_items": lambda _case: 2,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_line_list_update": {
        "op": "update",
        "repr": "line-list",
        "fixture": "nested",
        "child_items": lambda _case: 2,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_line_list_join": {
        "op": "join",
        "repr": "line-list",
        "fixture": "nested",
        "child_items": lambda _case: 2,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_line_list_old_join": {
        "op": "old_join",
        "repr": "line-list",
        "fixture": "nested",
        "child_items": lambda _case: 2,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_line_list_get_or_set": {
        "op": "get_or_set",
        "repr": "line-list",
        "fixture": "nested",
        "child_items": lambda _case: 2,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_line_list_old_get_or_set": {
        "op": "old_get_or_set",
        "repr": "line-list",
        "fixture": "nested",
        "child_items": lambda _case: 2,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_line_list_remove_no_prune": {
        "op": "remove_no_prune",
        "repr": "line-list",
        "fixture": "nested",
        "child_items": lambda _case: 2,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_line_list_old_remove_no_prune": {
        "op": "old_remove_no_prune",
        "repr": "line-list",
        "fixture": "nested",
        "child_items": lambda _case: 2,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_line_list_remove_branches_no_prune": {
        "op": "remove_branches_no_prune",
        "repr": "line-list",
        "fixture": "nested",
        "child_items": lambda _case: 2,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_line_list_old_remove_branches_no_prune": {
        "op": "old_remove_branches_no_prune",
        "repr": "line-list",
        "fixture": "nested",
        "child_items": lambda _case: 2,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_line_list_create_path": {
        "op": "create_path",
        "repr": "line-list",
        "fixture": "nested",
        "child_items": lambda _case: 2,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case) + 1,
    },
    "node_layout_nested_line_list_old_create_path": {
        "op": "old_create_path",
        "repr": "line-list",
        "fixture": "nested",
        "child_items": lambda _case: 2,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case) + 1,
    },
    "node_layout_nested_dense_lookup": {
        "op": "get",
        "repr": "dense",
        "fixture": "nested",
        "child_items": lambda _case: 3,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_dense_old_join": {
        "op": "old_join",
        "repr": "dense",
        "fixture": "nested",
        "child_items": lambda _case: 3,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_dense_join": {
        "op": "join",
        "repr": "dense",
        "fixture": "nested",
        "child_items": lambda _case: 3,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_dense_update": {
        "op": "update",
        "repr": "dense",
        "fixture": "nested",
        "child_items": lambda _case: 3,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_dense_get_or_set": {
        "op": "get_or_set",
        "repr": "dense",
        "fixture": "nested",
        "child_items": lambda _case: 3,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_dense_old_get_or_set": {
        "op": "old_get_or_set",
        "repr": "dense",
        "fixture": "nested",
        "child_items": lambda _case: 3,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_dense_remove_no_prune": {
        "op": "remove_no_prune",
        "repr": "dense",
        "fixture": "nested",
        "child_items": lambda _case: 3,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_dense_old_remove_no_prune": {
        "op": "old_remove_no_prune",
        "repr": "dense",
        "fixture": "nested",
        "child_items": lambda _case: 3,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_dense_remove_branches_no_prune": {
        "op": "remove_branches_no_prune",
        "repr": "dense",
        "fixture": "nested",
        "child_items": lambda _case: 3,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_dense_old_remove_branches_no_prune": {
        "op": "old_remove_branches_no_prune",
        "repr": "dense",
        "fixture": "nested",
        "child_items": lambda _case: 3,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_dense_create_path": {
        "op": "create_path",
        "repr": "dense",
        "fixture": "nested",
        "child_items": lambda _case: 3,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case) + 1,
    },
    "node_layout_nested_dense_old_create_path": {
        "op": "old_create_path",
        "repr": "dense",
        "fixture": "nested",
        "child_items": lambda _case: 3,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case) + 1,
    },
    "node_layout_line_list_root_path_exists": {
        "op": "path_exists",
        "repr": "line-list",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_dense_root_path_exists": {
        "op": "path_exists",
        "repr": "dense",
        "fixture": "root",
        "child_items": lambda case: int(case),
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda _case: 1,
    },
    "node_layout_compressed_key_path_exists": {
        "op": "path_exists",
        "repr": "compressed",
        "fixture": "single-key",
        "child_items": lambda _case: 1,
        "path_depth": lambda _case: 1,
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_line_list_path_exists": {
        "op": "path_exists",
        "repr": "line-list",
        "fixture": "nested",
        "child_items": lambda _case: 2,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
    "node_layout_nested_dense_path_exists": {
        "op": "path_exists",
        "repr": "dense",
        "fixture": "nested",
        "child_items": lambda _case: 3,
        "path_depth": lambda case: int(case),
        "bytes_per_key": lambda case: int(case),
    },
}


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


def case_value(case: str) -> int:
    return int(case.strip().strip("()"))


def fmt_ns(ns: float) -> str:
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.3f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.3f} us"
    return f"{ns:.1f} ns"


def rows(parsed):
    result = []
    for (group, case), stats in parsed.items():
        meta = CASE_META.get(group)
        if meta is None:
            continue
        case_int = case_value(case)
        result.append(
            {
                "group": group,
                "case": case_int,
                "op": meta["op"],
                "repr": meta["repr"],
                "fixture": meta["fixture"],
                "child_items": meta["child_items"](case_int),
                "path_depth": meta["path_depth"](case_int),
                "bytes_per_key": meta["bytes_per_key"](case_int),
                **stats,
            }
        )
    return sorted(result, key=lambda row: (row["group"], row["case"]))


def render_table(rows):
    lines = [
        "| benchmark | op | repr | fixture | children | depth | bytes | median | mean | samples | iters |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {group}({case}) | {op} | {repr} | {fixture} | {child_items} | {path_depth} | {bytes_per_key} | {median} | {mean} | {samples} | {iters} |".format(
                group=row["group"],
                case=row["case"],
                op=row["op"],
                repr=row["repr"],
                fixture=row["fixture"],
                child_items=row["child_items"],
                path_depth=row["path_depth"],
                bytes_per_key=row["bytes_per_key"],
                median=fmt_ns(row["median_ns"]),
                mean=fmt_ns(row["mean_ns"]),
                samples=row["samples"],
                iters=row["iters"],
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
    table_rows = rows(parsed)
    if not table_rows:
        print("no node_layout benchmark rows found", file=sys.stderr)
        return 1
    print(render_table(table_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
