#!/usr/bin/env python3

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchDag:
    root: dict
    nodes: dict[str, dict]
    edges: list[dict]


def _matching_roots(records, depth, move):
    return [
        record
        for record in records
        if record.get("type") == "root"
        and (depth is None or record["depth"] == depth)
        and (move is None or record["move"] == move)
    ]


def load_search_dag(path, depth=None, move=None):
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    roots = _matching_roots(records, depth, move)
    if not roots:
        raise ValueError(f"{path}: no matching root trace")
    root = roots[-1]
    sequence = root["sequence"]
    nodes = {
        record["hash"]: record
        for record in records
        if record.get("type") == "node" and record["sequence"] == sequence
    }
    edges = [
        record
        for record in records
        if record.get("type") == "edge" and record["sequence"] == sequence
    ]
    if len(nodes) != root["positions"]:
        raise ValueError(
            f"{path}: root declares {root['positions']} positions, found {len(nodes)}"
        )
    if len(edges) != root["edges"]:
        raise ValueError(f"{path}: root declares {root['edges']} edges, found {len(edges)}")
    return SearchDag(root=root, nodes=nodes, edges=edges)


def _nodes_by_fen(dag):
    nodes = {}
    for node in dag.nodes.values():
        if node["fen"] in nodes:
            raise ValueError(f"trace contains duplicate FEN identity: {node['fen']}")
        nodes[node["fen"]] = node
    return nodes


def _frontier(dag, exclusive, common):
    hash_to_fen = {node_hash: node["fen"] for node_hash, node in dag.nodes.items()}
    frontier = {}
    for edge in dag.edges:
        child_fen = hash_to_fen.get(edge["child"])
        parent_fen = hash_to_fen.get(edge["parent"])
        if child_fen in exclusive and parent_fen in common:
            frontier.setdefault(edge["child"], []).append(
                {
                    "fen": child_fen,
                    "parent": edge["parent"],
                    "parent_fen": parent_fen,
                    "visits": edge["visits"],
                }
            )
    return frontier


def _evaluation_delta(baseline, candidate):
    if baseline["eval_visits"] == 0 or candidate["eval_visits"] == 0:
        return None
    if (
        baseline["min_eval"] == candidate["min_eval"]
        and baseline["max_eval"] == candidate["max_eval"]
    ):
        return None
    return {
        "hash": baseline["hash"],
        "fen": baseline["fen"],
        "baseline": {
            "visits": baseline["eval_visits"],
            "min": baseline["min_eval"],
            "max": baseline["max_eval"],
        },
        "candidate": {
            "visits": candidate["eval_visits"],
            "min": candidate["min_eval"],
            "max": candidate["max_eval"],
        },
    }


def compare_search_dags(baseline, candidate):
    baseline_nodes = _nodes_by_fen(baseline)
    candidate_nodes = _nodes_by_fen(candidate)
    baseline_positions = set(baseline_nodes)
    candidate_positions = set(candidate_nodes)
    common = baseline_positions & candidate_positions
    baseline_only = baseline_positions - candidate_positions
    candidate_only = candidate_positions - baseline_positions
    evaluation_deltas = [
        delta
        for fen in sorted(common)
        if (
            delta := _evaluation_delta(
                baseline_nodes[fen], candidate_nodes[fen]
            )
        )
        is not None
    ]
    return {
        "baseline_root": baseline.root,
        "candidate_root": candidate.root,
        "summary": {
            "common_positions": len(common),
            "baseline_only_positions": len(baseline_only),
            "candidate_only_positions": len(candidate_only),
            "common_positions_with_different_evaluation": len(evaluation_deltas),
            "baseline_truncated": baseline.root["truncated"],
            "candidate_truncated": candidate.root["truncated"],
        },
        "baseline_only_frontier": _frontier(baseline, baseline_only, common),
        "candidate_only_frontier": _frontier(candidate, candidate_only, common),
        "evaluation_deltas": evaluation_deltas,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the last matching fixed-depth search DAG from two "
            "EMBER_TRACE_SEARCH_DAG JSONL files."
        )
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--move")
    parser.add_argument("--output")
    args = parser.parse_args()

    baseline = load_search_dag(args.baseline, args.depth, args.move)
    candidate = load_search_dag(args.candidate, args.depth, args.move)
    result = compare_search_dags(baseline, candidate)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
