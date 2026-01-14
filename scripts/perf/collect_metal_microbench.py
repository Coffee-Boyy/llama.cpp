#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import sys


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)


def git_info(repo_root):
    git_hash = run(["git", "rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
    git_short = run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root).stdout.strip()
    dirty = bool(run(["git", "status", "--porcelain"], cwd=repo_root).stdout.strip())
    return git_hash, git_short, dirty


def collect_env():
    env = {}
    for key, value in os.environ.items():
        if key.startswith(("GGML_", "LLAMA_", "METAL_")):
            env[key] = value
    return env


def parse_benchmarks(raw):
    entries = {}
    for bench in raw.get("benchmarks", []):
        run_type = bench.get("run_type")
        if run_type not in (None, "iteration"):
            continue
        name = bench.get("name")
        if not name:
            continue
        entries[name] = {
            "real_time": bench.get("real_time"),
            "cpu_time": bench.get("cpu_time"),
            "time_unit": bench.get("time_unit"),
            "iterations": bench.get("iterations"),
        }
    return entries


def percentile(sorted_values, pct):
    if not sorted_values:
        return None
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    idx = (len(sorted_values) - 1) * (pct / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def summarize_samples(samples):
    summary = {}
    for name, values in samples.items():
        if not values:
            continue
        sorted_vals = sorted(values)
        median = statistics.median(sorted_vals)
        mad = statistics.median([abs(v - median) for v in sorted_vals])
        summary[name] = {
            "median": median,
            "p90": percentile(sorted_vals, 90),
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "mad": mad,
            "count": len(sorted_vals),
        }
    return summary


def read_last_entry(path):
    if not os.path.exists(path):
        return None
    last_line = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                last_line = line
    if not last_line:
        return None
    return json.loads(last_line)


def compare_entries(prev_entry, new_entry):
    prev = prev_entry.get("benchmarks", {}) if prev_entry else {}
    new = new_entry.get("benchmarks", {})
    if not prev or not new:
        return None

    rows = []
    for name, current in new.items():
        previous = prev.get(name)
        if not previous:
            rows.append((name, None, current.get("median"), None, None))
            continue
        old_time = previous.get("median")
        new_time = current.get("median")
        if old_time in (None, 0) or new_time is None:
            rows.append((name, old_time, new_time, None, None))
            continue
        delta = (new_time - old_time) / old_time * 100.0
        rows.append((name, old_time, new_time, delta, current.get("mad_pct")))
    return rows


def format_change(delta):
    if delta is None:
        return "n/a"
    return f"{delta:+.2f}%"


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_bin = os.path.join(repo_root, "build", "bin", "ggml-metal-microbench")
    default_log = os.path.join(repo_root, "docs", "perf", "metal-microbench.jsonl")
    default_runs = os.path.join(repo_root, "docs", "perf", "runs")

    parser = argparse.ArgumentParser(
        description="Collect ggml metal microbench results and append to a JSONL log."
    )
    parser.add_argument("--bin", default=default_bin, help="Path to ggml-metal-microbench binary.")
    parser.add_argument("--log", default=default_log, help="JSONL log file to append.")
    parser.add_argument("--runs-dir", default=default_runs, help="Directory for raw JSON outputs.")
    parser.add_argument("--label", default="", help="Short label for the run (change description).")
    parser.add_argument("--notes", default="", help="Freeform notes (e.g. memory observations).")
    parser.add_argument("--no-compare", action="store_true", help="Skip comparison vs last entry.")
    parser.add_argument("--repetitions", type=int, default=1, help="Number of repeated runs to aggregate.")

    args = parser.parse_args()

    if not os.path.exists(args.bin):
        print(f"Binary not found: {args.bin}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.runs_dir, exist_ok=True)
    git_hash, git_short, dirty = git_info(repo_root)

    timestamp = dt.datetime.now(dt.timezone.utc).astimezone()
    stamp = timestamp.strftime("%Y%m%d-%H%M%S")
    raw_names = []
    sample_times = {}
    sample_iterations = {}
    time_unit = None

    repetitions = max(1, args.repetitions)
    for idx in range(repetitions):
        run_stamp = f"{stamp}-r{idx+1}"
        raw_name = f"metal-microbench-{run_stamp}-{git_short}.json"
        raw_path = os.path.join(args.runs_dir, raw_name)
        raw_names.append(raw_name)

        run([args.bin, f"--benchmark_out={raw_path}", "--benchmark_out_format=json"])
        with open(raw_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        parsed = parse_benchmarks(raw)
        for name, metrics in parsed.items():
            sample_times.setdefault(name, []).append(metrics.get("real_time"))
            sample_iterations.setdefault(name, []).append(metrics.get("iterations"))
            time_unit = metrics.get("time_unit") or time_unit

    summary = summarize_samples(sample_times)
    for name, stats in summary.items():
        median = stats.get("median")
        mad = stats.get("mad")
        stats["mad_pct"] = None if not median else (mad / median * 100.0)

    entry = {
        "timestamp": timestamp.isoformat(),
        "git_hash": git_hash,
        "git_dirty": dirty,
        "label": args.label,
        "notes": args.notes,
        "repetitions": repetitions,
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "mac_ver": platform.mac_ver()[0],
        },
        "env": collect_env(),
        "raw_json": [os.path.relpath(os.path.join(args.runs_dir, name), repo_root) for name in raw_names],
        "benchmarks": summary,
        "samples": {
            "real_time": sample_times,
            "iterations": sample_iterations,
            "time_unit": time_unit,
        },
    }

    last_entry = None
    if not args.no_compare:
        last_entry = read_last_entry(args.log)

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, separators=(",", ":")) + "\n")

    if last_entry and not args.no_compare:
        changes = compare_entries(last_entry, entry)
        if changes:
            unit = entry.get("samples", {}).get("time_unit", "ns")
            print("Comparison vs previous run (median real_time):")
            for name, old_time, new_time, delta, mad_pct in changes:
                old_display = f"{old_time:.2f}" if old_time is not None else "n/a"
                new_display = f"{new_time:.2f}" if new_time is not None else "n/a"
                mad_display = f" (MAD {mad_pct:.2f}%)" if mad_pct is not None else ""
                print(f"- {name}: {old_display}->{new_display} {unit} ({format_change(delta)}){mad_display}")


if __name__ == "__main__":
    main()
