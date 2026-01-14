# Metal Microbench Tracking

This workflow captures compute and memory-related kernel performance for Metal hot paths,
and keeps an append-only history you can compare over time.

## Build

```
cmake -S . -B build \
  -DGGML_METAL=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_MICROBENCH=ON
cmake --build build --config Release
```

## Collect + Append

```
python3 scripts/perf/collect_metal_microbench.py \
  --label "tiled softmax v2" \
  --notes "Memory: fewer intermediate buffers, peak ~64MB via Xcode"
```

This appends a JSONL record to `docs/perf/metal-microbench.jsonl` and writes the raw
benchmark output to `docs/perf/runs/metal-microbench-<timestamp>-<git>.json`.

## Compare

By default the script prints a percent delta versus the previous run for each benchmark.
If you want to skip comparison:

```
python3 scripts/perf/collect_metal_microbench.py --no-compare
```

## Noise Handling

When you are looking for <2-3% improvements, take multiple repetitions and compare
median times instead of single runs. The script can aggregate repeated runs and
prints median deltas plus a MAD (median absolute deviation) percentage.

```
python3 scripts/perf/collect_metal_microbench.py --repetitions 7 --label "softmax unroll"
```

## Suggested Practice

- Add a short `--label` tied to the kernel change.
- Use `--notes` to record memory observations or profiling tooling.
- Commit the updated `docs/perf/metal-microbench.jsonl` when you want the change tracked.
