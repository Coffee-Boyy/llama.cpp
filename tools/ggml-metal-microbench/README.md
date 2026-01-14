# ggml-metal-microbench

Microbenchmarks for Metal inference hot paths using Google Benchmark.

## Build

```
cmake -S . -B build \
  -DGGML_METAL=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_METAL_MICROBENCH=ON
cmake --build build --config Release
```

## Run

```
./build/bin/ggml-metal-microbench
```

## Tracking

For tracking results over time (compute + memory notes), use:

```
python3 scripts/perf/collect_metal_microbench.py --label "change description"
```

See `docs/perf/metal-microbench.md` for the logging format and comparison behavior.

## Notes

- The benchmarks construct single-op graphs for core inference hot paths (matmul, RMS norm, RoPE, softmax, flash attention).
- If a kernel is not supported by the current Metal device, the benchmark will be skipped.
