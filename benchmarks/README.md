# FBPIC benchmarks

## `simple_step_benchmark.py`

A lightweight benchmark that:

1. Creates a configurable FBPIC simulation (periodic box + plasma species)
2. Runs warm-up iterations (for JIT and setup)
3. Times a fixed number of PIC steps
4. Prints coarse per-phase timings (gather, push, deposit, field push, etc.)

### Run (CPU)

```bash
python benchmarks/simple_step_benchmark.py \
  --steps 20 --warmup-steps 2 \
  --Nz 256 --Nr 96 --Nm 2 \
  --p-nz 2 --p-nr 2 --p-nt 4
```

### Run (GPU)

```bash
python benchmarks/simple_step_benchmark.py --use-cuda
```

### Notes

- If running from source (without `pip install -e .`), the script auto-adds repo root to `sys.path`.
- The per-phase timings are coarse wrappers around high-level methods and are meant for trend tracking, not full profiling.
- For cleaner profiler traces (without explicit synchronizations from the wrappers), use `--no-phase-breakdown`.
- You can pass `--exchange-period` to test sensitivity to particle exchange frequency.
- For detailed profiling, see `docs/source/advanced/profiling.rst`.
