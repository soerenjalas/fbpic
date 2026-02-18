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
- You can pass `--disable-cupy-mempool-free` to test impact of skipping
  `cupy.get_default_memory_pool().free_all_blocks()` after particle exchange.
- You can override selected CUDA launch parameters for tuning:
  - `--deposit-tpb` (particle deposition)
  - `--gather-tpb` (particle gathering)
  - `--copy-tpbx/--copy-tpby` (spectral copy kernels)
  - `--push-tpb` (particle push kernels)
  - `--sort-tpb` (sorting/rearrangement kernels)
  - `--disable-fused-ifft-scale` (A/B test legacy iFFT normalization path)
- Current built-in GPU defaults are architecture-aware (including A100),
  and overrides are useful for cluster-specific retuning.
- For detailed profiling, see `docs/source/advanced/profiling.rst`.
