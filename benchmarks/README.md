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
  - `--use-direct-axis0-fft` / `--disable-direct-axis0-fft`
    (A/B test direct `cupy.fft` path that bypasses explicit FFT copy kernels)
  - `--use-cuda-fastmath` / `--disable-cuda-fastmath`
    (A/B test CUDA kernel compilation with fastmath)
  - `--cuda-max-registers`
    (A/B test register-pressure/occupancy tradeoff in CUDA kernels)
  - `--use-unsorted-rho-deposition` / `--disable-unsorted-rho-deposition`
    (force on/off for GPU unsorted rho deposition; default is on)
  - `--use-unsorted-j-deposition` / `--disable-unsorted-j-deposition`
    (force on/off for GPU unsorted J deposition; default is on)
  - `--deposition-backend {numba,cupy_raw}`
    (A/B test experimental CuPy RawKernel backend for supported deposition kernels)
- Current built-in GPU defaults are architecture-aware (including A100),
  and overrides are useful for cluster-specific retuning.
- For `Nm=2` and `Nm=3`, the unsorted deposition path includes fused
  multi-mode kernels (for both rho and J) to reduce per-mode launch overhead.
- For detailed profiling, see `docs/source/advanced/profiling.rst`.

## `boosted_frame_benchmark.py`

A boosted-frame benchmark for production-like timing runs that:

1. Builds a boosted-frame LWFA setup (plasma + optional laser/bunch)
2. Runs warmup steps first (kernel pre-cooking/JIT)
3. Times only steady-state steps
4. Attaches no diagnostics (`sim.diags = []`) to avoid output jitter

### Run (GPU, same defaults as original boosted example)

```bash
python benchmarks/boosted_frame_benchmark.py
```

### Run (GPU, steady-state with pre-cooking)

```bash
python benchmarks/boosted_frame_benchmark.py \
  --warmup-steps 5
```

### Run (GPU, RawKernel deposition experiment)

```bash
python benchmarks/boosted_frame_benchmark.py \
  --Nm 3 --particle-shape cubic \
  --warmup-steps 5 \
  --deposition-backend cupy_raw
```

### Notes

- Running with no args now matches the defaults from
  `docs/source/example_input/boosted_frame_script.py`
  (including grid and default `N_step` formula).
- `--steps` overrides the timed step count; by default it uses the original
  boosted-frame `N_step` expression.
- Warmup is excluded from timed runtime.
- You can still control kernel behavior with environment variables (e.g.
  `FBPIC_USE_UNSORTED_J_DEPOSITION`, `FBPIC_DEPOSITION_BACKEND`) before launching the script.
