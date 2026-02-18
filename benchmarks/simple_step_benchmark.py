#!/usr/bin/env python3
"""Simple performance benchmark for FBPIC PIC steps.

This script creates a small configurable simulation, warms up JIT kernels,
then times a sequence of PIC steps and reports a coarse per-phase breakdown.
"""

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path
import sys

# Allow running this script directly from the source tree without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scipy.constants import c

from fbpic.main import Simulation


def apply_kernel_env_overrides(args):
    """Apply optional CUDA kernel launch overrides via environment vars."""
    if args.use_unsorted_rho_deposition and args.disable_unsorted_rho_deposition:
        raise ValueError("Cannot set both --use-unsorted-rho-deposition and --disable-unsorted-rho-deposition")
    if args.use_unsorted_j_deposition and args.disable_unsorted_j_deposition:
        raise ValueError("Cannot set both --use-unsorted-j-deposition and --disable-unsorted-j-deposition")
    if args.use_direct_axis0_fft and args.disable_direct_axis0_fft:
        raise ValueError("Cannot set both --use-direct-axis0-fft and --disable-direct-axis0-fft")
    if args.use_cuda_fastmath and args.disable_cuda_fastmath:
        raise ValueError("Cannot set both --use-cuda-fastmath and --disable-cuda-fastmath")
    if args.use_supercell_j_deposition and args.disable_supercell_j_deposition:
        raise ValueError("Cannot set both --use-supercell-j-deposition and --disable-supercell-j-deposition")
    if args.use_supercell_j_deposition and args.use_unsorted_j_deposition:
        raise ValueError("--use-supercell-j-deposition and --use-unsorted-j-deposition are mutually exclusive")

    if args.deposit_tpb is not None:
        os.environ["FBPIC_DEPOSIT_TPB"] = str(args.deposit_tpb)
    if args.gather_tpb is not None:
        os.environ["FBPIC_GATHER_TPB"] = str(args.gather_tpb)
    if args.copy_tpbx is not None:
        os.environ["FBPIC_COPY_TPBX"] = str(args.copy_tpbx)
    if args.copy_tpby is not None:
        os.environ["FBPIC_COPY_TPBY"] = str(args.copy_tpby)
    if args.push_tpb is not None:
        os.environ["FBPIC_PUSH_TPB"] = str(args.push_tpb)
    if args.sort_tpb is not None:
        os.environ["FBPIC_SORT_TPB"] = str(args.sort_tpb)
    if args.disable_fused_ifft_scale:
        os.environ["FBPIC_FFT_FUSE_IFFT_SCALE"] = "0"
    if args.use_direct_axis0_fft:
        os.environ["FBPIC_USE_DIRECT_AXIS0_FFT"] = "1"
    if args.disable_direct_axis0_fft:
        os.environ["FBPIC_USE_DIRECT_AXIS0_FFT"] = "0"
    if args.use_cuda_fastmath:
        os.environ["FBPIC_CUDA_FASTMATH"] = "1"
    if args.disable_cuda_fastmath:
        os.environ["FBPIC_CUDA_FASTMATH"] = "0"
    if args.cuda_max_registers is not None:
        os.environ["FBPIC_CUDA_MAX_REGISTERS"] = str(args.cuda_max_registers)
    if args.use_unsorted_rho_deposition:
        os.environ["FBPIC_USE_UNSORTED_RHO_DEPOSITION"] = "1"
    if args.disable_unsorted_rho_deposition:
        os.environ["FBPIC_USE_UNSORTED_RHO_DEPOSITION"] = "0"
    if args.use_unsorted_j_deposition:
        os.environ["FBPIC_USE_UNSORTED_J_DEPOSITION"] = "1"
    if args.disable_unsorted_j_deposition:
        os.environ["FBPIC_USE_UNSORTED_J_DEPOSITION"] = "0"
    if args.use_supercell_j_deposition:
        os.environ["FBPIC_USE_SUPERCELL_J_DEPOSITION"] = "1"
    if args.disable_supercell_j_deposition:
        os.environ["FBPIC_USE_SUPERCELL_J_DEPOSITION"] = "0"


def build_simulation(args):
    zmin = 0.0
    zmax = args.zmax
    rmax = args.rmax
    dt = args.dt if args.dt is not None else zmax / args.Nz / c

    sim = Simulation(
        Nz=args.Nz,
        zmax=zmax,
        Nr=args.Nr,
        rmax=rmax,
        Nm=args.Nm,
        dt=dt,
        p_zmin=zmin,
        p_zmax=zmax,
        p_rmin=0.0,
        p_rmax=args.plasma_rmax if args.plasma_rmax is not None else 0.8 * rmax,
        p_nz=args.p_nz,
        p_nr=args.p_nr,
        p_nt=args.p_nt,
        n_e=args.n_e,
        n_order=args.n_order,
        exchange_period=args.exchange_period,
        clear_cupy_mempool_on_exchange=(not args.disable_cupy_mempool_free),
        particle_shape=args.particle_shape,
        use_cuda=args.use_cuda,
        boundaries={"z": "periodic", "r": "reflective"},
        verbose_level=0,
    )
    return sim


def make_gpu_sync(sim):
    if not sim.use_cuda:
        return lambda: None
    try:
        import cupy

        return cupy.cuda.Stream.null.synchronize
    except Exception:
        # Fallback: no explicit sync if cupy import/sync is unavailable.
        return lambda: None


def wrap_method(obj, method_name, key, timings, calls, sync):
    original = getattr(obj, method_name)

    def wrapped(*args, **kwargs):
        sync()
        t0 = time.perf_counter()
        out = original(*args, **kwargs)
        sync()
        timings[key] += time.perf_counter() - t0
        calls[key] += 1
        return out

    setattr(obj, method_name, wrapped)


def instrument(sim):
    timings = defaultdict(float)
    calls = defaultdict(int)
    sync = make_gpu_sync(sim)

    wrap_method(sim, "deposit", "deposit", timings, calls, sync)
    wrap_method(sim, "exchange_and_damp_EB", "exchange_and_damp_EB", timings, calls, sync)
    wrap_method(sim.fld, "push", "field_push", timings, calls, sync)
    wrap_method(sim.comm, "exchange_particles", "exchange_particles", timings, calls, sync)

    for species in sim.ptcl:
        wrap_method(species, "gather", "gather", timings, calls, sync)
        wrap_method(species, "push_p", "push_p", timings, calls, sync)
        wrap_method(species, "push_x", "push_x", timings, calls, sync)

    return timings, calls


def run_benchmark(args):
    apply_kernel_env_overrides(args)
    sim = build_simulation(args)

    n_particles = sum(species.Ntot for species in sim.ptcl)
    n_cells = sim.fld.Nz * sim.fld.Nr * sim.fld.Nm

    # Warm-up for JIT compilation and data path setup.
    if args.warmup_steps > 0:
        sim.step(args.warmup_steps, show_progress=False)

    if args.no_phase_breakdown:
        timings, calls = {}, {}
    else:
        timings, calls = instrument(sim)

    t0 = time.perf_counter()
    sim.step(args.steps, show_progress=False)
    total = time.perf_counter() - t0

    print("=== FBPIC Simple Step Benchmark ===")
    print(f"backend             : {'GPU' if sim.use_cuda else 'CPU'}")
    if sim.use_cuda:
        # Print resolved kernel launch settings (not only env overrides)
        if len(sim.ptcl) > 0:
            p0 = sim.ptcl[0]
            print(f"deposit_tpb         : {p0.deposit_tpb}")
            print(f"gather_tpb          : {p0.gather_tpb}")
            print(f"push_tpb            : {getattr(p0, 'push_tpb', 'n/a')}")
            print(f"sort_tpb            : {getattr(p0, 'sort_tpb', 'n/a')}")
            print(f"unsorted_rho        : {getattr(p0, 'use_unsorted_rho_deposition', 'n/a')}")
            print(f"unsorted_J          : {getattr(p0, 'use_unsorted_J_deposition', 'n/a')}")
            print(f"supercell_J         : {getattr(p0, 'use_supercell_J_deposition', 'n/a')}")
        else:
            print("deposit_tpb         : n/a")
            print("gather_tpb          : n/a")
            print("push_tpb            : n/a")
            print("sort_tpb            : n/a")
            print("supercell_J         : n/a")

        # The copy kernels are configured in both FFT and DHT transforms
        fft_copy_tpb = getattr(sim.fld.trans[0].fft, 'dim_block', 'n/a')
        dht_copy_tpb = getattr(sim.fld.trans[0].dht0, 'dim_block', 'n/a')
        print(f"fft_copy_tpb        : {fft_copy_tpb}")
        print(f"dht_copy_tpb        : {dht_copy_tpb}")
        print(f"fuse_ifft_scale     : {getattr(sim.fld.trans[0].fft, 'fuse_ifft_scale', 'n/a')}")
        print(f"direct_axis0_fft    : {getattr(sim.fld.trans[0].fft, 'use_direct_axis0_fft', 'n/a')}")
        print(f"cuda_fastmath       : {os.environ.get('FBPIC_CUDA_FASTMATH', '0')}")
        print(f"cuda_max_registers  : {os.environ.get('FBPIC_CUDA_MAX_REGISTERS', 'default')}")
    print(f"steps               : {args.steps}")
    print(f"grid (Nz, Nr, Nm)   : ({sim.fld.Nz}, {sim.fld.Nr}, {sim.fld.Nm})")
    print(f"particles (total)   : {n_particles}")
    print(f"cells (Nz*Nr*Nm)    : {n_cells}")
    print(f"total runtime [s]   : {total:.6f}")
    print(f"time / step [s]     : {total / args.steps:.6f}")
    if n_particles > 0:
        pps = (n_particles * args.steps) / total
        print(f"particle-updates/s  : {pps:.3e}")

    if args.no_phase_breakdown:
        return

    print("\n--- Phase breakdown (coarse) ---")
    ordered_keys = [
        "exchange_particles",
        "gather",
        "push_p",
        "push_x",
        "deposit",
        "field_push",
        "exchange_and_damp_EB",
    ]
    known = 0.0
    for key in ordered_keys:
        t = timings.get(key, 0.0)
        c = calls.get(key, 0)
        known += t
        frac = 100.0 * t / total if total > 0 else 0.0
        print(f"{key:22s} : {t:10.6f} s  ({frac:6.2f} %)  calls={c}")

    other = max(0.0, total - known)
    frac_other = 100.0 * other / total if total > 0 else 0.0
    print(f"{'other/uninstrumented':22s} : {other:10.6f} s  ({frac_other:6.2f} %)")


def parse_args():
    p = argparse.ArgumentParser(description="Simple FBPIC benchmark")
    p.add_argument("--steps", type=int, default=20, help="timed PIC steps")
    p.add_argument("--warmup-steps", type=int, default=2, help="warm-up steps before timing")

    p.add_argument("--Nz", type=int, default=256)
    p.add_argument("--Nr", type=int, default=96)
    p.add_argument("--Nm", type=int, default=2)

    p.add_argument("--zmax", type=float, default=50e-6)
    p.add_argument("--rmax", type=float, default=25e-6)
    p.add_argument("--dt", type=float, default=None, help="default: zmax/Nz/c")

    p.add_argument("--n-e", dest="n_e", type=float, default=1e24)
    p.add_argument("--p-nz", dest="p_nz", type=int, default=2)
    p.add_argument("--p-nr", dest="p_nr", type=int, default=2)
    p.add_argument("--p-nt", dest="p_nt", type=int, default=4)
    p.add_argument("--plasma-rmax", type=float, default=None)

    p.add_argument("--n-order", type=int, default=16)
    p.add_argument("--exchange-period", type=int, default=None,
                   help="particle exchange period passed to Simulation")
    p.add_argument("--particle-shape", choices=["linear", "cubic"], default="linear")
    p.add_argument("--disable-cupy-mempool-free", action="store_true",
                   help="disable CuPy memory-pool free_all_blocks after particle exchange")
    p.add_argument("--deposit-tpb", type=int, default=None,
                   help="override FBPIC_DEPOSIT_TPB")
    p.add_argument("--gather-tpb", type=int, default=None,
                   help="override FBPIC_GATHER_TPB")
    p.add_argument("--copy-tpbx", type=int, default=None,
                   help="override FBPIC_COPY_TPBX")
    p.add_argument("--copy-tpby", type=int, default=None,
                   help="override FBPIC_COPY_TPBY")
    p.add_argument("--push-tpb", type=int, default=None,
                   help="override FBPIC_PUSH_TPB")
    p.add_argument("--sort-tpb", type=int, default=None,
                   help="override FBPIC_SORT_TPB")
    p.add_argument("--disable-fused-ifft-scale", action="store_true",
                   help="disable fused iFFT normalization+copy kernel on GPU")
    p.add_argument("--use-direct-axis0-fft", action="store_true",
                   help="experimental direct cupy.fft path along axis 0")
    p.add_argument("--disable-direct-axis0-fft", action="store_true",
                   help="force-disable direct cupy.fft axis-0 path")
    p.add_argument("--use-cuda-fastmath", action="store_true",
                   help="compile CUDA kernels with fastmath (experimental)")
    p.add_argument("--disable-cuda-fastmath", action="store_true",
                   help="force-disable CUDA fastmath compilation")
    p.add_argument("--cuda-max-registers", type=int, default=None,
                   help="set max_registers for CUDA kernel compilation (experimental)")
    p.add_argument("--use-unsorted-rho-deposition", action="store_true",
                   help="force-enable unsorted atomic rho deposition on GPU")
    p.add_argument("--disable-unsorted-rho-deposition", action="store_true",
                   help="force-disable unsorted atomic rho deposition on GPU")
    p.add_argument("--use-unsorted-j-deposition", action="store_true",
                   help="force-enable unsorted atomic J deposition on GPU (experimental)")
    p.add_argument("--disable-unsorted-j-deposition", action="store_true",
                   help="force-disable unsorted atomic J deposition on GPU")
    p.add_argument("--use-supercell-j-deposition", action="store_true",
                   help="enable experimental sorted supercell-style J deposition for cubic Nm=3")
    p.add_argument("--disable-supercell-j-deposition", action="store_true",
                   help="disable experimental sorted supercell-style J deposition")
    p.add_argument("--use-cuda", action="store_true", help="run benchmark on GPU")
    p.add_argument("--no-phase-breakdown", action="store_true",
                   help="disable wrapped per-phase timing (useful for cleaner nsys traces)")

    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
