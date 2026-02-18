#!/usr/bin/env python3
"""Boosted-frame benchmark with warmup (pre-cooking) and no diagnostics.

This benchmark is meant for steady-state performance checks:
- warmup steps are run first (JIT + data-path pre-cooking)
- timed steps are run afterwards
- no diagnostics are attached (no output I/O jitter)
"""

import argparse
import inspect
import sys
import time
from pathlib import Path

import numpy as np
from scipy.constants import c, e, m_e, m_p

# Allow running this script directly from the source tree without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def make_density_profile(ramp_up, plateau, ramp_down, n_e, w_matched):
    """Return the density profile function used by the benchmark."""
    rel_delta_n_over_w2 = 1.0 / (np.pi * 2.81e-15 * w_matched**4 * n_e)
    p_zmax = ramp_up + plateau + ramp_down

    def dens_func(z, r):
        n = np.ones_like(z)

        # Ramp up
        inv_ramp_up = 1.0 / ramp_up
        n = np.where(z < ramp_up, z * inv_ramp_up, n)

        # Ramp down
        inv_ramp_down = 1.0 / ramp_down
        n = np.where(
            (z >= ramp_up + plateau) & (z < p_zmax),
            -(z - p_zmax) * inv_ramp_down,
            n,
        )
        n = np.where(z >= p_zmax, 0.0, n)

        # Transverse guiding profile
        n = n * (1.0 + rel_delta_n_over_w2 * r**2)
        return n

    return dens_func


def build_simulation(args):
    """Build boosted-frame simulation without diagnostics."""
    # Import after argparse/env setup, so import-time CUDA options can be
    # controlled externally through environment variables.
    from fbpic.main import Simulation
    from fbpic.lpa_utils.bunch import add_particle_bunch
    from fbpic.lpa_utils.boosted_frame import BoostConverter
    from fbpic.lpa_utils.laser import add_laser_pulse
    from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser

    boost = BoostConverter(args.gamma_boost)

    z_span = args.zmax - args.zmin
    if args.dt is None:
        dt = min(args.rmax / (2 * boost.gamma0 * args.Nr) / c, z_span / args.Nz / c)
    else:
        dt = args.dt

    v_comoving = None
    if not args.disable_galilean:
        v_comoving = -c * np.sqrt(1.0 - 1.0 / boost.gamma0**2)

    # Build kwargs in a backward-compatible way: only pass arguments that
    # exist in the currently checked-out FBPIC branch.
    ctor_params = inspect.signature(Simulation.__init__).parameters

    sim_kwargs = {
        "Nz": args.Nz,
        "zmax": args.zmax,
        "Nr": args.Nr,
        "rmax": args.rmax,
        "Nm": args.Nm,
        "dt": dt,
    }

    optional_kwargs = {
        "zmin": args.zmin,
        "v_comoving": v_comoving,
        "gamma_boost": boost.gamma0,
        "n_order": args.n_order,
        "use_cuda": args.use_cuda,
        "boundaries": {"z": "open", "r": "reflective"},
        "exchange_period": args.exchange_period,
        "clear_cupy_mempool_on_exchange": (not args.disable_cupy_mempool_free),
        "particle_shape": args.particle_shape,
        "verbose_level": 0,
    }

    for key, value in optional_kwargs.items():
        if key in ctor_params and value is not None:
            sim_kwargs[key] = value

    if args.disable_cupy_mempool_free and "clear_cupy_mempool_on_exchange" not in ctor_params:
        print("[bench] Note: current branch does not support clear_cupy_mempool_on_exchange; ignoring flag.")

    sim = Simulation(**sim_kwargs)

    dens_func = make_density_profile(
        ramp_up=args.ramp_up,
        plateau=args.plateau,
        ramp_down=args.ramp_down,
        n_e=args.n_e,
        w_matched=args.w_matched,
    )
    p_zmax = args.ramp_up + args.plateau + args.ramp_down

    # Plasma electrons + ions
    sim.add_new_species(
        q=-e,
        m=m_e,
        n=args.n_e,
        dens_func=dens_func,
        boost_positions_in_dens_func=True,
        p_zmin=args.p_zmin,
        p_zmax=p_zmax,
        p_rmax=args.p_rmax,
        p_nz=args.p_nz,
        p_nr=args.p_nr,
        p_nt=args.p_nt,
    )
    sim.add_new_species(
        q=e,
        m=m_p,
        n=args.n_e,
        dens_func=dens_func,
        boost_positions_in_dens_func=True,
        p_zmin=args.p_zmin,
        p_zmax=p_zmax,
        p_rmax=args.p_rmax,
        p_nz=args.p_nz,
        p_nr=args.p_nr,
        p_nt=args.p_nt,
    )

    if not args.disable_bunch:
        bunch_zmax = args.bunch_zmin + args.bunch_length
        add_particle_bunch(
            sim,
            -e,
            m_e,
            args.bunch_gamma,
            args.bunch_n,
            args.bunch_zmin,
            bunch_zmax,
            0.0,
            args.bunch_rmax,
            boost=boost,
        )

    if not args.disable_laser:
        laser_profile = GaussianLaser(
            args.a0,
            args.w0,
            args.tau,
            args.z0,
            lambda0=args.lambda0,
            zf=args.zfoc,
        )
        add_laser_pulse(
            sim,
            laser_profile,
            gamma_boost=boost.gamma0,
            method="antenna",
            z0_antenna=0.0,
        )

    # Moving window in boosted frame
    v_window = c * (1 - 0.5 * args.n_e / 1.75e27)
    v_window_boosted, = boost.velocity([v_window])
    sim.set_moving_window(v=v_window_boosted)

    # Explicitly disable diagnostics (no output I/O)
    sim.diags = []

    return sim


def make_gpu_sync(sim):
    if not sim.use_cuda:
        return lambda: None
    try:
        import cupy

        return cupy.cuda.Stream.null.synchronize
    except Exception:
        return lambda: None


def get_original_default_steps(args, dt):
    """Return N_step from the original boosted-frame example formula."""
    from fbpic.lpa_utils.boosted_frame import BoostConverter

    boost = BoostConverter(args.gamma_boost)
    p_zmax = args.ramp_up + args.plateau + args.ramp_down
    L_interact = p_zmax - args.p_zmin
    v_window = c * (1 - 0.5 * args.n_e / 1.75e27)
    T_interact = boost.interaction_time(L_interact, (args.zmax - args.zmin), v_window)
    return int(T_interact / dt)


def run_benchmark(args):
    sim = build_simulation(args)
    sync = make_gpu_sync(sim)

    n_particles = sum(species.Ntot for species in sim.ptcl)
    timed_steps = args.steps
    if timed_steps is None:
        timed_steps = get_original_default_steps(args, sim.dt)

    # Pre-cooking / warmup (not timed)
    if args.warmup_steps > 0:
        sim.step(args.warmup_steps, show_progress=False)
        sync()

    # Timed steady-state section
    t0 = time.perf_counter()
    sim.step(timed_steps, show_progress=False)
    sync()
    elapsed = time.perf_counter() - t0

    print("=== FBPIC Boosted Frame Benchmark (steady-state) ===")
    print(f"backend             : {'GPU' if sim.use_cuda else 'CPU'}")
    print(f"warmup_steps        : {args.warmup_steps}")
    print(f"timed_steps         : {timed_steps}")
    print(f"grid (Nz, Nr, Nm)   : ({args.Nz}, {args.Nr}, {args.Nm})")
    print(f"particle_shape      : {args.particle_shape}")
    print(f"particles (total)   : {n_particles}")
    print(f"timed runtime [s]   : {elapsed:.6f}")
    print(f"time / step [s]     : {elapsed / timed_steps:.6f}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Boosted-frame benchmark with warmup and no diagnostics"
    )

    # Timing
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="number of timed steps (default: original boosted-frame N_step)",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="number of warmup (pre-cooking) steps before timing",
    )

    # Runtime backend
    p.add_argument("--cpu", dest="use_cuda", action="store_false", help="force CPU mode")
    p.set_defaults(use_cuda=True)

    # Grid + algorithm (defaults match docs/source/example_input/boosted_frame_script.py)
    p.add_argument("--Nz", type=int, default=600)
    p.add_argument("--Nr", type=int, default=75)
    p.add_argument("--Nm", type=int, default=2)
    p.add_argument("--zmin", type=float, default=-30e-6)
    p.add_argument("--zmax", type=float, default=0.0)
    p.add_argument("--rmax", type=float, default=150e-6)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--n-order", type=int, default=-1)
    p.add_argument("--gamma-boost", type=float, default=10.0)
    p.add_argument("--particle-shape", choices=["linear", "cubic"], default="linear")
    p.add_argument("--exchange-period", type=int, default=None)
    p.add_argument(
        "--disable-cupy-mempool-free",
        action="store_true",
        help="disable CuPy mempool free_all_blocks after particle exchange",
    )

    # Plasma profile + particles
    p.add_argument("--n-e", dest="n_e", type=float, default=3e24)
    p.add_argument("--p-zmin", type=float, default=0.0)
    p.add_argument("--p-rmax", type=float, default=100e-6)
    p.add_argument("--p-nz", type=int, default=2)
    p.add_argument("--p-nr", type=int, default=2)
    p.add_argument("--p-nt", type=int, default=6)
    p.add_argument("--w-matched", type=float, default=50e-6)
    p.add_argument("--ramp-up", type=float, default=0.5e-3)
    p.add_argument("--plateau", type=float, default=3.5e-3)
    p.add_argument("--ramp-down", type=float, default=0.5e-3)

    # Laser
    p.add_argument("--disable-laser", action="store_true")
    p.add_argument("--a0", type=float, default=2.0)
    p.add_argument("--w0", type=float, default=50e-6)
    p.add_argument("--tau", type=float, default=16e-15)
    p.add_argument("--z0", type=float, default=-10e-6)
    p.add_argument("--zfoc", type=float, default=0.0)
    p.add_argument("--lambda0", type=float, default=0.8e-6)

    # Witness bunch
    p.add_argument("--disable-bunch", action="store_true")
    p.add_argument("--bunch-zmin", type=float, default=-25e-6)
    p.add_argument("--bunch-length", type=float, default=3e-6)
    p.add_argument("--bunch-rmax", type=float, default=10e-6)
    p.add_argument("--bunch-gamma", type=float, default=400.0)
    p.add_argument("--bunch-n", type=float, default=5e23)

    p.add_argument(
        "--disable-galilean",
        action="store_true",
        help="disable Galilean frame (v_comoving)",
    )

    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
