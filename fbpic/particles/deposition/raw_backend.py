# Copyright 2016, FBPIC contributors
# Authors: FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Experimental CuPy RawKernel/CUBIN backend for selected deposition kernels."""

import os
from pathlib import Path

import cupy
import numpy as np


_KERNEL_SOURCE_PATH = Path(__file__).with_name("kernels") / "deposition_nm3_raw.cu"
_KERNEL_SOURCE = _KERNEL_SOURCE_PATH.read_text()


_nvrtc_module = None
_cubin_module = None
_nvrtc_functions = {}
_cubin_functions = {}


def _get_cubin_path():
    path = os.environ.get("FBPIC_DEPOSITION_CUBIN_PATH")
    if not path:
        raise RuntimeError(
            "FBPIC_DEPOSITION_BACKEND='cubin' requires FBPIC_DEPOSITION_CUBIN_PATH"
        )
    return path


def _get_kernel_function(kernel_name, backend):
    global _nvrtc_module, _cubin_module

    if backend == "cupy_raw":
        fn = _nvrtc_functions.get(kernel_name)
        if fn is not None:
            return fn
        if _nvrtc_module is None:
            _nvrtc_module = cupy.RawModule(
                code=_KERNEL_SOURCE,
                options=("-std=c++11",),
                backend="nvrtc",
            )
        fn = _nvrtc_module.get_function(kernel_name)
        _nvrtc_functions[kernel_name] = fn
        return fn

    if backend == "cubin":
        fn = _cubin_functions.get(kernel_name)
        if fn is not None:
            return fn
        if _cubin_module is None:
            _cubin_module = cupy.RawModule(path=_get_cubin_path())
        fn = _cubin_module.get_function(kernel_name)
        _cubin_functions[kernel_name] = fn
        return fn

    raise ValueError(f"Unsupported deposition backend: {backend!r}")


def launch_deposit_rho_gpu_unsorted_cubic_m3(
        dim_grid_1d, dim_block_1d,
        x, y, z, w, q,
        invdz, zmin, Nz,
        invdr, rmin, Nr,
        rho_m0, rho_m1, rho_m2,
        beta_n_m0, beta_n_m1, beta_n_m2,
        backend="cupy_raw"):
    """Launch fused unsorted cubic rho deposition (Nm=3) via selected backend."""
    kernel = _get_kernel_function("deposit_rho_gpu_unsorted_cubic_m3_raw", backend)

    kernel(
        (dim_grid_1d,),
        (dim_block_1d,),
        (
            x, y, z, w,
            np.float64(q),
            np.float64(invdz), np.float64(zmin), np.int32(Nz),
            np.float64(invdr), np.float64(rmin), np.int32(Nr),
            rho_m0, rho_m1, rho_m2,
            beta_n_m0, beta_n_m1, beta_n_m2,
            np.int32(w.shape[0]),
        ),
    )


def launch_deposit_J_gpu_unsorted_rel_cubic_m3(
        dim_grid_1d, dim_block_1d,
        x, y, z, w, q,
        ux, uy, uz, inv_gamma,
        invdz, zmin, Nz,
        invdr, rmin, Nr,
        j_r_m0, j_t_m0, j_z_m0,
        j_r_m1, j_t_m1, j_z_m1,
        j_r_m2, j_t_m2, j_z_m2,
        beta_n_m0, beta_n_m1, beta_n_m2,
        backend="cupy_raw"):
    """Launch fused unsorted relativistic cubic J deposition (Nm=3)."""
    kernel = _get_kernel_function("deposit_J_gpu_unsorted_rel_cubic_m3_raw", backend)

    kernel(
        (dim_grid_1d,),
        (dim_block_1d,),
        (
            x, y, z, w,
            np.float64(q),
            ux, uy, uz, inv_gamma,
            np.float64(invdz), np.float64(zmin), np.int32(Nz),
            np.float64(invdr), np.float64(rmin), np.int32(Nr),
            j_r_m0, j_t_m0, j_z_m0,
            j_r_m1, j_t_m1, j_z_m1,
            j_r_m2, j_t_m2, j_z_m2,
            beta_n_m0, beta_n_m1, beta_n_m2,
            np.int32(w.shape[0]),
        ),
    )
