# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
"""

from numba import cuda
from fbpic.utils.cuda import compile_cupy
import math
from fbpic.particles.deposition.particle_shapes import Sz_linear, \
    Sr_linear, Sz_cubic, Sr_cubic

# JIT-compilation of particle shapes
Sz_linear = cuda.jit(Sz_linear, device=True, inline=False)
Sr_linear = cuda.jit(Sr_linear, device=True, inline=False)
Sz_cubic = cuda.jit(Sz_cubic, device=True, inline=False)
Sr_cubic = cuda.jit(Sr_cubic, device=True, inline=False)

@compile_cupy
def deposit_rho_gpu_unsorted(x, y, z, w, q,
                        invdz, zmin, Nz,
                        invdr, rmin, Nr,
                        rho_m, m, beta_n):
    """
    Deposition of the charge density rho using numba on the GPU.
    Iterates over all particles.
    Calculates the weighted amount of rho that is deposited to the
    4 cells surounding the particle based on its shape (linear).

    The particles do not have to be sorted. Thus this kernel is
    less parallel then the standard deposition kernels.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    w : 1d array of floats
        The weights of the particles
        (For ionizable atoms: weight times the ionization level)

    q : float
        Charge of the species
        (For ionizable atoms: this is always the elementary charge e)

    rho_m: 2darray of complexs
        The charge density on the interpolation grid for
        mode m. (is modified by this function)

    m: int
        The index of the azimuthal mode

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the considered direction

    Nz, Nr : int
        Number of gridpoints along the considered direction

    beta_n : 1darray of floats
        Ruyten-corrected particle shape factor coefficients
    """
    # Get the 1D CUDA grid
    i = cuda.grid(1)

    if i < w.shape[0]:

        # Sign for the shape factors
        f = (-1)**m
        # Preliminary arrays for the cylindrical conversion
        # --------------------------------------------
        # Position
        xj = x[i]
        yj = y[i]
        zj = z[i]
        # Weights
        wj = q * w[i]

        # Cylindrical conversion
        rj = math.sqrt(xj**2 + yj**2)
        # Avoid division by 0.
        if (rj != 0.):
            invr = 1./rj
            cos = xj*invr  # Cosine
            sin = yj*invr  # Sine
        else:
            cos = 1.
            sin = 0.
        # Calculate azimuthal factor
        exptheta_m = 1. + 0.j
        for _ in range(m):
            exptheta_m *= (cos + 1.j*sin)

        # Positions of the particles, in the cell unit
        r_cell = invdr*(rj - rmin) - 0.5
        z_cell = invdz*(zj - zmin) - 0.5
       
        # Cell indices of the upper cell bounds
        ir = min( int(math.ceil(r_cell)), Nr )
        iz = int(math.ceil(z_cell))
        # Handle periodic boundaries in z
        if iz < 0:
            iz += Nz
        elif iz >= Nz:
            iz -= Nz

        # Calculate longitudinal indices at which to add charge
        iz0 = iz - 1
        iz1 = iz
        if iz0 < 0:
            iz0 += Nz
        # Calculate radial indices at which to add charge
        ir0 = ir - 1
        ir1 = min( ir, Nr-1 )
        if ir0 < 0:
            # Deposition below the axis: fold index into physical region
            ir0 = -(1 + ir0)

        # Ruyten-corrected shape factor coefficient
        bn = beta_n[ir]

        # Calculate rho
        # --------------------------------------------
        R_m_scal = wj * exptheta_m
        R_m_00 = Sr_linear(r_cell, 0, f, bn)*Sz_linear(z_cell, 0) * R_m_scal
        R_m_01 = Sr_linear(r_cell, 0, f, bn)*Sz_linear(z_cell, 1) * R_m_scal
        R_m_10 = Sr_linear(r_cell, 1, f, bn)*Sz_linear(z_cell, 0) * R_m_scal
        R_m_11 = Sr_linear(r_cell, 1, f, bn)*Sz_linear(z_cell, 1) * R_m_scal
  
        # Add the calculated fields to the global grid
        cuda.atomic.add(rho_m.real, (iz0, ir0), R_m_00.real)
        cuda.atomic.add(rho_m.real, (iz0, ir1), R_m_10.real)
        cuda.atomic.add(rho_m.real, (iz1, ir0), R_m_01.real)
        cuda.atomic.add(rho_m.real, (iz1, ir1), R_m_11.real)
        if m > 0:
            # For azimuthal modes beyond m=0: add imaginary part
            cuda.atomic.add(rho_m.imag, (iz0, ir0), R_m_00.imag)
            cuda.atomic.add(rho_m.imag, (iz0, ir1), R_m_10.imag)
            cuda.atomic.add(rho_m.imag, (iz1, ir0), R_m_01.imag)
            cuda.atomic.add(rho_m.imag, (iz1, ir1), R_m_11.imag)

# -------------------------------
# Field deposition - linear - J
# -------------------------------

@compile_cupy
def deposit_rho_gpu_unsorted_cubic(x, y, z, w, q,
                        invdz, zmin, Nz,
                        invdr, rmin, Nr,
                        rho_m, m, beta_n):
    """
    Deposition of the charge density rho using numba on the GPU.
    Iterates over all particles and deposits with cubic shape factors.

    Parameters are identical to `deposit_rho_gpu_unsorted`, except that
    cubic shape factors are used (16-cell stencil).
    """
    # Get the 1D CUDA grid
    i = cuda.grid(1)

    if i < w.shape[0]:

        # Sign for the shape factors
        f = (-1)**m
        # Preliminary arrays for the cylindrical conversion
        # --------------------------------------------
        # Position
        xj = x[i]
        yj = y[i]
        zj = z[i]
        # Weights
        wj = q * w[i]

        # Cylindrical conversion
        rj = math.sqrt(xj**2 + yj**2)
        # Avoid division by 0.
        if (rj != 0.):
            invr = 1./rj
            cos = xj*invr  # Cosine
            sin = yj*invr  # Sine
        else:
            cos = 1.
            sin = 0.
        # Calculate azimuthal factor
        exptheta_m = 1. + 0.j
        for _ in range(m):
            exptheta_m *= (cos + 1.j*sin)

        # Positions of the particles, in the cell unit
        r_cell = invdr*(rj - rmin) - 0.5
        z_cell = invdz*(zj - zmin) - 0.5

        # Cell indices of the upper cell bounds
        ir = min( int(math.ceil(r_cell)), Nr )
        iz = int(math.ceil(z_cell))
        # Handle periodic boundaries in z
        if iz < 0:
            iz += Nz
        elif iz >= Nz:
            iz -= Nz

        # Calculate longitudinal indices at which to add charge
        iz0 = iz - 2
        iz1 = iz - 1
        iz2 = iz
        iz3 = iz + 1
        if iz0 < 0:
            iz0 += Nz
        if iz1 < 0:
            iz1 += Nz
        if iz3 > Nz-1:
            iz3 -= Nz

        # Calculate radial indices at which to add charge
        ir0 = ir - 2
        ir1 = min( ir - 1, Nr-1 )
        ir2 = min( ir    , Nr-1 )
        ir3 = min( ir + 1, Nr-1 )
        if ir0 < 0:
            # Deposition below the axis: fold index into physical region
            ir0 = -(1 + ir0)
        if ir1 < 0:
            # Deposition below the axis: fold index into physical region
            ir1 = -(1 + ir1)

        # Ruyten-corrected shape factor coefficient
        bn = beta_n[ir]

        # Precompute shape coefficients
        Sz0 = Sz_cubic(z_cell, 0)
        Sz1 = Sz_cubic(z_cell, 1)
        Sz2 = Sz_cubic(z_cell, 2)
        Sz3 = Sz_cubic(z_cell, 3)

        Sr0 = Sr_cubic(r_cell, 0, f, bn)
        Sr1 = Sr_cubic(r_cell, 1, f, bn)
        Sr2 = Sr_cubic(r_cell, 2, f, bn)
        Sr3 = Sr_cubic(r_cell, 3, f, bn)

        # Calculate rho
        R_m_scal = wj * exptheta_m

        R00 = Sr0*Sz0 * R_m_scal
        R01 = Sr0*Sz1 * R_m_scal
        R02 = Sr0*Sz2 * R_m_scal
        R03 = Sr0*Sz3 * R_m_scal

        R10 = Sr1*Sz0 * R_m_scal
        R11 = Sr1*Sz1 * R_m_scal
        R12 = Sr1*Sz2 * R_m_scal
        R13 = Sr1*Sz3 * R_m_scal

        R20 = Sr2*Sz0 * R_m_scal
        R21 = Sr2*Sz1 * R_m_scal
        R22 = Sr2*Sz2 * R_m_scal
        R23 = Sr2*Sz3 * R_m_scal

        R30 = Sr3*Sz0 * R_m_scal
        R31 = Sr3*Sz1 * R_m_scal
        R32 = Sr3*Sz2 * R_m_scal
        R33 = Sr3*Sz3 * R_m_scal

        # Add the calculated fields to the global grid
        cuda.atomic.add(rho_m.real, (iz0, ir0), R00.real)
        cuda.atomic.add(rho_m.real, (iz1, ir0), R01.real)
        cuda.atomic.add(rho_m.real, (iz2, ir0), R02.real)
        cuda.atomic.add(rho_m.real, (iz3, ir0), R03.real)

        cuda.atomic.add(rho_m.real, (iz0, ir1), R10.real)
        cuda.atomic.add(rho_m.real, (iz1, ir1), R11.real)
        cuda.atomic.add(rho_m.real, (iz2, ir1), R12.real)
        cuda.atomic.add(rho_m.real, (iz3, ir1), R13.real)

        cuda.atomic.add(rho_m.real, (iz0, ir2), R20.real)
        cuda.atomic.add(rho_m.real, (iz1, ir2), R21.real)
        cuda.atomic.add(rho_m.real, (iz2, ir2), R22.real)
        cuda.atomic.add(rho_m.real, (iz3, ir2), R23.real)

        cuda.atomic.add(rho_m.real, (iz0, ir3), R30.real)
        cuda.atomic.add(rho_m.real, (iz1, ir3), R31.real)
        cuda.atomic.add(rho_m.real, (iz2, ir3), R32.real)
        cuda.atomic.add(rho_m.real, (iz3, ir3), R33.real)

        if m > 0:
            # For azimuthal modes beyond m=0: add imaginary part
            cuda.atomic.add(rho_m.imag, (iz0, ir0), R00.imag)
            cuda.atomic.add(rho_m.imag, (iz1, ir0), R01.imag)
            cuda.atomic.add(rho_m.imag, (iz2, ir0), R02.imag)
            cuda.atomic.add(rho_m.imag, (iz3, ir0), R03.imag)

            cuda.atomic.add(rho_m.imag, (iz0, ir1), R10.imag)
            cuda.atomic.add(rho_m.imag, (iz1, ir1), R11.imag)
            cuda.atomic.add(rho_m.imag, (iz2, ir1), R12.imag)
            cuda.atomic.add(rho_m.imag, (iz3, ir1), R13.imag)

            cuda.atomic.add(rho_m.imag, (iz0, ir2), R20.imag)
            cuda.atomic.add(rho_m.imag, (iz1, ir2), R21.imag)
            cuda.atomic.add(rho_m.imag, (iz2, ir2), R22.imag)
            cuda.atomic.add(rho_m.imag, (iz3, ir2), R23.imag)

            cuda.atomic.add(rho_m.imag, (iz0, ir3), R30.imag)
            cuda.atomic.add(rho_m.imag, (iz1, ir3), R31.imag)
            cuda.atomic.add(rho_m.imag, (iz2, ir3), R32.imag)
            cuda.atomic.add(rho_m.imag, (iz3, ir3), R33.imag)

# -------------------------------
# Field deposition - linear - J
# -------------------------------

@compile_cupy
def deposit_J_gpu_unsorted(x, y, z, w, q,
                        vx, vy, vz,
                        invdz, zmin, Nz,
                        invdr, rmin, Nr,
                        j_r_m, j_t_m, j_z_m, m,
                        beta_n):
    """
    Deposition of the current J using numba on the GPU.
    Iterates over all particles.
    Calculates the weighted amount of J that is deposited to the
    4 cells surounding the particle based on its shape (linear).

    The particles do not have to be sorted. Thus this kernel is
    less parallel then the standard deposition kernels.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    w : 1d array of floats
        The weights of the particles
        (For ionizable atoms: weight times the ionization level)

    q : float
        Charge of the species
        (For ionizable atoms: this is always the elementary charge e)

    vx, vy, vz : 1darray of floats (in meters * second^-1)
        The non-relativistic velocity of the particles

    j_r_m, j_t_m, j_z_m,: 2darrays of complexs
        The current component in each direction (r, t, z)
        on the interpolation grid for mode m.
        (is modified by this function)

    m: int
        The index of the azimuthal mode considered

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    beta_n : 1darray of floats
        Ruyten-corrected particle shape factor coefficients
    """
    # Get the 1D CUDA grid
    i = cuda.grid(1)

    if i < w.shape[0]:

        # Sign for the shape factors
        f = (-1)**m

        # Preliminary arrays for the cylindrical conversion
        # --------------------------------------------
        # Position
        xj = x[i]
        yj = y[i]
        zj = z[i]
        # Velocity
        vxj = vx[i]
        vyj = vy[i]
        vzj = vz[i]
        # Weights
        wj = q * w[i]

        # Cylindrical conversion
        rj = math.sqrt(xj**2 + yj**2)
        # Avoid division by 0.
        if (rj != 0.):
            invr = 1./rj
            cos = xj*invr  # Cosine
            sin = yj*invr  # Sine
        else:
            cos = 1.
            sin = 0.
        # Calculate azimuthal factor
        exptheta_m = 1. + 0.j
        for _ in range(m):
            exptheta_m *= (cos + 1.j*sin)

        # Positions of the particles, in the cell unit
        r_cell = invdr*(rj - rmin) - 0.5
        z_cell = invdz*(zj - zmin) - 0.5
       
        # Cell indices of the upper cell bounds
        ir = min( int(math.ceil(r_cell)), Nr )
        iz = int(math.ceil(z_cell))
        # Handle periodic boundaries in z
        if iz < 0:
            iz += Nz
        elif iz >= Nz:
            iz -= Nz

        # Calculate longitudinal indices at which to add charge
        iz0 = iz - 1
        iz1 = iz
        if iz0 < 0:
            iz0 += Nz
        # Calculate radial indices at which to add charge
        ir0 = ir - 1
        ir1 = min( ir, Nr-1 )
        if ir0 < 0:
            # Deposition below the axis: fold index into physical region
            ir0 = -(1 + ir0)
        
        # Ruyten-corrected shape factor coefficient
        bn = beta_n[ir]

        # Calculate the currents
        # ----------------------
        J_r_m_scal = wj * (cos*vxj + sin*vyj) * exptheta_m
        J_t_m_scal = wj * (cos*vyj - sin*vxj) * exptheta_m
        J_z_m_scal = wj * vzj * exptheta_m

        J_t_m_00 = Sr_linear(r_cell, 0, -f, bn)*Sz_linear(z_cell, 0) * J_t_m_scal
        J_r_m_00 = Sr_linear(r_cell, 0, -f, bn)*Sz_linear(z_cell, 0) * J_r_m_scal
        J_z_m_00 = Sr_linear(r_cell, 0, f, bn)*Sz_linear(z_cell, 0) * J_z_m_scal
        J_r_m_01 = Sr_linear(r_cell, 0, -f, bn)*Sz_linear(z_cell, 1) * J_r_m_scal
        J_t_m_01 = Sr_linear(r_cell, 0, -f, bn)*Sz_linear(z_cell, 1) * J_t_m_scal
        J_z_m_01 = Sr_linear(r_cell, 0, f, bn)*Sz_linear(z_cell, 1) * J_z_m_scal

        J_r_m_10 = Sr_linear(r_cell, 1, -f, bn)*Sz_linear(z_cell, 0) * J_r_m_scal
        J_t_m_10 = Sr_linear(r_cell, 1, -f, bn)*Sz_linear(z_cell, 0) * J_t_m_scal
        J_z_m_10 = Sr_linear(r_cell, 1, f, bn)*Sz_linear(z_cell, 0) * J_z_m_scal
        J_r_m_11 = Sr_linear(r_cell, 1, -f, bn)*Sz_linear(z_cell, 1) * J_r_m_scal
        J_t_m_11 = Sr_linear(r_cell, 1, -f, bn)*Sz_linear(z_cell, 1) * J_t_m_scal
        J_z_m_11 = Sr_linear(r_cell, 1, f, bn)*Sz_linear(z_cell, 1) * J_z_m_scal

        # Atomically add the registers to global memory
        # jr
        cuda.atomic.add(j_r_m.real, (iz0, ir0), J_r_m_00.real)
        cuda.atomic.add(j_r_m.real, (iz0, ir1), J_r_m_10.real)
        cuda.atomic.add(j_r_m.real, (iz1, ir0), J_r_m_01.real)
        cuda.atomic.add(j_r_m.real, (iz1, ir1), J_r_m_11.real)
        if m > 0:
            cuda.atomic.add(j_r_m.imag, (iz0, ir0), J_r_m_00.imag)
            cuda.atomic.add(j_r_m.imag, (iz0, ir1), J_r_m_10.imag)
            cuda.atomic.add(j_r_m.imag, (iz1, ir0), J_r_m_01.imag)
            cuda.atomic.add(j_r_m.imag, (iz1, ir1), J_r_m_11.imag)
        # jt
        cuda.atomic.add(j_t_m.real, (iz0, ir0), J_t_m_00.real)
        cuda.atomic.add(j_t_m.real, (iz0, ir1), J_t_m_10.real)
        cuda.atomic.add(j_t_m.real, (iz1, ir0), J_t_m_01.real)
        cuda.atomic.add(j_t_m.real, (iz1, ir1), J_t_m_11.real)
        if m > 0:
            cuda.atomic.add(j_t_m.imag, (iz0, ir0), J_t_m_00.imag)
            cuda.atomic.add(j_t_m.imag, (iz0, ir1), J_t_m_10.imag)
            cuda.atomic.add(j_t_m.imag, (iz1, ir0), J_t_m_01.imag)
            cuda.atomic.add(j_t_m.imag, (iz1, ir1), J_t_m_11.imag)
        # jz
        cuda.atomic.add(j_z_m.real, (iz0, ir0), J_z_m_00.real)
        cuda.atomic.add(j_z_m.real, (iz0, ir1), J_z_m_10.real)
        cuda.atomic.add(j_z_m.real, (iz1, ir0), J_z_m_01.real)
        cuda.atomic.add(j_z_m.real, (iz1, ir1), J_z_m_11.real)
        if m > 0:
            cuda.atomic.add(j_z_m.imag, (iz0, ir0), J_z_m_00.imag)
            cuda.atomic.add(j_z_m.imag, (iz0, ir1), J_z_m_10.imag)
            cuda.atomic.add(j_z_m.imag, (iz1, ir0), J_z_m_01.imag)
            cuda.atomic.add(j_z_m.imag, (iz1, ir1), J_z_m_11.imag)