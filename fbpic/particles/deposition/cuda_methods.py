# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the deposition methods for rho and J for linear and cubic
order shapes on the GPU using CUDA.
"""
from numba import cuda
from fbpic.utils.cuda import compile_cupy
import math
from scipy.constants import c
import numpy as np
from fbpic.particles.deposition.particle_shapes import Sz_linear, \
    Sr_linear, Sz_cubic, Sr_cubic

# JIT-compilation of particle shapes
Sz_linear = cuda.jit(Sz_linear, device=True, inline=False)
Sr_linear = cuda.jit(Sr_linear, device=True, inline=False)
Sz_cubic = cuda.jit(Sz_cubic, device=True, inline=False)
Sr_cubic = cuda.jit(Sr_cubic, device=True, inline=False)

# -------------------------------
# Field deposition - linear - rho
# -------------------------------

@compile_cupy
def deposit_rho_gpu_linear(x, y, z, w, q,
                           invdz, zmin, Nz,
                           invdr, rmin, Nr,
                           rho_m0, rho_m1,
                           cell_idx, prefix_sum,
                           beta_n_m0, beta_n_m1):
    """
    Deposition of the charge density rho using numba on the GPU.
    Iterates over the cells and over the particles per cell.
    Calculates the weighted amount of rho that is deposited to the
    4 cells surounding the particle based on its shape (linear).

    The particles are sorted by their cell index (the lower cell
    in r and z that they deposit to) and the deposited field
    is split into 4 variables (one for each possible direction,
    e.g. upper in z, lower in r) to maintain parallelism while
    avoiding any race conditions.

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

    rho_m0, rho_m1 : 2darrays of complexs
        The charge density on the interpolation grid for
        mode 0 and 1. (is modified by this function)

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the considered direction

    Nz, Nr : int
        Number of gridpoints along the considered direction

    cell_idx : 1darray of integers
        The cell index of the particle

    prefix_sum : 1darray of integers
        Represents the cumulative sum of
        the particles per cell

    beta_n_m0, beta_n_m1 : 1darrays of floats
        Ruyten-corrected particle shape factor coefficients for mode 0 and 1
    """
    # Get the 1D CUDA grid
    i = cuda.grid(1)
    # Deposit the field per cell in parallel (for threads < number of cells)
    if i < prefix_sum.shape[0]:
        # Retrieve index of upper grid point (in z and r) from prefix-sum index
        # (See calculation of prefix-sum index in `get_cell_idx_per_particle`)
        iz_upper = int( i / (Nr+1) )
        ir_upper = int( i - iz_upper * (Nr+1) )
        # Calculate the inclusive offset for the current cell
        # It represents the number of particles contained in all other cells
        # with an index smaller than i + the total number of particles in the
        # current cell (inclusive).
        incl_offset = np.int32(prefix_sum[i])
        # Calculate the frequency per cell from the offset and the previous
        # offset (prefix_sum[i-1]).
        if i > 0:
            frequency_per_cell = np.int32(incl_offset - prefix_sum[i - 1])
        if i == 0:
            frequency_per_cell = np.int32(incl_offset)

        # Declare local field arrays
        R_m0_00 = 0.
        R_m0_01 = 0.
        R_m0_10 = 0.
        R_m0_11 = 0.
        R_m1_00 = 0. + 0.j
        R_m1_01 = 0. + 0.j
        R_m1_10 = 0. + 0.j
        R_m1_11 = 0. + 0.j

        for j in range(frequency_per_cell):
            # Get the particle index before the sorting
            # --------------------------------------------
            # (Since incl_offset is a cumulative sum of particle number,
            # and since python index starts at 0, one has to add -1)
            ptcl_idx = incl_offset-1-j

            # Preliminary arrays for the cylindrical conversion
            # --------------------------------------------
            # Position
            xj = x[ptcl_idx]
            yj = y[ptcl_idx]
            zj = z[ptcl_idx]
            # Weights
            wj = q * w[ptcl_idx]

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
            exptheta_m0 = 1.
            exptheta_m1 = cos + 1.j*sin

            # Positions of the particles, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5

            # Ruyten-corrected shape factor coefficients for both modes
            ir = min( int(math.ceil(r_cell)), Nr )
            bn_m0 = beta_n_m0[ir]
            bn_m1 = beta_n_m1[ir]

            # Calculate rho
            # --------------------------------------------
            # Mode 0
            R_m0_scal = wj * exptheta_m0
            # Mode 1
            R_m1_scal = wj * exptheta_m1

            R_m0_00 += Sr_linear(r_cell, 0, 1, bn_m0)*Sz_linear(z_cell, 0) * R_m0_scal
            R_m0_01 += Sr_linear(r_cell, 0, 1, bn_m0)*Sz_linear(z_cell, 1) * R_m0_scal
            R_m1_00 += Sr_linear(r_cell, 0,-1, bn_m1)*Sz_linear(z_cell, 0) * R_m1_scal
            R_m1_01 += Sr_linear(r_cell, 0,-1, bn_m1)*Sz_linear(z_cell, 1) * R_m1_scal
            R_m0_10 += Sr_linear(r_cell, 1, 1, bn_m0)*Sz_linear(z_cell, 0) * R_m0_scal
            R_m0_11 += Sr_linear(r_cell, 1, 1, bn_m0)*Sz_linear(z_cell, 1) * R_m0_scal
            R_m1_10 += Sr_linear(r_cell, 1,-1, bn_m1)*Sz_linear(z_cell, 0) * R_m1_scal
            R_m1_11 += Sr_linear(r_cell, 1,-1, bn_m1)*Sz_linear(z_cell, 1) * R_m1_scal

        # Calculate longitudinal indices at which to add charge
        iz0 = iz_upper - 1
        iz1 = iz_upper
        if iz0 < 0:
            iz0 += Nz
        # Calculate radial indices at which to add charge
        ir0 = ir_upper - 1
        ir1 = min( ir_upper, Nr-1 )
        if ir0 < 0:
            # Deposition below the axis: fold index into physical region
            ir0 = -(1 + ir0)

        # Atomically add the registers to global memory
        if frequency_per_cell > 0:
            # Mode 0
            cuda.atomic.add(rho_m0.real, (iz0, ir0), R_m0_00.real)
            cuda.atomic.add(rho_m0.real, (iz0, ir1), R_m0_10.real)
            cuda.atomic.add(rho_m0.real, (iz1, ir0), R_m0_01.real)
            cuda.atomic.add(rho_m0.real, (iz1, ir1), R_m0_11.real)
            # Mode 1
            cuda.atomic.add(rho_m1.real, (iz0, ir0), R_m1_00.real)
            cuda.atomic.add(rho_m1.imag, (iz0, ir0), R_m1_00.imag)
            cuda.atomic.add(rho_m1.real, (iz0, ir1), R_m1_10.real)
            cuda.atomic.add(rho_m1.imag, (iz0, ir1), R_m1_10.imag)
            cuda.atomic.add(rho_m1.real, (iz1, ir0), R_m1_01.real)
            cuda.atomic.add(rho_m1.imag, (iz1, ir0), R_m1_01.imag)
            cuda.atomic.add(rho_m1.real, (iz1, ir1), R_m1_11.real)
            cuda.atomic.add(rho_m1.imag, (iz1, ir1), R_m1_11.imag)


# -------------------------------
# Field deposition - linear - J
# -------------------------------

@compile_cupy
def deposit_J_gpu_linear(x, y, z, w, q,
                         ux, uy, uz, inv_gamma,
                         invdz, zmin, Nz,
                         invdr, rmin, Nr,
                         j_r_m0, j_r_m1,
                         j_t_m0, j_t_m1,
                         j_z_m0, j_z_m1,
                         cell_idx, prefix_sum,
                         beta_n_m0, beta_n_m1):
    """
    Deposition of the current J using numba on the GPU.
    Iterates over the cells and over the particles per cell.
    Calculates the weighted amount of J that is deposited to the
    4 cells surounding the particle based on its shape (linear).

    The particles are sorted by their cell index (the lower cell
    in r and z that they deposit to) and the deposited field
    is split into 4 variables (one for each possible direction,
    e.g. upper in z, lower in r) to maintain parallelism while
    avoiding any race conditions.

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

    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    j_r_m0, j_r_m1, j_t_m0, j_t_m1, j_z_m0, j_z_m1,: 2darrays of complexs
        The current component in each direction (r, t, z)
        on the interpolation grid for mode 0 and 1.
        (is modified by this function)

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    cell_idx : 1darray of integers
        The cell index of the particle

    prefix_sum : 1darray of integers
        Represents the cumulative sum of
        the particles per cell

    beta_n_m0, beta_n_m1 : 1darrays of floats
        Ruyten-corrected particle shape factor coefficients for mode 0 and 1
    """
    # Get the 1D CUDA grid
    i = cuda.grid(1)
    # Deposit the field per cell in parallel (for threads < number of cells)
    if i < prefix_sum.shape[0]:
        # Retrieve index of upper grid point (in z and r) from prefix-sum index
        # (See calculation of prefix-sum index in `get_cell_idx_per_particle`)
        iz_upper = int( i / (Nr+1) )
        ir_upper = int( i - iz_upper * (Nr+1) )
        # Calculate the inclusive offset for the current cell
        # It represents the number of particles contained in all other cells
        # with an index smaller than i + the total number of particles in the
        # current cell (inclusive).
        incl_offset = np.int32(prefix_sum[i])
        # Calculate the frequency per cell from the offset and the previous
        # offset (prefix_sum[i-1]).
        if i > 0:
            frequency_per_cell = np.int32(incl_offset - prefix_sum[i-1])
        if i == 0:
            frequency_per_cell = np.int32(incl_offset)

        # Declare the local field value for
        # all possible deposition directions,
        # depending on the shape order and per mode for r,t and z.

        J_r_m0_00 = 0.
        J_r_m1_00 = 0. + 0.j
        J_t_m0_00 = 0.
        J_t_m1_00 = 0. + 0.j
        J_z_m0_00 = 0.
        J_z_m1_00 = 0. + 0.j

        J_r_m0_01 = 0.
        J_r_m1_01 = 0. + 0.j
        J_t_m0_01 = 0.
        J_t_m1_01 = 0. + 0.j
        J_z_m0_01 = 0.
        J_z_m1_01 = 0. + 0.j

        J_r_m0_10 = 0.
        J_r_m1_10 = 0. + 0.j
        J_t_m0_10 = 0.
        J_t_m1_10 = 0. + 0.j
        J_z_m0_10 = 0.
        J_z_m1_10 = 0. + 0.j

        J_r_m0_11 = 0.
        J_r_m1_11 = 0. + 0.j
        J_t_m0_11 = 0.
        J_t_m1_11 = 0. + 0.j
        J_z_m0_11 = 0.
        J_z_m1_11 = 0. + 0.j


        # Loop over the number of particles per cell
        for j in range(frequency_per_cell):
            # Get the particle index
            # ----------------------
            # (Since incl_offset is a cumulative sum of particle number,
            # and since python index starts at 0, one has to add -1)
            ptcl_idx = incl_offset-1-j

            # Preliminary arrays for the cylindrical conversion
            # --------------------------------------------
            # Position
            xj = x[ptcl_idx]
            yj = y[ptcl_idx]
            zj = z[ptcl_idx]
            # Velocity
            uxj = ux[ptcl_idx]
            uyj = uy[ptcl_idx]
            uzj = uz[ptcl_idx]
            # Inverse gamma
            inv_gammaj = inv_gamma[ptcl_idx]
            # Weights
            wj = q * w[ptcl_idx]

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
            exptheta_m0 = 1.
            exptheta_m1 = cos + 1.j*sin

            # Get weights for the deposition
            # --------------------------------------------
            # Positions of the particles, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5

            # Ruyten-corrected shape factor coefficients for both modes
            ir = min( int(math.ceil(r_cell)), Nr )
            bn_m0 = beta_n_m0[ir]
            bn_m1 = beta_n_m1[ir]

            # Calculate the currents
            # ----------------------
            # Mode 0
            J_r_m0_scal = wj * c * inv_gammaj*(cos*uxj + sin*uyj) * exptheta_m0
            J_t_m0_scal = wj * c * inv_gammaj*(cos*uyj - sin*uxj) * exptheta_m0
            J_z_m0_scal = wj * c * inv_gammaj*uzj * exptheta_m0
            # Mode 1
            J_r_m1_scal = wj * c * inv_gammaj*(cos*uxj + sin*uyj) * exptheta_m1
            J_t_m1_scal = wj * c * inv_gammaj*(cos*uyj - sin*uxj) * exptheta_m1
            J_z_m1_scal = wj * c * inv_gammaj*uzj * exptheta_m1

            J_r_m0_00 += Sr_linear(r_cell, 0,-1, bn_m0)*Sz_linear(z_cell, 0) * J_r_m0_scal
            J_t_m0_00 += Sr_linear(r_cell, 0,-1, bn_m0)*Sz_linear(z_cell, 0) * J_t_m0_scal
            J_z_m0_00 += Sr_linear(r_cell, 0, 1, bn_m0)*Sz_linear(z_cell, 0) * J_z_m0_scal
            J_r_m0_01 += Sr_linear(r_cell, 0,-1, bn_m0)*Sz_linear(z_cell, 1) * J_r_m0_scal
            J_t_m0_01 += Sr_linear(r_cell, 0,-1, bn_m0)*Sz_linear(z_cell, 1) * J_t_m0_scal
            J_z_m0_01 += Sr_linear(r_cell, 0, 1, bn_m0)*Sz_linear(z_cell, 1) * J_z_m0_scal
            J_r_m1_00 += Sr_linear(r_cell, 0, 1, bn_m1)*Sz_linear(z_cell, 0) * J_r_m1_scal
            J_t_m1_00 += Sr_linear(r_cell, 0, 1, bn_m1)*Sz_linear(z_cell, 0) * J_t_m1_scal
            J_z_m1_00 += Sr_linear(r_cell, 0,-1, bn_m1)*Sz_linear(z_cell, 0) * J_z_m1_scal
            J_r_m1_01 += Sr_linear(r_cell, 0, 1, bn_m1)*Sz_linear(z_cell, 1) * J_r_m1_scal
            J_t_m1_01 += Sr_linear(r_cell, 0, 1, bn_m1)*Sz_linear(z_cell, 1) * J_t_m1_scal
            J_z_m1_01 += Sr_linear(r_cell, 0,-1, bn_m1)*Sz_linear(z_cell, 1) * J_z_m1_scal

            J_r_m0_10 += Sr_linear(r_cell, 1,-1, bn_m0)*Sz_linear(z_cell, 0) * J_r_m0_scal
            J_t_m0_10 += Sr_linear(r_cell, 1,-1, bn_m0)*Sz_linear(z_cell, 0) * J_t_m0_scal
            J_z_m0_10 += Sr_linear(r_cell, 1, 1, bn_m0)*Sz_linear(z_cell, 0) * J_z_m0_scal
            J_r_m0_11 += Sr_linear(r_cell, 1,-1, bn_m0)*Sz_linear(z_cell, 1) * J_r_m0_scal
            J_t_m0_11 += Sr_linear(r_cell, 1,-1, bn_m0)*Sz_linear(z_cell, 1) * J_t_m0_scal
            J_z_m0_11 += Sr_linear(r_cell, 1, 1, bn_m0)*Sz_linear(z_cell, 1) * J_z_m0_scal
            J_r_m1_10 += Sr_linear(r_cell, 1, 1, bn_m1)*Sz_linear(z_cell, 0) * J_r_m1_scal
            J_t_m1_10 += Sr_linear(r_cell, 1, 1, bn_m1)*Sz_linear(z_cell, 0) * J_t_m1_scal
            J_z_m1_10 += Sr_linear(r_cell, 1,-1, bn_m1)*Sz_linear(z_cell, 0) * J_z_m1_scal
            J_r_m1_11 += Sr_linear(r_cell, 1, 1, bn_m1)*Sz_linear(z_cell, 1) * J_r_m1_scal
            J_t_m1_11 += Sr_linear(r_cell, 1, 1, bn_m1)*Sz_linear(z_cell, 1) * J_t_m1_scal
            J_z_m1_11 += Sr_linear(r_cell, 1,-1, bn_m1)*Sz_linear(z_cell, 1) * J_z_m1_scal

        # Calculate longitudinal indices at which to add charge
        iz0 = iz_upper - 1
        iz1 = iz_upper
        if iz0 < 0:
            iz0 += Nz
        # Calculate radial indices at which to add charge
        ir0 = ir_upper - 1
        ir1 = min( ir_upper, Nr-1 )
        if ir0 < 0:
            # Deposition below the axis: fold index into physical region
            ir0 = -(1 + ir0)

        # Atomically add the registers to global memory
        if frequency_per_cell > 0:
            # jr: Mode 0
            cuda.atomic.add(j_r_m0.real, (iz0, ir0), J_r_m0_00.real)
            cuda.atomic.add(j_r_m0.real, (iz0, ir1), J_r_m0_10.real)
            cuda.atomic.add(j_r_m0.real, (iz1, ir0), J_r_m0_01.real)
            cuda.atomic.add(j_r_m0.real, (iz1, ir1), J_r_m0_11.real)
            # jr: Mode 1
            cuda.atomic.add(j_r_m1.real, (iz0, ir0), J_r_m1_00.real)
            cuda.atomic.add(j_r_m1.imag, (iz0, ir0), J_r_m1_00.imag)
            cuda.atomic.add(j_r_m1.real, (iz0, ir1), J_r_m1_10.real)
            cuda.atomic.add(j_r_m1.imag, (iz0, ir1), J_r_m1_10.imag)
            cuda.atomic.add(j_r_m1.real, (iz1, ir0), J_r_m1_01.real)
            cuda.atomic.add(j_r_m1.imag, (iz1, ir0), J_r_m1_01.imag)
            cuda.atomic.add(j_r_m1.real, (iz1, ir1), J_r_m1_11.real)
            cuda.atomic.add(j_r_m1.imag, (iz1, ir1), J_r_m1_11.imag)
            # jt: Mode 0
            cuda.atomic.add(j_t_m0.real, (iz0, ir0), J_t_m0_00.real)
            cuda.atomic.add(j_t_m0.real, (iz0, ir1), J_t_m0_10.real)
            cuda.atomic.add(j_t_m0.real, (iz1, ir0), J_t_m0_01.real)
            cuda.atomic.add(j_t_m0.real, (iz1, ir1), J_t_m0_11.real)
            # jt: Mode 1
            cuda.atomic.add(j_t_m1.real, (iz0, ir0), J_t_m1_00.real)
            cuda.atomic.add(j_t_m1.imag, (iz0, ir0), J_t_m1_00.imag)
            cuda.atomic.add(j_t_m1.real, (iz0, ir1), J_t_m1_10.real)
            cuda.atomic.add(j_t_m1.imag, (iz0, ir1), J_t_m1_10.imag)
            cuda.atomic.add(j_t_m1.real, (iz1, ir0), J_t_m1_01.real)
            cuda.atomic.add(j_t_m1.imag, (iz1, ir0), J_t_m1_01.imag)
            cuda.atomic.add(j_t_m1.real, (iz1, ir1), J_t_m1_11.real)
            cuda.atomic.add(j_t_m1.imag, (iz1, ir1), J_t_m1_11.imag)
            # jz: Mode 0
            cuda.atomic.add(j_z_m0.real, (iz0, ir0), J_z_m0_00.real)
            cuda.atomic.add(j_z_m0.real, (iz0, ir1), J_z_m0_10.real)
            cuda.atomic.add(j_z_m0.real, (iz1, ir0), J_z_m0_01.real)
            cuda.atomic.add(j_z_m0.real, (iz1, ir1), J_z_m0_11.real)
            # jz: Mode 1
            cuda.atomic.add(j_z_m1.real, (iz0, ir0), J_z_m1_00.real)
            cuda.atomic.add(j_z_m1.imag, (iz0, ir0), J_z_m1_00.imag)
            cuda.atomic.add(j_z_m1.real, (iz0, ir1), J_z_m1_10.real)
            cuda.atomic.add(j_z_m1.imag, (iz0, ir1), J_z_m1_10.imag)
            cuda.atomic.add(j_z_m1.real, (iz1, ir0), J_z_m1_01.real)
            cuda.atomic.add(j_z_m1.imag, (iz1, ir0), J_z_m1_01.imag)
            cuda.atomic.add(j_z_m1.real, (iz1, ir1), J_z_m1_11.real)
            cuda.atomic.add(j_z_m1.imag, (iz1, ir1), J_z_m1_11.imag)

# -------------------------------
# Field deposition - cubic - rho
# -------------------------------

@compile_cupy
def deposit_rho_gpu_cubic(x, y, z, w, q,
                          invdz, zmin, Nz,
                          invdr, rmin, Nr,
                          rho_m0, rho_m1,
                          cell_idx, prefix_sum,
                          beta_n_m0, beta_n_m1):
    """
    Deposition of the charge density rho using numba on the GPU.
    Iterates over the cells and over the particles per cell.
    Calculates the weighted amount of rho that is deposited to the
    16 cells surounding the particle based on its shape (cubic).

    The particles are sorted by their cell index (the lower cell
    in r and z that they deposit to) and the deposited field
    is split into 16 variables (one for each surrounding cell) to
    maintain parallelism while avoiding any race conditions.

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

    rho_m0, rho_m1 : 2darrays of complexs
        The charge density on the interpolation grid for
        mode 0 and 1. (is modified by this function)

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the considered direction

    Nz, Nr : int
        Number of gridpoints along the considered direction

    cell_idx : 1darray of integers
        The cell index of the particle

    prefix_sum : 1darray of integers
        Represents the cumulative sum of
        the particles per cell

    beta_n_m0, beta_n_m1 : 1darrays of floats
        Ruyten-corrected particle shape factor coefficients for mode 0 and 1
    """
    # Get the 1D CUDA grid
    i = cuda.grid(1)
    # Deposit the field per cell in parallel (for threads < number of cells)
    if i < prefix_sum.shape[0]:
        # Retrieve index of upper grid point (in z and r) from prefix-sum index
        # (See calculation of prefix-sum index in `get_cell_idx_per_particle`)
        iz_upper = int( i / (Nr+1) )
        ir_upper = int( i - iz_upper * (Nr+1) )
        # Calculate the inclusive offset for the current cell
        # It represents the number of particles contained in all other cells
        # with an index smaller than i + the total number of particles in the
        # current cell (inclusive).
        incl_offset = np.int32(prefix_sum[i])
        # Calculate the frequency per cell from the offset and the previous
        # offset (prefix_sum[i-1]).
        if i > 0:
            frequency_per_cell = np.int32(incl_offset - prefix_sum[i - 1])
        if i == 0:
            frequency_per_cell = np.int32(incl_offset)

        # Declare local field arrays
        R_m0_00 = 0.
        R_m1_00 = 0. + 0.j

        R_m0_01 = 0.
        R_m1_01 = 0. + 0.j

        R_m0_02 = 0.
        R_m1_02 = 0. + 0.j

        R_m0_03 = 0.
        R_m1_03 = 0. + 0.j

        R_m0_10 = 0.
        R_m1_10 = 0. + 0.j

        R_m0_11 = 0.
        R_m1_11 = 0. + 0.j

        R_m0_12 = 0.
        R_m1_12 = 0. + 0.j

        R_m0_13 = 0.
        R_m1_13 = 0. + 0.j

        R_m0_20 = 0.
        R_m1_20 = 0. + 0.j

        R_m0_21 = 0.
        R_m1_21 = 0. + 0.j

        R_m0_22 = 0.
        R_m1_22 = 0. + 0.j

        R_m0_23 = 0.
        R_m1_23 = 0. + 0.j

        R_m0_30 = 0.
        R_m1_30 = 0. + 0.j

        R_m0_31 = 0.
        R_m1_31 = 0. + 0.j

        R_m0_32 = 0.
        R_m1_32 = 0. + 0.j

        R_m0_33 = 0.
        R_m1_33 = 0. + 0.j

        for j in range(frequency_per_cell):
            # Get the particle index before the sorting
            # --------------------------------------------
            # (Since incl_offset is a cumulative sum of particle number,
            # and since python index starts at 0, one has to add -1)
            ptcl_idx = incl_offset-1-j

            # Preliminary arrays for the cylindrical conversion
            # --------------------------------------------
            # Position
            xj = x[ptcl_idx]
            yj = y[ptcl_idx]
            zj = z[ptcl_idx]
            # Weights
            wj = q * w[ptcl_idx]

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
            exptheta_m0 = 1.
            exptheta_m1 = cos + 1.j*sin

            # Positions of the particles, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5

            # Ruyten-corrected shape factor coefficients for both modes
            ir = min( int(math.ceil(r_cell)), Nr )
            bn_m0 = beta_n_m0[ir]
            bn_m1 = beta_n_m1[ir]

            # Calculate rho
            # -------------
            # Mode 0
            R_m0_scal = wj * exptheta_m0
            # Mode 1
            R_m1_scal = wj * exptheta_m1

            R_m0_00 += Sr_cubic(r_cell, 0, 1, bn_m0)*Sz_cubic(z_cell, 0)*R_m0_scal
            R_m1_00 += Sr_cubic(r_cell, 0,-1, bn_m1)*Sz_cubic(z_cell, 0)*R_m1_scal
            R_m0_01 += Sr_cubic(r_cell, 0, 1, bn_m0)*Sz_cubic(z_cell, 1)*R_m0_scal
            R_m1_01 += Sr_cubic(r_cell, 0,-1, bn_m1)*Sz_cubic(z_cell, 1)*R_m1_scal
            R_m0_02 += Sr_cubic(r_cell, 0, 1, bn_m0)*Sz_cubic(z_cell, 2)*R_m0_scal
            R_m1_02 += Sr_cubic(r_cell, 0,-1, bn_m1)*Sz_cubic(z_cell, 2)*R_m1_scal
            R_m0_03 += Sr_cubic(r_cell, 0, 1, bn_m0)*Sz_cubic(z_cell, 3)*R_m0_scal
            R_m1_03 += Sr_cubic(r_cell, 0,-1, bn_m1)*Sz_cubic(z_cell, 3)*R_m1_scal

            R_m0_10 += Sr_cubic(r_cell, 1, 1, bn_m0)*Sz_cubic(z_cell, 0)*R_m0_scal
            R_m1_10 += Sr_cubic(r_cell, 1,-1, bn_m1)*Sz_cubic(z_cell, 0)*R_m1_scal
            R_m0_11 += Sr_cubic(r_cell, 1, 1, bn_m0)*Sz_cubic(z_cell, 1)*R_m0_scal
            R_m1_11 += Sr_cubic(r_cell, 1,-1, bn_m1)*Sz_cubic(z_cell, 1)*R_m1_scal
            R_m0_12 += Sr_cubic(r_cell, 1, 1, bn_m0)*Sz_cubic(z_cell, 2)*R_m0_scal
            R_m1_12 += Sr_cubic(r_cell, 1,-1, bn_m1)*Sz_cubic(z_cell, 2)*R_m1_scal
            R_m0_13 += Sr_cubic(r_cell, 1, 1, bn_m0)*Sz_cubic(z_cell, 3)*R_m0_scal
            R_m1_13 += Sr_cubic(r_cell, 1,-1, bn_m1)*Sz_cubic(z_cell, 3)*R_m1_scal

            R_m0_20 += Sr_cubic(r_cell, 2, 1, bn_m0)*Sz_cubic(z_cell, 0)*R_m0_scal
            R_m1_20 += Sr_cubic(r_cell, 2,-1, bn_m1)*Sz_cubic(z_cell, 0)*R_m1_scal
            R_m0_21 += Sr_cubic(r_cell, 2, 1, bn_m0)*Sz_cubic(z_cell, 1)*R_m0_scal
            R_m1_21 += Sr_cubic(r_cell, 2,-1, bn_m1)*Sz_cubic(z_cell, 1)*R_m1_scal
            R_m0_22 += Sr_cubic(r_cell, 2, 1, bn_m0)*Sz_cubic(z_cell, 2)*R_m0_scal
            R_m1_22 += Sr_cubic(r_cell, 2,-1, bn_m1)*Sz_cubic(z_cell, 2)*R_m1_scal
            R_m0_23 += Sr_cubic(r_cell, 2, 1, bn_m0)*Sz_cubic(z_cell, 3)*R_m0_scal
            R_m1_23 += Sr_cubic(r_cell, 2,-1, bn_m1)*Sz_cubic(z_cell, 3)*R_m1_scal

            R_m0_30 += Sr_cubic(r_cell, 3, 1, bn_m0)*Sz_cubic(z_cell, 0)*R_m0_scal
            R_m1_30 += Sr_cubic(r_cell, 3,-1, bn_m1)*Sz_cubic(z_cell, 0)*R_m1_scal
            R_m0_31 += Sr_cubic(r_cell, 3, 1, bn_m0)*Sz_cubic(z_cell, 1)*R_m0_scal
            R_m1_31 += Sr_cubic(r_cell, 3,-1, bn_m1)*Sz_cubic(z_cell, 1)*R_m1_scal
            R_m0_32 += Sr_cubic(r_cell, 3, 1, bn_m0)*Sz_cubic(z_cell, 2)*R_m0_scal
            R_m1_32 += Sr_cubic(r_cell, 3,-1, bn_m1)*Sz_cubic(z_cell, 2)*R_m1_scal
            R_m0_33 += Sr_cubic(r_cell, 3, 1, bn_m0)*Sz_cubic(z_cell, 3)*R_m0_scal
            R_m1_33 += Sr_cubic(r_cell, 3,-1, bn_m1)*Sz_cubic(z_cell, 3)*R_m1_scal

        # Calculate longitudinal indices at which to add charge
        iz0 = iz_upper - 2
        iz1 = iz_upper - 1
        iz2 = iz_upper
        iz3 = iz_upper + 1
        if iz0 < 0:
            iz0 += Nz
        if iz1 < 0:
            iz1 += Nz
        if iz3 > Nz-1:
            iz3 -= Nz
        # Calculate radial indices at which to add charge
        ir0 = ir_upper - 2
        ir1 = min( ir_upper - 1, Nr-1 )
        ir2 = min( ir_upper    , Nr-1 )
        ir3 = min( ir_upper + 1, Nr-1 )
        if ir0 < 0:
            # Deposition below the axis: fold index into physical region
            ir0 = -(1 + ir0)
        if ir1 < 0:
            # Deposition below the axis: fold index into physical region
            ir1 = -(1 + ir1)

        # Atomically add the registers to global memory
        if frequency_per_cell > 0:
            # Mode 0
            cuda.atomic.add(rho_m0.real, (iz0, ir0), R_m0_00.real)
            cuda.atomic.add(rho_m0.real, (iz0, ir1), R_m0_10.real)
            cuda.atomic.add(rho_m0.real, (iz0, ir2), R_m0_20.real)
            cuda.atomic.add(rho_m0.real, (iz0, ir3), R_m0_30.real)
            cuda.atomic.add(rho_m0.real, (iz1, ir0), R_m0_01.real)
            cuda.atomic.add(rho_m0.real, (iz1, ir1), R_m0_11.real)
            cuda.atomic.add(rho_m0.real, (iz1, ir2), R_m0_21.real)
            cuda.atomic.add(rho_m0.real, (iz1, ir3), R_m0_31.real)
            cuda.atomic.add(rho_m0.real, (iz2, ir0), R_m0_02.real)
            cuda.atomic.add(rho_m0.real, (iz2, ir1), R_m0_12.real)
            cuda.atomic.add(rho_m0.real, (iz2, ir2), R_m0_22.real)
            cuda.atomic.add(rho_m0.real, (iz2, ir3), R_m0_32.real)
            cuda.atomic.add(rho_m0.real, (iz3, ir0), R_m0_03.real)
            cuda.atomic.add(rho_m0.real, (iz3, ir1), R_m0_13.real)
            cuda.atomic.add(rho_m0.real, (iz3, ir2), R_m0_23.real)
            cuda.atomic.add(rho_m0.real, (iz3, ir3), R_m0_33.real)
            # Mode 1
            cuda.atomic.add(rho_m1.real, (iz0, ir0), R_m1_00.real)
            cuda.atomic.add(rho_m1.imag, (iz0, ir0), R_m1_00.imag)
            cuda.atomic.add(rho_m1.real, (iz0, ir1), R_m1_10.real)
            cuda.atomic.add(rho_m1.imag, (iz0, ir1), R_m1_10.imag)
            cuda.atomic.add(rho_m1.real, (iz0, ir2), R_m1_20.real)
            cuda.atomic.add(rho_m1.imag, (iz0, ir2), R_m1_20.imag)
            cuda.atomic.add(rho_m1.real, (iz0, ir3), R_m1_30.real)
            cuda.atomic.add(rho_m1.imag, (iz0, ir3), R_m1_30.imag)
            cuda.atomic.add(rho_m1.real, (iz1, ir0), R_m1_01.real)
            cuda.atomic.add(rho_m1.imag, (iz1, ir0), R_m1_01.imag)
            cuda.atomic.add(rho_m1.real, (iz1, ir1), R_m1_11.real)
            cuda.atomic.add(rho_m1.imag, (iz1, ir1), R_m1_11.imag)
            cuda.atomic.add(rho_m1.real, (iz1, ir2), R_m1_21.real)
            cuda.atomic.add(rho_m1.imag, (iz1, ir2), R_m1_21.imag)
            cuda.atomic.add(rho_m1.real, (iz1, ir3), R_m1_31.real)
            cuda.atomic.add(rho_m1.imag, (iz1, ir3), R_m1_31.imag)
            cuda.atomic.add(rho_m1.real, (iz2, ir0), R_m1_02.real)
            cuda.atomic.add(rho_m1.imag, (iz2, ir0), R_m1_02.imag)
            cuda.atomic.add(rho_m1.real, (iz2, ir1), R_m1_12.real)
            cuda.atomic.add(rho_m1.imag, (iz2, ir1), R_m1_12.imag)
            cuda.atomic.add(rho_m1.real, (iz2, ir2), R_m1_22.real)
            cuda.atomic.add(rho_m1.imag, (iz2, ir2), R_m1_22.imag)
            cuda.atomic.add(rho_m1.real, (iz2, ir3), R_m1_32.real)
            cuda.atomic.add(rho_m1.imag, (iz2, ir3), R_m1_32.imag)
            cuda.atomic.add(rho_m1.real, (iz3, ir0), R_m1_03.real)
            cuda.atomic.add(rho_m1.imag, (iz3, ir0), R_m1_03.imag)
            cuda.atomic.add(rho_m1.real, (iz3, ir1), R_m1_13.real)
            cuda.atomic.add(rho_m1.imag, (iz3, ir1), R_m1_13.imag)
            cuda.atomic.add(rho_m1.real, (iz3, ir2), R_m1_23.real)
            cuda.atomic.add(rho_m1.imag, (iz3, ir2), R_m1_23.imag)
            cuda.atomic.add(rho_m1.real, (iz3, ir3), R_m1_33.real)
            cuda.atomic.add(rho_m1.imag, (iz3, ir3), R_m1_33.imag)

# -------------------------------
# Field deposition - cubic - J
# -------------------------------

@compile_cupy
def deposit_J_gpu_cubic(x, y, z, w, q,
                        ux, uy, uz, inv_gamma,
                        invdz, zmin, Nz,
                        invdr, rmin, Nr,
                        j_r_m0, j_r_m1,
                        j_t_m0, j_t_m1,
                        j_z_m0, j_z_m1,
                        cell_idx, prefix_sum,
                        beta_n_m0, beta_n_m1):
    """
    Deposition of the current J using numba on the GPU.
    Iterates over the cells and over the particles per cell.
    Calculates the weighted amount of J that is deposited to the
    16 cells surounding the particle based on its shape (cubic).

    The particles are sorted by their cell index (the lower cell
    in r and z that they deposit to) and the deposited field
    is split into 16 variables (one for each cell) to maintain
    parallelism while avoiding any race conditions.

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

    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    j_r_m0, j_r_m1, j_t_m0, j_t_m1, j_z_m0, j_z_m1,: 2darrays of complexs
        The current component in each direction (r, t, z)
        on the interpolation grid for mode 0 and 1.
        (is modified by this function)

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    cell_idx : 1darray of integers
        The cell index of the particle

    prefix_sum : 1darray of integers
        Represents the cumulative sum of
        the particles per cell

    beta_n_m0, beta_n_m1 : 1darrays of floats
        Ruyten-corrected particle shape factor coefficients for mode 0 and 1
    """
    # Get the 1D CUDA grid
    i = cuda.grid(1)
    # Deposit the field per cell in parallel (for threads < number of cells)
    if i < prefix_sum.shape[0]:
        # Retrieve index of upper grid point (in z and r) from prefix-sum index
        # (See calculation of prefix-sum index in `get_cell_idx_per_particle`)
        iz_upper = int( i / (Nr+1) )
        ir_upper = int( i - iz_upper * (Nr+1) )
        # Calculate the inclusive offset for the current cell
        # It represents the number of particles contained in all other cells
        # with an index smaller than i + the total number of particles in the
        # current cell (inclusive).
        incl_offset = np.int32(prefix_sum[i])
        # Calculate the frequency per cell from the offset and the previous
        # offset (prefix_sum[i-1]).
        if i > 0:
            frequency_per_cell = np.int32(incl_offset - prefix_sum[i-1])
        if i == 0:
            frequency_per_cell = np.int32(incl_offset)

        # Declare the local field value for
        # all possible deposition directions,
        # depending on the shape order and per mode for r,t and z.
        J_r_m0_00 = 0.
        J_t_m0_00 = 0.
        J_z_m0_00 = 0.
        J_r_m1_00 = 0. + 0.j
        J_t_m1_00 = 0. + 0.j
        J_z_m1_00 = 0. + 0.j

        J_r_m0_01 = 0.
        J_t_m0_01 = 0.
        J_z_m0_01 = 0.
        J_r_m1_01 = 0. + 0.j
        J_t_m1_01 = 0. + 0.j
        J_z_m1_01 = 0. + 0.j

        J_r_m0_02 = 0.
        J_t_m0_02 = 0.
        J_z_m0_02 = 0.
        J_r_m1_02 = 0. + 0.j
        J_t_m1_02 = 0. + 0.j
        J_z_m1_02 = 0. + 0.j

        J_r_m0_03 = 0.
        J_t_m0_03 = 0.
        J_z_m0_03 = 0.
        J_r_m1_03 = 0. + 0.j
        J_t_m1_03 = 0. + 0.j
        J_z_m1_03 = 0. + 0.j

        J_r_m0_10 = 0.
        J_t_m0_10 = 0.
        J_z_m0_10 = 0.
        J_r_m1_10 = 0. + 0.j
        J_t_m1_10 = 0. + 0.j
        J_z_m1_10 = 0. + 0.j

        J_r_m0_11 = 0.
        J_t_m0_11 = 0.
        J_z_m0_11 = 0.
        J_r_m1_11 = 0. + 0.j
        J_t_m1_11 = 0. + 0.j
        J_z_m1_11 = 0. + 0.j

        J_r_m0_12 = 0.
        J_t_m0_12 = 0.
        J_z_m0_12 = 0.
        J_r_m1_12 = 0. + 0.j
        J_t_m1_12 = 0. + 0.j
        J_z_m1_12 = 0. + 0.j

        J_r_m0_13 = 0.
        J_t_m0_13 = 0.
        J_z_m0_13 = 0.
        J_r_m1_13 = 0. + 0.j
        J_t_m1_13 = 0. + 0.j
        J_z_m1_13 = 0. + 0.j

        J_r_m0_20 = 0.
        J_t_m0_20 = 0.
        J_z_m0_20 = 0.
        J_r_m1_20 = 0. + 0.j
        J_t_m1_20 = 0. + 0.j
        J_z_m1_20 = 0. + 0.j

        J_r_m0_21 = 0.
        J_t_m0_21 = 0.
        J_z_m0_21 = 0.
        J_r_m1_21 = 0. + 0.j
        J_t_m1_21 = 0. + 0.j
        J_z_m1_21 = 0. + 0.j

        J_r_m0_22 = 0.
        J_t_m0_22 = 0.
        J_z_m0_22 = 0.
        J_r_m1_22 = 0. + 0.j
        J_t_m1_22 = 0. + 0.j
        J_z_m1_22 = 0. + 0.j

        J_r_m0_23 = 0.
        J_t_m0_23 = 0.
        J_z_m0_23 = 0.
        J_r_m1_23 = 0. + 0.j
        J_t_m1_23 = 0. + 0.j
        J_z_m1_23 = 0. + 0.j

        J_r_m0_30 = 0.
        J_t_m0_30 = 0.
        J_z_m0_30 = 0.
        J_r_m1_30 = 0. + 0.j
        J_t_m1_30 = 0. + 0.j
        J_z_m1_30 = 0. + 0.j

        J_r_m0_31 = 0.
        J_t_m0_31 = 0.
        J_z_m0_31 = 0.
        J_r_m1_31 = 0. + 0.j
        J_t_m1_31 = 0. + 0.j
        J_z_m1_31 = 0. + 0.j

        J_r_m0_32 = 0.
        J_t_m0_32 = 0.
        J_z_m0_32 = 0.
        J_r_m1_32 = 0. + 0.j
        J_t_m1_32 = 0. + 0.j
        J_z_m1_32 = 0. + 0.j

        J_r_m0_33 = 0.
        J_t_m0_33 = 0.
        J_z_m0_33 = 0.
        J_r_m1_33 = 0. + 0.j
        J_t_m1_33 = 0. + 0.j
        J_z_m1_33 = 0. + 0.j

        # Loop over the number of particles per cell
        for j in range(frequency_per_cell):
            # Get the particle index
            # ----------------------
            # (Since incl_offset is a cumulative sum of particle number,
            # and since python index starts at 0, one has to add -1)
            ptcl_idx = incl_offset-1-j

            # Preliminary arrays for the cylindrical conversion
            # --------------------------------------------
            # Position
            xj = x[ptcl_idx]
            yj = y[ptcl_idx]
            zj = z[ptcl_idx]
            # Velocity
            uxj = ux[ptcl_idx]
            uyj = uy[ptcl_idx]
            uzj = uz[ptcl_idx]
            # Inverse gamma
            inv_gammaj = inv_gamma[ptcl_idx]
            # Weights
            wj = q * w[ptcl_idx]

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
            exptheta_m0 = 1.
            exptheta_m1 = cos + 1.j*sin

            # Get weights for the deposition
            # --------------------------------------------
            # Positions of the particles, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5

            # Ruyten-corrected shape factor coefficients for both modes
            ir = min( int(math.ceil(r_cell)), Nr )
            bn_m0 = beta_n_m0[ir]
            bn_m1 = beta_n_m1[ir]

            # Calculate the currents
            # --------------------------------------------
            # Mode 0
            J_r_m0_scal = wj * c * inv_gammaj*(cos*uxj + sin*uyj) * exptheta_m0
            J_t_m0_scal = wj * c * inv_gammaj*(cos*uyj - sin*uxj) * exptheta_m0
            J_z_m0_scal = wj * c * inv_gammaj*uzj * exptheta_m0
            # Mode 1
            J_r_m1_scal = wj * c * inv_gammaj*(cos*uxj + sin*uyj) * exptheta_m1
            J_t_m1_scal = wj * c * inv_gammaj*(cos*uyj - sin*uxj) * exptheta_m1
            J_z_m1_scal = wj * c * inv_gammaj*uzj * exptheta_m1

            J_r_m0_00 += Sr_cubic(r_cell, 0,-1, bn_m0)*Sz_cubic(z_cell, 0)*J_r_m0_scal
            J_r_m1_00 += Sr_cubic(r_cell, 0, 1, bn_m1)*Sz_cubic(z_cell, 0)*J_r_m1_scal
            J_r_m0_01 += Sr_cubic(r_cell, 0,-1, bn_m0)*Sz_cubic(z_cell, 1)*J_r_m0_scal
            J_r_m1_01 += Sr_cubic(r_cell, 0, 1, bn_m1)*Sz_cubic(z_cell, 1)*J_r_m1_scal
            J_r_m0_02 += Sr_cubic(r_cell, 0,-1, bn_m0)*Sz_cubic(z_cell, 2)*J_r_m0_scal
            J_r_m1_02 += Sr_cubic(r_cell, 0, 1, bn_m1)*Sz_cubic(z_cell, 2)*J_r_m1_scal
            J_r_m0_03 += Sr_cubic(r_cell, 0,-1, bn_m0)*Sz_cubic(z_cell, 3)*J_r_m0_scal
            J_r_m1_03 += Sr_cubic(r_cell, 0, 1, bn_m1)*Sz_cubic(z_cell, 3)*J_r_m1_scal

            J_r_m0_10 += Sr_cubic(r_cell, 1,-1, bn_m0)*Sz_cubic(z_cell, 0)*J_r_m0_scal
            J_r_m1_10 += Sr_cubic(r_cell, 1, 1, bn_m1)*Sz_cubic(z_cell, 0)*J_r_m1_scal
            J_r_m0_11 += Sr_cubic(r_cell, 1,-1, bn_m0)*Sz_cubic(z_cell, 1)*J_r_m0_scal
            J_r_m1_11 += Sr_cubic(r_cell, 1, 1, bn_m1)*Sz_cubic(z_cell, 1)*J_r_m1_scal
            J_r_m0_12 += Sr_cubic(r_cell, 1,-1, bn_m0)*Sz_cubic(z_cell, 2)*J_r_m0_scal
            J_r_m1_12 += Sr_cubic(r_cell, 1, 1, bn_m1)*Sz_cubic(z_cell, 2)*J_r_m1_scal
            J_r_m0_13 += Sr_cubic(r_cell, 1,-1, bn_m0)*Sz_cubic(z_cell, 3)*J_r_m0_scal
            J_r_m1_13 += Sr_cubic(r_cell, 1, 1, bn_m1)*Sz_cubic(z_cell, 3)*J_r_m1_scal

            J_r_m0_20 += Sr_cubic(r_cell, 2,-1, bn_m0)*Sz_cubic(z_cell, 0)*J_r_m0_scal
            J_r_m1_20 += Sr_cubic(r_cell, 2, 1, bn_m1)*Sz_cubic(z_cell, 0)*J_r_m1_scal
            J_r_m0_21 += Sr_cubic(r_cell, 2,-1, bn_m0)*Sz_cubic(z_cell, 1)*J_r_m0_scal
            J_r_m1_21 += Sr_cubic(r_cell, 2, 1, bn_m1)*Sz_cubic(z_cell, 1)*J_r_m1_scal
            J_r_m0_22 += Sr_cubic(r_cell, 2,-1, bn_m0)*Sz_cubic(z_cell, 2)*J_r_m0_scal
            J_r_m1_22 += Sr_cubic(r_cell, 2, 1, bn_m1)*Sz_cubic(z_cell, 2)*J_r_m1_scal
            J_r_m0_23 += Sr_cubic(r_cell, 2,-1, bn_m0)*Sz_cubic(z_cell, 3)*J_r_m0_scal
            J_r_m1_23 += Sr_cubic(r_cell, 2, 1, bn_m1)*Sz_cubic(z_cell, 3)*J_r_m1_scal

            J_r_m0_30 += Sr_cubic(r_cell, 3,-1, bn_m0)*Sz_cubic(z_cell, 0)*J_r_m0_scal
            J_r_m1_30 += Sr_cubic(r_cell, 3, 1, bn_m1)*Sz_cubic(z_cell, 0)*J_r_m1_scal
            J_r_m0_31 += Sr_cubic(r_cell, 3,-1, bn_m0)*Sz_cubic(z_cell, 1)*J_r_m0_scal
            J_r_m1_31 += Sr_cubic(r_cell, 3, 1, bn_m1)*Sz_cubic(z_cell, 1)*J_r_m1_scal
            J_r_m0_32 += Sr_cubic(r_cell, 3,-1, bn_m0)*Sz_cubic(z_cell, 2)*J_r_m0_scal
            J_r_m1_32 += Sr_cubic(r_cell, 3, 1, bn_m1)*Sz_cubic(z_cell, 2)*J_r_m1_scal
            J_r_m0_33 += Sr_cubic(r_cell, 3,-1, bn_m0)*Sz_cubic(z_cell, 3)*J_r_m0_scal
            J_r_m1_33 += Sr_cubic(r_cell, 3, 1, bn_m1)*Sz_cubic(z_cell, 3)*J_r_m1_scal

            J_t_m0_00 += Sr_cubic(r_cell, 0,-1, bn_m0)*Sz_cubic(z_cell, 0)*J_t_m0_scal
            J_t_m1_00 += Sr_cubic(r_cell, 0, 1, bn_m1)*Sz_cubic(z_cell, 0)*J_t_m1_scal
            J_t_m0_01 += Sr_cubic(r_cell, 0,-1, bn_m0)*Sz_cubic(z_cell, 1)*J_t_m0_scal
            J_t_m1_01 += Sr_cubic(r_cell, 0, 1, bn_m1)*Sz_cubic(z_cell, 1)*J_t_m1_scal
            J_t_m0_02 += Sr_cubic(r_cell, 0,-1, bn_m0)*Sz_cubic(z_cell, 2)*J_t_m0_scal
            J_t_m1_02 += Sr_cubic(r_cell, 0, 1, bn_m1)*Sz_cubic(z_cell, 2)*J_t_m1_scal
            J_t_m0_03 += Sr_cubic(r_cell, 0,-1, bn_m0)*Sz_cubic(z_cell, 3)*J_t_m0_scal
            J_t_m1_03 += Sr_cubic(r_cell, 0, 1, bn_m1)*Sz_cubic(z_cell, 3)*J_t_m1_scal

            J_t_m0_10 += Sr_cubic(r_cell, 1,-1, bn_m0)*Sz_cubic(z_cell, 0)*J_t_m0_scal
            J_t_m1_10 += Sr_cubic(r_cell, 1, 1, bn_m1)*Sz_cubic(z_cell, 0)*J_t_m1_scal
            J_t_m0_11 += Sr_cubic(r_cell, 1,-1, bn_m0)*Sz_cubic(z_cell, 1)*J_t_m0_scal
            J_t_m1_11 += Sr_cubic(r_cell, 1, 1, bn_m1)*Sz_cubic(z_cell, 1)*J_t_m1_scal
            J_t_m0_12 += Sr_cubic(r_cell, 1,-1, bn_m0)*Sz_cubic(z_cell, 2)*J_t_m0_scal
            J_t_m1_12 += Sr_cubic(r_cell, 1, 1, bn_m1)*Sz_cubic(z_cell, 2)*J_t_m1_scal
            J_t_m0_13 += Sr_cubic(r_cell, 1,-1, bn_m0)*Sz_cubic(z_cell, 3)*J_t_m0_scal
            J_t_m1_13 += Sr_cubic(r_cell, 1, 1, bn_m1)*Sz_cubic(z_cell, 3)*J_t_m1_scal

            J_t_m0_20 += Sr_cubic(r_cell, 2,-1, bn_m0)*Sz_cubic(z_cell, 0)*J_t_m0_scal
            J_t_m1_20 += Sr_cubic(r_cell, 2, 1, bn_m1)*Sz_cubic(z_cell, 0)*J_t_m1_scal
            J_t_m0_21 += Sr_cubic(r_cell, 2,-1, bn_m0)*Sz_cubic(z_cell, 1)*J_t_m0_scal
            J_t_m1_21 += Sr_cubic(r_cell, 2, 1, bn_m1)*Sz_cubic(z_cell, 1)*J_t_m1_scal
            J_t_m0_22 += Sr_cubic(r_cell, 2,-1, bn_m0)*Sz_cubic(z_cell, 2)*J_t_m0_scal
            J_t_m1_22 += Sr_cubic(r_cell, 2, 1, bn_m1)*Sz_cubic(z_cell, 2)*J_t_m1_scal
            J_t_m0_23 += Sr_cubic(r_cell, 2,-1, bn_m0)*Sz_cubic(z_cell, 3)*J_t_m0_scal
            J_t_m1_23 += Sr_cubic(r_cell, 2, 1, bn_m1)*Sz_cubic(z_cell, 3)*J_t_m1_scal

            J_t_m0_30 += Sr_cubic(r_cell, 3,-1, bn_m0)*Sz_cubic(z_cell, 0)*J_t_m0_scal
            J_t_m1_30 += Sr_cubic(r_cell, 3, 1, bn_m1)*Sz_cubic(z_cell, 0)*J_t_m1_scal
            J_t_m0_31 += Sr_cubic(r_cell, 3,-1, bn_m0)*Sz_cubic(z_cell, 1)*J_t_m0_scal
            J_t_m1_31 += Sr_cubic(r_cell, 3, 1, bn_m1)*Sz_cubic(z_cell, 1)*J_t_m1_scal
            J_t_m0_32 += Sr_cubic(r_cell, 3,-1, bn_m0)*Sz_cubic(z_cell, 2)*J_t_m0_scal
            J_t_m1_32 += Sr_cubic(r_cell, 3, 1, bn_m1)*Sz_cubic(z_cell, 2)*J_t_m1_scal
            J_t_m0_33 += Sr_cubic(r_cell, 3,-1, bn_m0)*Sz_cubic(z_cell, 3)*J_t_m0_scal
            J_t_m1_33 += Sr_cubic(r_cell, 3, 1, bn_m1)*Sz_cubic(z_cell, 3)*J_t_m1_scal

            J_z_m0_00 += Sr_cubic(r_cell, 0, 1, bn_m0)*Sz_cubic(z_cell, 0)*J_z_m0_scal
            J_z_m1_00 += Sr_cubic(r_cell, 0,-1, bn_m1)*Sz_cubic(z_cell, 0)*J_z_m1_scal
            J_z_m0_01 += Sr_cubic(r_cell, 0, 1, bn_m0)*Sz_cubic(z_cell, 1)*J_z_m0_scal
            J_z_m1_01 += Sr_cubic(r_cell, 0,-1, bn_m1)*Sz_cubic(z_cell, 1)*J_z_m1_scal
            J_z_m0_02 += Sr_cubic(r_cell, 0, 1, bn_m0)*Sz_cubic(z_cell, 2)*J_z_m0_scal
            J_z_m1_02 += Sr_cubic(r_cell, 0,-1, bn_m1)*Sz_cubic(z_cell, 2)*J_z_m1_scal
            J_z_m0_03 += Sr_cubic(r_cell, 0, 1, bn_m0)*Sz_cubic(z_cell, 3)*J_z_m0_scal
            J_z_m1_03 += Sr_cubic(r_cell, 0,-1, bn_m1)*Sz_cubic(z_cell, 3)*J_z_m1_scal

            J_z_m0_10 += Sr_cubic(r_cell, 1, 1, bn_m0)*Sz_cubic(z_cell, 0)*J_z_m0_scal
            J_z_m1_10 += Sr_cubic(r_cell, 1,-1, bn_m1)*Sz_cubic(z_cell, 0)*J_z_m1_scal
            J_z_m0_11 += Sr_cubic(r_cell, 1, 1, bn_m0)*Sz_cubic(z_cell, 1)*J_z_m0_scal
            J_z_m1_11 += Sr_cubic(r_cell, 1,-1, bn_m1)*Sz_cubic(z_cell, 1)*J_z_m1_scal
            J_z_m0_12 += Sr_cubic(r_cell, 1, 1, bn_m0)*Sz_cubic(z_cell, 2)*J_z_m0_scal
            J_z_m1_12 += Sr_cubic(r_cell, 1,-1, bn_m1)*Sz_cubic(z_cell, 2)*J_z_m1_scal
            J_z_m0_13 += Sr_cubic(r_cell, 1, 1, bn_m0)*Sz_cubic(z_cell, 3)*J_z_m0_scal
            J_z_m1_13 += Sr_cubic(r_cell, 1,-1, bn_m1)*Sz_cubic(z_cell, 3)*J_z_m1_scal

            J_z_m0_20 += Sr_cubic(r_cell, 2, 1, bn_m0)*Sz_cubic(z_cell, 0)*J_z_m0_scal
            J_z_m1_20 += Sr_cubic(r_cell, 2,-1, bn_m1)*Sz_cubic(z_cell, 0)*J_z_m1_scal
            J_z_m0_21 += Sr_cubic(r_cell, 2, 1, bn_m0)*Sz_cubic(z_cell, 1)*J_z_m0_scal
            J_z_m1_21 += Sr_cubic(r_cell, 2,-1, bn_m1)*Sz_cubic(z_cell, 1)*J_z_m1_scal
            J_z_m0_22 += Sr_cubic(r_cell, 2, 1, bn_m0)*Sz_cubic(z_cell, 2)*J_z_m0_scal
            J_z_m1_22 += Sr_cubic(r_cell, 2,-1, bn_m1)*Sz_cubic(z_cell, 2)*J_z_m1_scal
            J_z_m0_23 += Sr_cubic(r_cell, 2, 1, bn_m0)*Sz_cubic(z_cell, 3)*J_z_m0_scal
            J_z_m1_23 += Sr_cubic(r_cell, 2,-1, bn_m1)*Sz_cubic(z_cell, 3)*J_z_m1_scal

            J_z_m0_30 += Sr_cubic(r_cell, 3, 1, bn_m0)*Sz_cubic(z_cell, 0)*J_z_m0_scal
            J_z_m1_30 += Sr_cubic(r_cell, 3,-1, bn_m1)*Sz_cubic(z_cell, 0)*J_z_m1_scal
            J_z_m0_31 += Sr_cubic(r_cell, 3, 1, bn_m0)*Sz_cubic(z_cell, 1)*J_z_m0_scal
            J_z_m1_31 += Sr_cubic(r_cell, 3,-1, bn_m1)*Sz_cubic(z_cell, 1)*J_z_m1_scal
            J_z_m0_32 += Sr_cubic(r_cell, 3, 1, bn_m0)*Sz_cubic(z_cell, 2)*J_z_m0_scal
            J_z_m1_32 += Sr_cubic(r_cell, 3,-1, bn_m1)*Sz_cubic(z_cell, 2)*J_z_m1_scal
            J_z_m0_33 += Sr_cubic(r_cell, 3, 1, bn_m0)*Sz_cubic(z_cell, 3)*J_z_m0_scal
            J_z_m1_33 += Sr_cubic(r_cell, 3,-1, bn_m1)*Sz_cubic(z_cell, 3)*J_z_m1_scal

        # Calculate longitudinal indices at which to add charge
        iz0 = iz_upper - 2
        iz1 = iz_upper - 1
        iz2 = iz_upper
        iz3 = iz_upper + 1
        if iz0 < 0:
            iz0 += Nz
        if iz1 < 0:
            iz1 += Nz
        if iz3 > Nz-1:
            iz3 -= Nz
        # Calculate radial indices at which to add charge
        ir0 = ir_upper - 2
        ir1 = min( ir_upper - 1, Nr-1 )
        ir2 = min( ir_upper    , Nr-1 )
        ir3 = min( ir_upper + 1, Nr-1 )
        if ir0 < 0:
            # Deposition below the axis: fold index into physical region
            ir0 = -(1 + ir0)
        if ir1 < 0:
            # Deposition below the axis: fold index into physical region
            ir1 = -(1 + ir1)

        # Atomically add the registers to global memory
        if frequency_per_cell > 0:
            # jr: Mode 0
            cuda.atomic.add(j_r_m0.real, (iz0, ir0), J_r_m0_00.real)
            cuda.atomic.add(j_r_m0.real, (iz0, ir1), J_r_m0_10.real)
            cuda.atomic.add(j_r_m0.real, (iz0, ir2), J_r_m0_20.real)
            cuda.atomic.add(j_r_m0.real, (iz0, ir3), J_r_m0_30.real)
            cuda.atomic.add(j_r_m0.real, (iz1, ir0), J_r_m0_01.real)
            cuda.atomic.add(j_r_m0.real, (iz1, ir1), J_r_m0_11.real)
            cuda.atomic.add(j_r_m0.real, (iz1, ir2), J_r_m0_21.real)
            cuda.atomic.add(j_r_m0.real, (iz1, ir3), J_r_m0_31.real)
            cuda.atomic.add(j_r_m0.real, (iz2, ir0), J_r_m0_02.real)
            cuda.atomic.add(j_r_m0.real, (iz2, ir1), J_r_m0_12.real)
            cuda.atomic.add(j_r_m0.real, (iz2, ir2), J_r_m0_22.real)
            cuda.atomic.add(j_r_m0.real, (iz2, ir3), J_r_m0_32.real)
            cuda.atomic.add(j_r_m0.real, (iz3, ir0), J_r_m0_03.real)
            cuda.atomic.add(j_r_m0.real, (iz3, ir1), J_r_m0_13.real)
            cuda.atomic.add(j_r_m0.real, (iz3, ir2), J_r_m0_23.real)
            cuda.atomic.add(j_r_m0.real, (iz3, ir3), J_r_m0_33.real)
            # jr: Mode 1
            cuda.atomic.add(j_r_m1.real, (iz0, ir0), J_r_m1_00.real)
            cuda.atomic.add(j_r_m1.imag, (iz0, ir0), J_r_m1_00.imag)
            cuda.atomic.add(j_r_m1.real, (iz0, ir1), J_r_m1_10.real)
            cuda.atomic.add(j_r_m1.imag, (iz0, ir1), J_r_m1_10.imag)
            cuda.atomic.add(j_r_m1.real, (iz0, ir2), J_r_m1_20.real)
            cuda.atomic.add(j_r_m1.imag, (iz0, ir2), J_r_m1_20.imag)
            cuda.atomic.add(j_r_m1.real, (iz0, ir3), J_r_m1_30.real)
            cuda.atomic.add(j_r_m1.imag, (iz0, ir3), J_r_m1_30.imag)
            cuda.atomic.add(j_r_m1.real, (iz1, ir0), J_r_m1_01.real)
            cuda.atomic.add(j_r_m1.imag, (iz1, ir0), J_r_m1_01.imag)
            cuda.atomic.add(j_r_m1.real, (iz1, ir1), J_r_m1_11.real)
            cuda.atomic.add(j_r_m1.imag, (iz1, ir1), J_r_m1_11.imag)
            cuda.atomic.add(j_r_m1.real, (iz1, ir2), J_r_m1_21.real)
            cuda.atomic.add(j_r_m1.imag, (iz1, ir2), J_r_m1_21.imag)
            cuda.atomic.add(j_r_m1.real, (iz1, ir3), J_r_m1_31.real)
            cuda.atomic.add(j_r_m1.imag, (iz1, ir3), J_r_m1_31.imag)
            cuda.atomic.add(j_r_m1.real, (iz2, ir0), J_r_m1_02.real)
            cuda.atomic.add(j_r_m1.imag, (iz2, ir0), J_r_m1_02.imag)
            cuda.atomic.add(j_r_m1.real, (iz2, ir1), J_r_m1_12.real)
            cuda.atomic.add(j_r_m1.imag, (iz2, ir1), J_r_m1_12.imag)
            cuda.atomic.add(j_r_m1.real, (iz2, ir2), J_r_m1_22.real)
            cuda.atomic.add(j_r_m1.imag, (iz2, ir2), J_r_m1_22.imag)
            cuda.atomic.add(j_r_m1.real, (iz2, ir3), J_r_m1_32.real)
            cuda.atomic.add(j_r_m1.imag, (iz2, ir3), J_r_m1_32.imag)
            cuda.atomic.add(j_r_m1.real, (iz3, ir0), J_r_m1_03.real)
            cuda.atomic.add(j_r_m1.imag, (iz3, ir0), J_r_m1_03.imag)
            cuda.atomic.add(j_r_m1.real, (iz3, ir1), J_r_m1_13.real)
            cuda.atomic.add(j_r_m1.imag, (iz3, ir1), J_r_m1_13.imag)
            cuda.atomic.add(j_r_m1.real, (iz3, ir2), J_r_m1_23.real)
            cuda.atomic.add(j_r_m1.imag, (iz3, ir2), J_r_m1_23.imag)
            cuda.atomic.add(j_r_m1.real, (iz3, ir3), J_r_m1_33.real)
            cuda.atomic.add(j_r_m1.imag, (iz3, ir3), J_r_m1_33.imag)
            # jt: Mode 0
            cuda.atomic.add(j_t_m0.real, (iz0, ir0), J_t_m0_00.real)
            cuda.atomic.add(j_t_m0.real, (iz0, ir1), J_t_m0_10.real)
            cuda.atomic.add(j_t_m0.real, (iz0, ir2), J_t_m0_20.real)
            cuda.atomic.add(j_t_m0.real, (iz0, ir3), J_t_m0_30.real)
            cuda.atomic.add(j_t_m0.real, (iz1, ir0), J_t_m0_01.real)
            cuda.atomic.add(j_t_m0.real, (iz1, ir1), J_t_m0_11.real)
            cuda.atomic.add(j_t_m0.real, (iz1, ir2), J_t_m0_21.real)
            cuda.atomic.add(j_t_m0.real, (iz1, ir3), J_t_m0_31.real)
            cuda.atomic.add(j_t_m0.real, (iz2, ir0), J_t_m0_02.real)
            cuda.atomic.add(j_t_m0.real, (iz2, ir1), J_t_m0_12.real)
            cuda.atomic.add(j_t_m0.real, (iz2, ir2), J_t_m0_22.real)
            cuda.atomic.add(j_t_m0.real, (iz2, ir3), J_t_m0_32.real)
            cuda.atomic.add(j_t_m0.real, (iz3, ir0), J_t_m0_03.real)
            cuda.atomic.add(j_t_m0.real, (iz3, ir1), J_t_m0_13.real)
            cuda.atomic.add(j_t_m0.real, (iz3, ir2), J_t_m0_23.real)
            cuda.atomic.add(j_t_m0.real, (iz3, ir3), J_t_m0_33.real)
            # jt: Mode 1
            cuda.atomic.add(j_t_m1.real, (iz0, ir0), J_t_m1_00.real)
            cuda.atomic.add(j_t_m1.imag, (iz0, ir0), J_t_m1_00.imag)
            cuda.atomic.add(j_t_m1.real, (iz0, ir1), J_t_m1_10.real)
            cuda.atomic.add(j_t_m1.imag, (iz0, ir1), J_t_m1_10.imag)
            cuda.atomic.add(j_t_m1.real, (iz0, ir2), J_t_m1_20.real)
            cuda.atomic.add(j_t_m1.imag, (iz0, ir2), J_t_m1_20.imag)
            cuda.atomic.add(j_t_m1.real, (iz0, ir3), J_t_m1_30.real)
            cuda.atomic.add(j_t_m1.imag, (iz0, ir3), J_t_m1_30.imag)
            cuda.atomic.add(j_t_m1.real, (iz1, ir0), J_t_m1_01.real)
            cuda.atomic.add(j_t_m1.imag, (iz1, ir0), J_t_m1_01.imag)
            cuda.atomic.add(j_t_m1.real, (iz1, ir1), J_t_m1_11.real)
            cuda.atomic.add(j_t_m1.imag, (iz1, ir1), J_t_m1_11.imag)
            cuda.atomic.add(j_t_m1.real, (iz1, ir2), J_t_m1_21.real)
            cuda.atomic.add(j_t_m1.imag, (iz1, ir2), J_t_m1_21.imag)
            cuda.atomic.add(j_t_m1.real, (iz1, ir3), J_t_m1_31.real)
            cuda.atomic.add(j_t_m1.imag, (iz1, ir3), J_t_m1_31.imag)
            cuda.atomic.add(j_t_m1.real, (iz2, ir0), J_t_m1_02.real)
            cuda.atomic.add(j_t_m1.imag, (iz2, ir0), J_t_m1_02.imag)
            cuda.atomic.add(j_t_m1.real, (iz2, ir1), J_t_m1_12.real)
            cuda.atomic.add(j_t_m1.imag, (iz2, ir1), J_t_m1_12.imag)
            cuda.atomic.add(j_t_m1.real, (iz2, ir2), J_t_m1_22.real)
            cuda.atomic.add(j_t_m1.imag, (iz2, ir2), J_t_m1_22.imag)
            cuda.atomic.add(j_t_m1.real, (iz2, ir3), J_t_m1_32.real)
            cuda.atomic.add(j_t_m1.imag, (iz2, ir3), J_t_m1_32.imag)
            cuda.atomic.add(j_t_m1.real, (iz3, ir0), J_t_m1_03.real)
            cuda.atomic.add(j_t_m1.imag, (iz3, ir0), J_t_m1_03.imag)
            cuda.atomic.add(j_t_m1.real, (iz3, ir1), J_t_m1_13.real)
            cuda.atomic.add(j_t_m1.imag, (iz3, ir1), J_t_m1_13.imag)
            cuda.atomic.add(j_t_m1.real, (iz3, ir2), J_t_m1_23.real)
            cuda.atomic.add(j_t_m1.imag, (iz3, ir2), J_t_m1_23.imag)
            cuda.atomic.add(j_t_m1.real, (iz3, ir3), J_t_m1_33.real)
            cuda.atomic.add(j_t_m1.imag, (iz3, ir3), J_t_m1_33.imag)
            # jz: Mode 0
            cuda.atomic.add(j_z_m0.real, (iz0, ir0), J_z_m0_00.real)
            cuda.atomic.add(j_z_m0.real, (iz0, ir1), J_z_m0_10.real)
            cuda.atomic.add(j_z_m0.real, (iz0, ir2), J_z_m0_20.real)
            cuda.atomic.add(j_z_m0.real, (iz0, ir3), J_z_m0_30.real)
            cuda.atomic.add(j_z_m0.real, (iz1, ir0), J_z_m0_01.real)
            cuda.atomic.add(j_z_m0.real, (iz1, ir1), J_z_m0_11.real)
            cuda.atomic.add(j_z_m0.real, (iz1, ir2), J_z_m0_21.real)
            cuda.atomic.add(j_z_m0.real, (iz1, ir3), J_z_m0_31.real)
            cuda.atomic.add(j_z_m0.real, (iz2, ir0), J_z_m0_02.real)
            cuda.atomic.add(j_z_m0.real, (iz2, ir1), J_z_m0_12.real)
            cuda.atomic.add(j_z_m0.real, (iz2, ir2), J_z_m0_22.real)
            cuda.atomic.add(j_z_m0.real, (iz2, ir3), J_z_m0_32.real)
            cuda.atomic.add(j_z_m0.real, (iz3, ir0), J_z_m0_03.real)
            cuda.atomic.add(j_z_m0.real, (iz3, ir1), J_z_m0_13.real)
            cuda.atomic.add(j_z_m0.real, (iz3, ir2), J_z_m0_23.real)
            cuda.atomic.add(j_z_m0.real, (iz3, ir3), J_z_m0_33.real)
            # jz: Mode 1
            cuda.atomic.add(j_z_m1.real, (iz0, ir0), J_z_m1_00.real)
            cuda.atomic.add(j_z_m1.imag, (iz0, ir0), J_z_m1_00.imag)
            cuda.atomic.add(j_z_m1.real, (iz0, ir1), J_z_m1_10.real)
            cuda.atomic.add(j_z_m1.imag, (iz0, ir1), J_z_m1_10.imag)
            cuda.atomic.add(j_z_m1.real, (iz0, ir2), J_z_m1_20.real)
            cuda.atomic.add(j_z_m1.imag, (iz0, ir2), J_z_m1_20.imag)
            cuda.atomic.add(j_z_m1.real, (iz0, ir3), J_z_m1_30.real)
            cuda.atomic.add(j_z_m1.imag, (iz0, ir3), J_z_m1_30.imag)
            cuda.atomic.add(j_z_m1.real, (iz1, ir0), J_z_m1_01.real)
            cuda.atomic.add(j_z_m1.imag, (iz1, ir0), J_z_m1_01.imag)
            cuda.atomic.add(j_z_m1.real, (iz1, ir1), J_z_m1_11.real)
            cuda.atomic.add(j_z_m1.imag, (iz1, ir1), J_z_m1_11.imag)
            cuda.atomic.add(j_z_m1.real, (iz1, ir2), J_z_m1_21.real)
            cuda.atomic.add(j_z_m1.imag, (iz1, ir2), J_z_m1_21.imag)
            cuda.atomic.add(j_z_m1.real, (iz1, ir3), J_z_m1_31.real)
            cuda.atomic.add(j_z_m1.imag, (iz1, ir3), J_z_m1_31.imag)
            cuda.atomic.add(j_z_m1.real, (iz2, ir0), J_z_m1_02.real)
            cuda.atomic.add(j_z_m1.imag, (iz2, ir0), J_z_m1_02.imag)
            cuda.atomic.add(j_z_m1.real, (iz2, ir1), J_z_m1_12.real)
            cuda.atomic.add(j_z_m1.imag, (iz2, ir1), J_z_m1_12.imag)
            cuda.atomic.add(j_z_m1.real, (iz2, ir2), J_z_m1_22.real)
            cuda.atomic.add(j_z_m1.imag, (iz2, ir2), J_z_m1_22.imag)
            cuda.atomic.add(j_z_m1.real, (iz2, ir3), J_z_m1_32.real)
            cuda.atomic.add(j_z_m1.imag, (iz2, ir3), J_z_m1_32.imag)
            cuda.atomic.add(j_z_m1.real, (iz3, ir0), J_z_m1_03.real)
            cuda.atomic.add(j_z_m1.imag, (iz3, ir0), J_z_m1_03.imag)
            cuda.atomic.add(j_z_m1.real, (iz3, ir1), J_z_m1_13.real)
            cuda.atomic.add(j_z_m1.imag, (iz3, ir1), J_z_m1_13.imag)
            cuda.atomic.add(j_z_m1.real, (iz3, ir2), J_z_m1_23.real)
            cuda.atomic.add(j_z_m1.imag, (iz3, ir2), J_z_m1_23.imag)
            cuda.atomic.add(j_z_m1.real, (iz3, ir3), J_z_m1_33.real)
            cuda.atomic.add(j_z_m1.imag, (iz3, ir3), J_z_m1_33.imag)


@compile_cupy
def deposit_J_gpu_cubic_m3_supercell(x, y, z, w, q,
                        ux, uy, uz, inv_gamma,
                        invdz, zmin, Nz,
                        invdr, rmin, Nr,
                        j_r_m0, j_t_m0, j_z_m0,
                        j_r_m1, j_t_m1, j_z_m1,
                        j_r_m2, j_t_m2, j_z_m2,
                        cell_idx, prefix_sum,
                        beta_n_m0, beta_n_m1, beta_n_m2):
    """
    Experimental fused cubic J deposition for Nm=3.

    This kernel iterates over sorted particle cells (`prefix_sum`) and
    accumulates contributions for m=0,1,2 in local per-cell buffers before
    atomically flushing to global memory.
    """
    i = cuda.grid(1)

    if i < prefix_sum.shape[0]:
        # Retrieve index of upper grid point from flattened cell index.
        iz_upper = int(i / (Nr + 1))
        ir_upper = int(i - iz_upper * (Nr + 1))

        incl_offset = np.int32(prefix_sum[i])
        if i > 0:
            frequency_per_cell = np.int32(incl_offset - prefix_sum[i - 1])
        else:
            frequency_per_cell = np.int32(incl_offset)

        if frequency_per_cell <= 0:
            return

        # 16-point cubic stencil accumulators (4 z x 4 r)
        jr0_acc = cuda.local.array(16, dtype=np.float64)
        jt0_acc = cuda.local.array(16, dtype=np.float64)
        jz0_acc = cuda.local.array(16, dtype=np.float64)

        jr1r_acc = cuda.local.array(16, dtype=np.float64)
        jr1i_acc = cuda.local.array(16, dtype=np.float64)
        jt1r_acc = cuda.local.array(16, dtype=np.float64)
        jt1i_acc = cuda.local.array(16, dtype=np.float64)
        jz1r_acc = cuda.local.array(16, dtype=np.float64)
        jz1i_acc = cuda.local.array(16, dtype=np.float64)

        jr2r_acc = cuda.local.array(16, dtype=np.float64)
        jr2i_acc = cuda.local.array(16, dtype=np.float64)
        jt2r_acc = cuda.local.array(16, dtype=np.float64)
        jt2i_acc = cuda.local.array(16, dtype=np.float64)
        jz2r_acc = cuda.local.array(16, dtype=np.float64)
        jz2i_acc = cuda.local.array(16, dtype=np.float64)

        for k in range(16):
            jr0_acc[k] = 0.
            jt0_acc[k] = 0.
            jz0_acc[k] = 0.

            jr1r_acc[k] = 0.
            jr1i_acc[k] = 0.
            jt1r_acc[k] = 0.
            jt1i_acc[k] = 0.
            jz1r_acc[k] = 0.
            jz1i_acc[k] = 0.

            jr2r_acc[k] = 0.
            jr2i_acc[k] = 0.
            jt2r_acc[k] = 0.
            jt2i_acc[k] = 0.
            jz2r_acc[k] = 0.
            jz2i_acc[k] = 0.

        for j in range(frequency_per_cell):
            ptcl_idx = incl_offset - 1 - j

            xj = x[ptcl_idx]
            yj = y[ptcl_idx]
            zj = z[ptcl_idx]

            uxj = ux[ptcl_idx]
            uyj = uy[ptcl_idx]
            uzj = uz[ptcl_idx]
            inv_gammaj = inv_gamma[ptcl_idx]
            wj = q * w[ptcl_idx]

            rj = math.sqrt(xj**2 + yj**2)
            if rj != 0.:
                invr = 1. / rj
                cos = xj * invr
                sin = yj * invr
            else:
                cos = 1.
                sin = 0.

            cos2 = cos * cos - sin * sin
            sin2 = 2. * cos * sin

            r_cell = invdr * (rj - rmin) - 0.5
            z_cell = invdz * (zj - zmin) - 0.5

            ir = min(int(math.ceil(r_cell)), Nr)
            bn0 = beta_n_m0[ir]
            bn1 = beta_n_m1[ir]
            bn2 = beta_n_m2[ir]

            Sz = (
                Sz_cubic(z_cell, 0), Sz_cubic(z_cell, 1),
                Sz_cubic(z_cell, 2), Sz_cubic(z_cell, 3)
            )

            Sr_rt0 = (
                Sr_cubic(r_cell, 0, -1, bn0), Sr_cubic(r_cell, 1, -1, bn0),
                Sr_cubic(r_cell, 2, -1, bn0), Sr_cubic(r_cell, 3, -1, bn0)
            )
            Sr_rt1 = (
                Sr_cubic(r_cell, 0,  1, bn1), Sr_cubic(r_cell, 1,  1, bn1),
                Sr_cubic(r_cell, 2,  1, bn1), Sr_cubic(r_cell, 3,  1, bn1)
            )
            Sr_rt2 = (
                Sr_cubic(r_cell, 0, -1, bn2), Sr_cubic(r_cell, 1, -1, bn2),
                Sr_cubic(r_cell, 2, -1, bn2), Sr_cubic(r_cell, 3, -1, bn2)
            )

            Sr_z0 = (
                Sr_cubic(r_cell, 0,  1, bn0), Sr_cubic(r_cell, 1,  1, bn0),
                Sr_cubic(r_cell, 2,  1, bn0), Sr_cubic(r_cell, 3,  1, bn0)
            )
            Sr_z1 = (
                Sr_cubic(r_cell, 0, -1, bn1), Sr_cubic(r_cell, 1, -1, bn1),
                Sr_cubic(r_cell, 2, -1, bn1), Sr_cubic(r_cell, 3, -1, bn1)
            )
            Sr_z2 = (
                Sr_cubic(r_cell, 0,  1, bn2), Sr_cubic(r_cell, 1,  1, bn2),
                Sr_cubic(r_cell, 2,  1, bn2), Sr_cubic(r_cell, 3,  1, bn2)
            )

            base = wj * c * inv_gammaj
            jr0 = base * (cos * uxj + sin * uyj)
            jt0 = base * (cos * uyj - sin * uxj)
            jz0 = base * uzj

            jr1_r = jr0 * cos
            jr1_i = jr0 * sin
            jt1_r = jt0 * cos
            jt1_i = jt0 * sin
            jz1_r = jz0 * cos
            jz1_i = jz0 * sin

            jr2_r = jr0 * cos2
            jr2_i = jr0 * sin2
            jt2_r = jt0 * cos2
            jt2_i = jt0 * sin2
            jz2_r = jz0 * cos2
            jz2_i = jz0 * sin2

            for ia in range(4):
                sz = Sz[ia]
                row = 4 * ia
                for ib in range(4):
                    idx = row + ib

                    w_rt0 = sz * Sr_rt0[ib]
                    w_rt1 = sz * Sr_rt1[ib]
                    w_rt2 = sz * Sr_rt2[ib]
                    w_z0 = sz * Sr_z0[ib]
                    w_z1 = sz * Sr_z1[ib]
                    w_z2 = sz * Sr_z2[ib]

                    jr0_acc[idx] += w_rt0 * jr0
                    jt0_acc[idx] += w_rt0 * jt0
                    jz0_acc[idx] += w_z0 * jz0

                    jr1r_acc[idx] += w_rt1 * jr1_r
                    jr1i_acc[idx] += w_rt1 * jr1_i
                    jt1r_acc[idx] += w_rt1 * jt1_r
                    jt1i_acc[idx] += w_rt1 * jt1_i
                    jz1r_acc[idx] += w_z1 * jz1_r
                    jz1i_acc[idx] += w_z1 * jz1_i

                    jr2r_acc[idx] += w_rt2 * jr2_r
                    jr2i_acc[idx] += w_rt2 * jr2_i
                    jt2r_acc[idx] += w_rt2 * jt2_r
                    jt2i_acc[idx] += w_rt2 * jt2_i
                    jz2r_acc[idx] += w_z2 * jz2_r
                    jz2i_acc[idx] += w_z2 * jz2_i

        # Output indices for this source cell.
        iz0 = iz_upper - 2
        iz1 = iz_upper - 1
        iz2 = iz_upper
        iz3 = iz_upper + 1
        if iz0 < 0:
            iz0 += Nz
        if iz1 < 0:
            iz1 += Nz
        if iz3 > Nz - 1:
            iz3 -= Nz

        ir0 = ir_upper - 2
        ir1 = min(ir_upper - 1, Nr - 1)
        ir2 = min(ir_upper, Nr - 1)
        ir3 = min(ir_upper + 1, Nr - 1)
        if ir0 < 0:
            ir0 = -(1 + ir0)
        if ir1 < 0:
            ir1 = -(1 + ir1)

        izs = (iz0, iz1, iz2, iz3)
        irs = (ir0, ir1, ir2, ir3)

        for ia in range(4):
            izp = izs[ia]
            row = 4 * ia
            for ib in range(4):
                irp = irs[ib]
                idx = row + ib

                cuda.atomic.add(j_r_m0.real, (izp, irp), jr0_acc[idx])
                cuda.atomic.add(j_t_m0.real, (izp, irp), jt0_acc[idx])
                cuda.atomic.add(j_z_m0.real, (izp, irp), jz0_acc[idx])

                cuda.atomic.add(j_r_m1.real, (izp, irp), jr1r_acc[idx])
                cuda.atomic.add(j_r_m1.imag, (izp, irp), jr1i_acc[idx])
                cuda.atomic.add(j_t_m1.real, (izp, irp), jt1r_acc[idx])
                cuda.atomic.add(j_t_m1.imag, (izp, irp), jt1i_acc[idx])
                cuda.atomic.add(j_z_m1.real, (izp, irp), jz1r_acc[idx])
                cuda.atomic.add(j_z_m1.imag, (izp, irp), jz1i_acc[idx])

                cuda.atomic.add(j_r_m2.real, (izp, irp), jr2r_acc[idx])
                cuda.atomic.add(j_r_m2.imag, (izp, irp), jr2i_acc[idx])
                cuda.atomic.add(j_t_m2.real, (izp, irp), jt2r_acc[idx])
                cuda.atomic.add(j_t_m2.imag, (izp, irp), jt2i_acc[idx])
                cuda.atomic.add(j_z_m2.real, (izp, irp), jz2r_acc[idx])
                cuda.atomic.add(j_z_m2.imag, (izp, irp), jz2i_acc[idx])
