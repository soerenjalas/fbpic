# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Partile-In-Cell code (FB-PIC)
It defines the FFT object, which performs Fourier transforms along the axis 0,
and is used in spectral_transformer.py
"""
import os
import numpy as np
import numba
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    from fbpic.utils.cuda import cuda_tpb_bpg_2d, get_cuda_copy_tpb
    from .cuda_methods import cuda_copy_2d_to_1d, cuda_copy_1d_to_2d, \
        cuda_copy_1d_to_2d_and_scale
    import cupy
    from cupy.cuda import cufft

# Check if the MKL FFT is available
try:
    from .mkl_fft import MKLFFT
    mkl_installed = True
except OSError:
    import pyfftw
    mkl_installed = False

class FFT(object):
    """
    Object that performs Fourier transform of 2D arrays along the z axis,
    (axis 0) either on the CPU (using pyfftw) or on the GPU (using cufft)

    See the methods `transform` and `inverse transform` for more information
    """

    def __init__(self, Nr, Nz, use_cuda=False, nthreads=None ):
        """
        Initialize an FFT object

        Parameters
        ----------
        Nr: int
           Number of grid points along the r axis (axis -1)

        Nz: int
           Number of grid points along the z axis (axis 0)

        use_cuda: bool, optional
           Whether to perform the Fourier transform on the z axis

        nthreads : int, optional
            Number of threads for the FFTW transform.
            If None, the default number of threads of numba is used
            (environment variable NUMBA_NUM_THREADS)
        """
        # Check whether to use cuda
        self.use_cuda = use_cuda
        if (self.use_cuda is True) and (cuda_installed is False) :
            self.use_cuda = False
            print('** Cuda not available for Fourier transform.')
            print('** Performing the Fourier transform on the CPU.')

        # Check whether to use MKL
        self.use_mkl = mkl_installed

        # Initialize the object for calculation on the GPU
        if self.use_cuda:
            # Optional experimental path: perform FFT directly on the 2D array
            # with cupy.fft along axis 0 (no explicit custom copy kernels).
            # This can be toggled for A/B testing.
            self.use_direct_axis0_fft = os.environ.get(
                'FBPIC_USE_DIRECT_AXIS0_FFT', '0').lower() in ('1', 'true', 'yes')

            # Set number of CUDA threads per block for copy 1d/2d kernels
            # (can be overridden via FBPIC_COPY_TPBX/FBPIC_COPY_TPBY)
            copy_tpb = get_cuda_copy_tpb(default_v100=(8,32), default_other=(2,16))
            # Initialize the dimension of the grid and blocks
            self.dim_grid, self.dim_block = cuda_tpb_bpg_2d(Nz, Nr, *copy_tpb)
            # Initialize 1d buffer for cufft
            self.buffer1d_in = cupy.empty(
                (Nz*Nr,), dtype=np.complex128)
            self.buffer1d_out = cupy.empty(
                (Nz*Nr,), dtype=np.complex128)
            # Initialize the CUDA FFT plan object
            self.fft = cufft.Plan1d(Nz, cufft.CUFFT_Z2Z, Nr)
            self.inv_Nz = 1./Nz         # For normalization of the iFFT
            # Whether to fuse inverse FFT normalization into copy kernel
            # (set FBPIC_FFT_FUSE_IFFT_SCALE=0 to disable for A/B testing)
            self.fuse_ifft_scale = os.environ.get(
                'FBPIC_FFT_FUSE_IFFT_SCALE', '1').lower() not in ('0', 'false', 'no')

        # Initialize the object for calculation on the CPU
        else:
            # For MKL FFT
            if self.use_mkl:
                # Initialize the MKL plan with dummy array
                spect_buffer = np.zeros( (Nz, Nr), dtype=np.complex128 )
                self.mklfft = MKLFFT( spect_buffer )

            # For FFTW
            else:
                # Determine number of threads
                if nthreads is None:
                    # Get the default number of threads for numba
                    nthreads = numba.config.NUMBA_NUM_THREADS
                # Initialize the FFT plan with dummy arrays
                interp_buffer = np.zeros( (Nz, Nr), dtype=np.complex128 )
                spect_buffer = np.zeros( (Nz, Nr), dtype=np.complex128 )
                self.fft = pyfftw.FFTW( interp_buffer, spect_buffer,
                        axes=(0,), direction='FFTW_FORWARD', threads=nthreads)
                self.ifft = pyfftw.FFTW( spect_buffer, interp_buffer,
                        axes=(0,), direction='FFTW_BACKWARD', threads=nthreads)


    def transform( self, array_in, array_out ):
        """
        Perform the Fourier transform of array_in,
        and store the result in array_out

        Parameters
        ----------
        array_in, array_out: cuda device arrays or numpy arrays
            When using the GPU, these should be cuda device array.
            When using the CPU, array_out should be one of the
            two buffers that are returned by `get_buffers`
        """
        if self.use_cuda :
            if self.use_direct_axis0_fft:
                # Experimental direct path: rely on cupy.fft for axis-0 transform.
                # Note: cupy.fft returns a new array; copy into pre-allocated output.
                array_out[:, :] = cupy.fft.fft(array_in, axis=0)
            else:
                # Copy 2D arrays to 1D array for optimized 1D batch FFT
                cuda_copy_2d_to_1d[self.dim_grid, self.dim_block](
                    array_in, self.buffer1d_in)
                # Perform forward FFT
                self.fft.fft(self.buffer1d_in,
                             self.buffer1d_out,
                             cufft.CUFFT_FORWARD)
                # Copy 1D arrays back to 2D array
                cuda_copy_1d_to_2d[self.dim_grid, self.dim_block](
                    self.buffer1d_out, array_out)
        elif self.use_mkl:
            # Perform the FFT on the CPU using MKL
            self.mklfft.transform( array_in, array_out )
        else :
            # Perform the FFT on the CPU using FFTW
            self.fft.update_arrays( new_input_array=array_in,
                                    new_output_array=array_out )
            self.fft()

    def inverse_transform( self, array_in, array_out ):
        """
        Perform the inverse Fourier transform of array_in,
        and store the result in array_out

        Parameters
        ----------
        array_in, array_out: cuda device arrays or numpy arrays
            When using the GPU, these should be cuda device array.
            When using the CPU, array_in should be one of the
            two buffers that are returned by `get_buffers`
        """
        if self.use_cuda :
            if self.use_direct_axis0_fft:
                # Experimental direct path: cupy.fft.ifft uses numpy normalization
                # convention and already applies 1/N scaling.
                array_out[:, :] = cupy.fft.ifft(array_in, axis=0)
            else:
                # Copy 2D arrays to 1D array for optimized 1D batch FFT
                cuda_copy_2d_to_1d[self.dim_grid, self.dim_block](
                    array_in, self.buffer1d_in)
                # Perform inverse FFT
                self.fft.fft(self.buffer1d_in,
                             self.buffer1d_out,
                             cufft.CUFFT_INVERSE)
                if self.fuse_ifft_scale:
                    # Copy 1D arrays back to 2D array and normalize in one kernel
                    cuda_copy_1d_to_2d_and_scale[self.dim_grid, self.dim_block](
                        self.buffer1d_out, array_out, self.inv_Nz)
                else:
                    # Legacy two-step path (useful for A/B profiling)
                    cupy.multiply(self.buffer1d_out, self.inv_Nz, out=self.buffer1d_out)
                    cuda_copy_1d_to_2d[self.dim_grid, self.dim_block](
                        self.buffer1d_out, array_out)
        elif self.use_mkl:
            # Perform the inverse FFT on the CPU using MKL
            self.mklfft.inverse_transform( array_in, array_out )
        else :
            # Perform the inverse FFT on the CPU using FFTW
            self.ifft.update_arrays( new_input_array=array_in,
                                    new_output_array=array_out )
            self.ifft()
