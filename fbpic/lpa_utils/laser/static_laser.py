from scipy.constants import c
from .direct_injection import add_laser_direct
from numba import cuda
from fbpic.utils.cuda import send_data_to_gpu, receive_data_from_gpu
from fbpic.utils.cuda import cuda_tpb_bpg_2d

class StaticField(object):
    """Freeze inital field and propagate it with the velocity of the moving window"""

    def __init__( self, sim, use_cuda):
        self.sim = sim
        self.use_cuda = use_cuda
        self.Nz = sim.fld.Nz
        self.Nr = sim.fld.Nr
        self.Nm = sim.fld.Nm
        self.Ep = []
        self.Em = []
        self.Ez = []
        self.Bp = []
        self.Bm = []
        self.Bz = []

        if use_cuda:
            receive_data_from_gpu(sim)

        for spect in sim.fld.spect:
            self.Ep.append(spect.Ep.copy())
            self.Em.append(spect.Em.copy())
            self.Ez.append(spect.Ez.copy())
            self.Bp.append(spect.Bp.copy())
            self.Bm.append(spect.Bm.copy())
            self.Bz.append(spect.Bz.copy())

        if use_cuda:
            send_data_to_gpu(sim)
            self.send_to_gpu()

    def send_to_gpu( self ):
        for i in range(self.Nm):
            self.Ep[i] = cuda.to_device( self.Ep[i] )
            self.Em[i] = cuda.to_device( self.Em[i] )
            self.Ez[i] = cuda.to_device( self.Ez[i] )
            self.Bp[i] = cuda.to_device( self.Bp[i] )
            self.Bm[i] = cuda.to_device( self.Bm[i] )
            self.Bz[i] = cuda.to_device( self.Bz[i] )


    def add_field( self ):
        for i in range(self.Nm):
            fld = self.sim.fld.spect[i]
            if self.use_cuda:
                dim_grid, dim_block = cuda_tpb_bpg_2d( self.Nz, self.Nr, 1, 16 )
                cuda_add_static_field[dim_grid, dim_block](
                               fld.Ep, fld.Em, fld.Ez,
                               fld.Bp, fld.Bm, fld.Bz,
                               self.Ep[i], self.Em[i], self.Ez[i],
                               self.Bp[i], self.Bm[i], self.Bz[i],
                               self.Nz, self.Nr )
            else:
                cpu_add_static_field( fld.Ep, fld.Em, fld.Ez,
                               fld.Bp, fld.Bm, fld.Bz,
                               self.Ep[i], self.Em[i], self.Ez[i],
                               self.Bp[i], self.Bm[i], self.Bz[i],
                               self.Nz, self.Nr )

    def remove_field( self ):
        for i in range(self.Nm):
            fld = self.sim.fld.spect[i]
            if self.use_cuda:
                dim_grid, dim_block = cuda_tpb_bpg_2d( self.Nz, self.Nr, 1, 16 )
                cuda_remove_static_field[dim_grid, dim_block](
                               fld.Ep, fld.Em, fld.Ez,
                               fld.Bp, fld.Bm, fld.Bz,
                               self.Ep[i], self.Em[i], self.Ez[i],
                               self.Bp[i], self.Bm[i], self.Bz[i],
                               self.Nz, self.Nr )
            else:
                cpu_remove_static_field( fld.Ep, fld.Em, fld.Ez,
                               fld.Bp, fld.Bm, fld.Bz,
                               self.Ep[i], self.Em[i], self.Ez[i],
                               self.Bp[i], self.Bm[i], self.Bz[i],
                               self.Nz, self.Nr )


@cuda.jit
def cuda_add_static_field( Ep, Em, Ez, Bp, Bm, Bz,
                           Eps, Ems, Ezs, Bps, Bms, Bzs,
                           Nz, Nr):
    # Cuda 2D grid
    iz, ir = cuda.grid(2)

    if (iz < Nz) and (ir < Nr) :

        Ep[iz, ir] += Eps[iz, ir]
        Em[iz, ir] += Ems[iz, ir]
        Ez[iz, ir] += Ezs[iz, ir]
        Bp[iz, ir] += Bps[iz, ir]
        Bm[iz, ir] += Bms[iz, ir]
        Bz[iz, ir] += Bzs[iz, ir]

def cpu_add_static_field( Ep, Em, Ez, Bp, Bm, Bz,
                           Eps, Ems, Ezs, Bps, Bms, Bzs,
                           Nz, Nr):

        Ep += Eps
        Em += Ems
        Ez += Ezs
        Bp += Bps
        Bm += Bms
        Bz += Bzs

@cuda.jit
def cuda_remove_static_field( Ep, Em, Ez, Bp, Bm, Bz,
                           Eps, Ems, Ezs, Bps, Bms, Bzs,
                           Nz, Nr):
    # Cuda 2D grid
    iz, ir = cuda.grid(2)

    if (iz < Nz) and (ir < Nr) :

        Ep[iz, ir] -= Eps[iz, ir]
        Em[iz, ir] -= Ems[iz, ir]
        Ez[iz, ir] -= Ezs[iz, ir]
        Bp[iz, ir] -= Bps[iz, ir]
        Bm[iz, ir] -= Bms[iz, ir]
        Bz[iz, ir] -= Bzs[iz, ir]

def cpu_remove_static_field( Ep, Em, Ez, Bp, Bm, Bz,
                           Eps, Ems, Ezs, Bps, Bms, Bzs,
                           Nz, Nr):

        Ep -= Eps
        Em -= Ems
        Ez -= Ezs
        Bp -= Bps
        Bm -= Bms
        Bz -= Bzs