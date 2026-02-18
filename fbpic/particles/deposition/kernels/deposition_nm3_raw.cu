extern "C" {

__device__ __forceinline__ double Sz_cubic(double cell_position, int index) {
    int iz = (int)ceil(cell_position) - 2;
    double u = cell_position - iz - 1.0;
    if (index == 0) {
        return (1.0/6.0) * (1.0-u)*(1.0-u)*(1.0-u);
    }
    else if (index == 1) {
        return (1.0/6.0) * (3.0*u*u*u - 6.0*u*u + 4.0);
    }
    else if (index == 2) {
        double t = 1.0 - u;
        return (1.0/6.0) * (3.0*t*t*t - 6.0*t*t + 4.0);
    }
    else {
        return (1.0/6.0) * u*u*u;
    }
}

__device__ __forceinline__ double Sr_cubic(
    double cell_position, int index, int flip, double beta_n)
{
    int ir = (int)ceil(cell_position) - 2;
    double u = cell_position - ir - 1.0;

    double s;
    if (index == 0) {
        double t = 1.0 - u;
        s = (1.0/6.0) * t*t*t;
    }
    else if (index == 1) {
        s = (1.0/6.0) * (3.0*u*u*u - 6.0*u*u + 4.0);
        s += beta_n * (1.0-u) * u;
    }
    else if (index == 2) {
        double t = 1.0 - u;
        s = (1.0/6.0) * (3.0*t*t*t - 6.0*t*t + 4.0);
        s -= beta_n * (1.0-u) * u;
    }
    else {
        s = (1.0/6.0) * u*u*u;
    }

    if (index + ir < 0) {
        s *= (double)flip;
    }
    return s;
}

__global__ void deposit_rho_gpu_unsorted_cubic_m3_raw(
    const double* x,
    const double* y,
    const double* z,
    const double* w,
    const double q,
    const double invdz,
    const double zmin,
    const int Nz,
    const double invdr,
    const double rmin,
    const int Nr,
    double* rho_m0,
    double* rho_m1,
    double* rho_m2,
    const double* beta_n_m0,
    const double* beta_n_m1,
    const double* beta_n_m2,
    const int Ntot)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= Ntot) {
        return;
    }

    const double xj = x[i];
    const double yj = y[i];
    const double zj = z[i];
    const double wj = q * w[i];

    const double rj = sqrt(xj*xj + yj*yj);
    double cs, sn;
    if (rj != 0.0) {
        const double invr = 1.0/rj;
        cs = xj*invr;
        sn = yj*invr;
    }
    else {
        cs = 1.0;
        sn = 0.0;
    }

    const double cs2 = cs*cs - sn*sn;
    const double sn2 = 2.0*cs*sn;

    const double r_cell = invdr*(rj - rmin) - 0.5;
    const double z_cell = invdz*(zj - zmin) - 0.5;

    const int ir = min((int)ceil(r_cell), Nr);
    int iz = (int)ceil(z_cell);
    if (iz < 0) {
        iz += Nz;
    }
    else if (iz >= Nz) {
        iz -= Nz;
    }

    int iz0 = iz - 2;
    int iz1 = iz - 1;
    int iz2 = iz;
    int iz3 = iz + 1;
    if (iz0 < 0) iz0 += Nz;
    if (iz1 < 0) iz1 += Nz;
    if (iz3 > Nz-1) iz3 -= Nz;

    int ir0 = ir - 2;
    int ir1 = min(ir - 1, Nr - 1);
    int ir2 = min(ir, Nr - 1);
    int ir3 = min(ir + 1, Nr - 1);
    if (ir0 < 0) ir0 = -(1 + ir0);
    if (ir1 < 0) ir1 = -(1 + ir1);

    const double bn0 = beta_n_m0[ir];
    const double bn1 = beta_n_m1[ir];
    const double bn2 = beta_n_m2[ir];

    const double Sz0 = Sz_cubic(z_cell, 0);
    const double Sz1 = Sz_cubic(z_cell, 1);
    const double Sz2 = Sz_cubic(z_cell, 2);
    const double Sz3 = Sz_cubic(z_cell, 3);

    const double Sr00 = Sr_cubic(r_cell, 0,  1, bn0);
    const double Sr01 = Sr_cubic(r_cell, 1,  1, bn0);
    const double Sr02 = Sr_cubic(r_cell, 2,  1, bn0);
    const double Sr03 = Sr_cubic(r_cell, 3,  1, bn0);

    const double Sr10 = Sr_cubic(r_cell, 0, -1, bn1);
    const double Sr11 = Sr_cubic(r_cell, 1, -1, bn1);
    const double Sr12 = Sr_cubic(r_cell, 2, -1, bn1);
    const double Sr13 = Sr_cubic(r_cell, 3, -1, bn1);

    const double Sr20 = Sr_cubic(r_cell, 0,  1, bn2);
    const double Sr21 = Sr_cubic(r_cell, 1,  1, bn2);
    const double Sr22 = Sr_cubic(r_cell, 2,  1, bn2);
    const double Sr23 = Sr_cubic(r_cell, 3,  1, bn2);

    const double R0 = wj;
    const double R1_r = wj * cs;
    const double R1_i = wj * sn;
    const double R2_r = wj * cs2;
    const double R2_i = wj * sn2;

    const int izs[4] = {iz0, iz1, iz2, iz3};
    const int irs[4] = {ir0, ir1, ir2, ir3};
    const double Sz[4] = {Sz0, Sz1, Sz2, Sz3};

    const double Sr0[4] = {Sr00, Sr01, Sr02, Sr03};
    const double Sr1[4] = {Sr10, Sr11, Sr12, Sr13};
    const double Sr2[4] = {Sr20, Sr21, Sr22, Sr23};

    for (int ia = 0; ia < 4; ia++) {
        const int izp = izs[ia];
        const double sz = Sz[ia];

        for (int ib = 0; ib < 4; ib++) {
            const int irp = irs[ib];
            const int idx2 = 2*(izp*Nr + irp);

            const double w0 = sz * Sr0[ib];
            const double w1 = sz * Sr1[ib];
            const double w2 = sz * Sr2[ib];

            // m=0 (real only)
            atomicAdd(&rho_m0[idx2], w0 * R0);

            // m=1
            atomicAdd(&rho_m1[idx2],   w1 * R1_r);
            atomicAdd(&rho_m1[idx2+1], w1 * R1_i);

            // m=2
            atomicAdd(&rho_m2[idx2],   w2 * R2_r);
            atomicAdd(&rho_m2[idx2+1], w2 * R2_i);
        }
    }
}

__global__ void deposit_J_gpu_unsorted_rel_cubic_m3_raw(
    const double* x,
    const double* y,
    const double* z,
    const double* w,
    const double q,
    const double* ux,
    const double* uy,
    const double* uz,
    const double* inv_gamma,
    const double invdz,
    const double zmin,
    const int Nz,
    const double invdr,
    const double rmin,
    const int Nr,
    double* j_r_m0,
    double* j_t_m0,
    double* j_z_m0,
    double* j_r_m1,
    double* j_t_m1,
    double* j_z_m1,
    double* j_r_m2,
    double* j_t_m2,
    double* j_z_m2,
    const double* beta_n_m0,
    const double* beta_n_m1,
    const double* beta_n_m2,
    const int Ntot)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= Ntot) {
        return;
    }

    const double c = 299792458.0;

    const double xj = x[i];
    const double yj = y[i];
    const double zj = z[i];

    const double uxj = ux[i];
    const double uyj = uy[i];
    const double uzj = uz[i];
    const double inv_gammaj = inv_gamma[i];
    const double wj = q * w[i];

    const double rj = sqrt(xj*xj + yj*yj);
    double cs, sn;
    if (rj != 0.0) {
        const double invr = 1.0/rj;
        cs = xj*invr;
        sn = yj*invr;
    }
    else {
        cs = 1.0;
        sn = 0.0;
    }

    const double cs2 = cs*cs - sn*sn;
    const double sn2 = 2.0*cs*sn;

    const double r_cell = invdr*(rj - rmin) - 0.5;
    const double z_cell = invdz*(zj - zmin) - 0.5;

    const int ir = min((int)ceil(r_cell), Nr);
    int iz = (int)ceil(z_cell);
    if (iz < 0) {
        iz += Nz;
    }
    else if (iz >= Nz) {
        iz -= Nz;
    }

    int iz0 = iz - 2;
    int iz1 = iz - 1;
    int iz2 = iz;
    int iz3 = iz + 1;
    if (iz0 < 0) iz0 += Nz;
    if (iz1 < 0) iz1 += Nz;
    if (iz3 > Nz-1) iz3 -= Nz;

    int ir0 = ir - 2;
    int ir1 = min(ir - 1, Nr - 1);
    int ir2 = min(ir, Nr - 1);
    int ir3 = min(ir + 1, Nr - 1);
    if (ir0 < 0) ir0 = -(1 + ir0);
    if (ir1 < 0) ir1 = -(1 + ir1);

    const double bn0 = beta_n_m0[ir];
    const double bn1 = beta_n_m1[ir];
    const double bn2 = beta_n_m2[ir];

    const double Sz0 = Sz_cubic(z_cell, 0);
    const double Sz1 = Sz_cubic(z_cell, 1);
    const double Sz2 = Sz_cubic(z_cell, 2);
    const double Sz3 = Sz_cubic(z_cell, 3);

    const double Sr_rt00 = Sr_cubic(r_cell, 0, -1, bn0);
    const double Sr_rt01 = Sr_cubic(r_cell, 1, -1, bn0);
    const double Sr_rt02 = Sr_cubic(r_cell, 2, -1, bn0);
    const double Sr_rt03 = Sr_cubic(r_cell, 3, -1, bn0);

    const double Sr_rt10 = Sr_cubic(r_cell, 0,  1, bn1);
    const double Sr_rt11 = Sr_cubic(r_cell, 1,  1, bn1);
    const double Sr_rt12 = Sr_cubic(r_cell, 2,  1, bn1);
    const double Sr_rt13 = Sr_cubic(r_cell, 3,  1, bn1);

    const double Sr_rt20 = Sr_cubic(r_cell, 0, -1, bn2);
    const double Sr_rt21 = Sr_cubic(r_cell, 1, -1, bn2);
    const double Sr_rt22 = Sr_cubic(r_cell, 2, -1, bn2);
    const double Sr_rt23 = Sr_cubic(r_cell, 3, -1, bn2);

    const double Sr_z00 = Sr_cubic(r_cell, 0,  1, bn0);
    const double Sr_z01 = Sr_cubic(r_cell, 1,  1, bn0);
    const double Sr_z02 = Sr_cubic(r_cell, 2,  1, bn0);
    const double Sr_z03 = Sr_cubic(r_cell, 3,  1, bn0);

    const double Sr_z10 = Sr_cubic(r_cell, 0, -1, bn1);
    const double Sr_z11 = Sr_cubic(r_cell, 1, -1, bn1);
    const double Sr_z12 = Sr_cubic(r_cell, 2, -1, bn1);
    const double Sr_z13 = Sr_cubic(r_cell, 3, -1, bn1);

    const double Sr_z20 = Sr_cubic(r_cell, 0,  1, bn2);
    const double Sr_z21 = Sr_cubic(r_cell, 1,  1, bn2);
    const double Sr_z22 = Sr_cubic(r_cell, 2,  1, bn2);
    const double Sr_z23 = Sr_cubic(r_cell, 3,  1, bn2);

    const double base = wj * c * inv_gammaj;
    const double jr0 = base * (cs*uxj + sn*uyj);
    const double jt0 = base * (cs*uyj - sn*uxj);
    const double jz0 = base * uzj;

    const double jr1_r = jr0 * cs;
    const double jr1_i = jr0 * sn;
    const double jt1_r = jt0 * cs;
    const double jt1_i = jt0 * sn;
    const double jz1_r = jz0 * cs;
    const double jz1_i = jz0 * sn;

    const double jr2_r = jr0 * cs2;
    const double jr2_i = jr0 * sn2;
    const double jt2_r = jt0 * cs2;
    const double jt2_i = jt0 * sn2;
    const double jz2_r = jz0 * cs2;
    const double jz2_i = jz0 * sn2;

    const int izs[4] = {iz0, iz1, iz2, iz3};
    const int irs[4] = {ir0, ir1, ir2, ir3};
    const double Sz[4] = {Sz0, Sz1, Sz2, Sz3};

    const double Sr_rt0[4] = {Sr_rt00, Sr_rt01, Sr_rt02, Sr_rt03};
    const double Sr_rt1[4] = {Sr_rt10, Sr_rt11, Sr_rt12, Sr_rt13};
    const double Sr_rt2[4] = {Sr_rt20, Sr_rt21, Sr_rt22, Sr_rt23};
    const double Sr_z0[4] = {Sr_z00, Sr_z01, Sr_z02, Sr_z03};
    const double Sr_z1[4] = {Sr_z10, Sr_z11, Sr_z12, Sr_z13};
    const double Sr_z2[4] = {Sr_z20, Sr_z21, Sr_z22, Sr_z23};

    for (int ia = 0; ia < 4; ia++) {
        const int izp = izs[ia];
        const double sz = Sz[ia];

        for (int ib = 0; ib < 4; ib++) {
            const int irp = irs[ib];
            const int idx2 = 2*(izp*Nr + irp);

            const double w_rt0 = sz * Sr_rt0[ib];
            const double w_rt1 = sz * Sr_rt1[ib];
            const double w_rt2 = sz * Sr_rt2[ib];
            const double w_z0 = sz * Sr_z0[ib];
            const double w_z1 = sz * Sr_z1[ib];
            const double w_z2 = sz * Sr_z2[ib];

            // m=0 (real only)
            atomicAdd(&j_r_m0[idx2], w_rt0 * jr0);
            atomicAdd(&j_t_m0[idx2], w_rt0 * jt0);
            atomicAdd(&j_z_m0[idx2], w_z0  * jz0);

            // m=1
            atomicAdd(&j_r_m1[idx2],   w_rt1 * jr1_r);
            atomicAdd(&j_r_m1[idx2+1], w_rt1 * jr1_i);
            atomicAdd(&j_t_m1[idx2],   w_rt1 * jt1_r);
            atomicAdd(&j_t_m1[idx2+1], w_rt1 * jt1_i);
            atomicAdd(&j_z_m1[idx2],   w_z1  * jz1_r);
            atomicAdd(&j_z_m1[idx2+1], w_z1  * jz1_i);

            // m=2
            atomicAdd(&j_r_m2[idx2],   w_rt2 * jr2_r);
            atomicAdd(&j_r_m2[idx2+1], w_rt2 * jr2_i);
            atomicAdd(&j_t_m2[idx2],   w_rt2 * jt2_r);
            atomicAdd(&j_t_m2[idx2+1], w_rt2 * jt2_i);
            atomicAdd(&j_z_m2[idx2],   w_z2  * jz2_r);
            atomicAdd(&j_z_m2[idx2+1], w_z2  * jz2_i);
        }
    }
}

} // extern "C"
