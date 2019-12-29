#include "Particles.h"
#include "Alloc.h"
#include "helper.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <cmath>
#include <algorithm>

#define TPB 32


/** "Global" environment auxilliary variables necessary to run h_move_particle() */
typedef struct {
    FPpart dt_sub_cycling;
    FPpart dto2;
    FPpart qomdt2;
} dt_info;


/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}


/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** GPU kernel to move a single particle */
__global__ void g_move_particle(int stream_offset, int nop, int n_sub_cycles, int part_NiterMover, struct grid grd,
                                struct parameters param, const dt_info dt_inf,
                                particles_pointers part, const field_pointers field, const grd_pointers grd_p) 
{
    // getting thread ID
    const int i = blockIdx.x * blockDim.x + threadIdx.x + stream_offset;
    if (i - stream_offset >= nop) return;

    // auxiliary variables
    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];

    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    xptilde = part.x[i];
    yptilde = part.y[i];
    zptilde = part.z[i];

    // start subcycling
    for (int i_sub=0; i_sub < n_sub_cycles; i_sub++) {
        // calculate the average velocity iteratively
        for (int innter = 0; innter < part_NiterMover; innter++) {
            // interpolation G-->P
            ix = 2 + int((part.x[i] - grd.xStart) * grd.invdx);
            iy = 2 + int((part.y[i] - grd.yStart) * grd.invdy);
            iz = 2 + int((part.z[i] - grd.zStart) * grd.invdz);

            // calculate weights
            xi[0] = part.x[i] - grd_p.XN_flat[get_idx(ix - 1, iy, iz, grd.nyn, grd.nzn)];
            eta[0] = part.y[i] - grd_p.YN_flat[get_idx(ix, iy - 1, iz, grd.nyn, grd.nzn)];
            zeta[0] = part.z[i] - grd_p.ZN_flat[get_idx(ix, iy, iz - 1, grd.nyn, grd.nzn)];
            xi[1] = grd_p.XN_flat[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - part.x[i];
            eta[1] = grd_p.YN_flat[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - part.y[i];
            zeta[1] = grd_p.ZN_flat[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - part.z[i];
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                        weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd.invVOL;

            // set to zero local electric and magnetic field
            Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++) {
                        Exl += weight[ii][jj][kk] * field.Ex_flat[get_idx(ix - ii, iy - jj, iz - kk, grd.nyn, grd.nzn)];
                        Eyl += weight[ii][jj][kk] * field.Ey_flat[get_idx(ix - ii, iy - jj, iz - kk, grd.nyn, grd.nzn)];
                        Ezl += weight[ii][jj][kk] * field.Ez_flat[get_idx(ix - ii, iy - jj, iz - kk, grd.nyn, grd.nzn)];
                        Bxl += weight[ii][jj][kk] *
                               field.Bxn_flat[get_idx(ix - ii, iy - jj, iz - kk, grd.nyn, grd.nzn)];
                        Byl += weight[ii][jj][kk] *
                               field.Byn_flat[get_idx(ix - ii, iy - jj, iz - kk, grd.nyn, grd.nzn)];
                        Bzl += weight[ii][jj][kk] *
                               field.Bzn_flat[get_idx(ix - ii, iy - jj, iz - kk, grd.nyn, grd.nzn)];
                    }

            // end interpolation
            omdtsq = dt_inf.qomdt2 * dt_inf.qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
            denom = 1.0 / (1.0 + omdtsq);
            // solve the position equation
            ut = part.u[i] + dt_inf.qomdt2 * Exl;
            vt = part.v[i] + dt_inf.qomdt2 * Eyl;
            wt = part.w[i] + dt_inf.qomdt2 * Ezl;
            udotb = ut * Bxl + vt * Byl + wt * Bzl;
            // solve the velocity equation
            uptilde = (ut + dt_inf.qomdt2 * (vt * Bzl - wt * Byl + dt_inf.qomdt2 * udotb * Bxl)) * denom;
            vptilde = (vt + dt_inf.qomdt2 * (wt * Bxl - ut * Bzl + dt_inf.qomdt2 * udotb * Byl)) * denom;
            wptilde = (wt + dt_inf.qomdt2 * (ut * Byl - vt * Bxl + dt_inf.qomdt2 * udotb * Bzl)) * denom;
            // update position
            part.x[i] = xptilde + uptilde * dt_inf.dto2;
            part.y[i] = yptilde + vptilde * dt_inf.dto2;
            part.z[i] = zptilde + wptilde * dt_inf.dto2;


        } // end of iteration
        // update the final position and velocity
        part.u[i] = 2.0 * uptilde - part.u[i];
        part.v[i] = 2.0 * vptilde - part.v[i];
        part.w[i] = 2.0 * wptilde - part.w[i];
        part.x[i] = xptilde + uptilde * dt_inf.dt_sub_cycling;
        part.y[i] = yptilde + vptilde * dt_inf.dt_sub_cycling;
        part.z[i] = zptilde + wptilde * dt_inf.dt_sub_cycling;


        //////////
        //////////
        ////////// BC

        // X-DIRECTION: BC particles
        if (part.x[i] > grd.Lx) {
            if (param.PERIODICX == true) { // PERIODIC
                part.x[i] = part.x[i] - grd.Lx;
            } else { // REFLECTING BC
                part.u[i] = -part.u[i];
                part.x[i] = 2 * grd.Lx - part.x[i];
            }
        }

        if (part.x[i] < 0) {
            if (param.PERIODICX == true) { // PERIODIC
                part.x[i] = part.x[i] + grd.Lx;
            } else { // REFLECTING BC
                part.u[i] = -part.u[i];
                part.x[i] = -part.x[i];
            }
        }


        // Y-DIRECTION: BC particles
        if (part.y[i] > grd.Ly) {
            if (param.PERIODICY == true) { // PERIODIC
                part.y[i] = part.y[i] - grd.Ly;
            } else { // REFLECTING BC
                part.v[i] = -part.v[i];
                part.y[i] = 2 * grd.Ly - part.y[i];
            }
        }

        if (part.y[i] < 0) {
            if (param.PERIODICY == true) { // PERIODIC
                part.y[i] = part.y[i] + grd.Ly;
            } else { // REFLECTING BC
                part.v[i] = -part.v[i];
                part.y[i] = -part.y[i];
            }
        }

        // Z-DIRECTION: BC particles
        if (part.z[i] > grd.Lz) {
            if (param.PERIODICZ == true) { // PERIODIC
                part.z[i] = part.z[i] - grd.Lz;
            } else { // REFLECTING BC
                part.w[i] = -part.w[i];
                part.z[i] = 2 * grd.Lz - part.z[i];
            }
        }

        if (part.z[i] < 0) {
            if (param.PERIODICZ == true) { // PERIODIC
                part.z[i] = part.z[i] + grd.Lz;
            } else { // REFLECTING BC
                part.w[i] = -part.w[i];
                part.z[i] = -part.z[i];
            }
        }
    }

}


/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param,
             particles_pointers p_p, field_pointers f_p, grd_pointers g_p, int grdSize, int field_size) 
{
    // print species and subcycling
    std::cout << std::endl << "***  In [mover_PC]: MOVER with SUBCYCLYING "<< param->n_sub_cycles
              << " - species " << part->species_ID << " ***" << std::endl;

    // "global" environment variables
    FPpart dt_sub_cycling = (FPpart) param->dt / ((double) part->n_sub_cycles);
    FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;
    const dt_info dt_inf { dt_sub_cycling, dto2, qomdt2 };

    /*
     * The following steps are taken:
     * 1. Copy grd and field to GPU 
     * 2. For each batch/full batch:
     *     1. Copy relevant Particles to GPU
     *     2. Launch kernels that modify Particles in GPU memory
     *     3. Copy back relevant Particles to CPU
     */

    // Copy grid/field constants to GPU
    copy_mover_constants_to_GPU(field, grd, f_p, g_p, grdSize, field_size);

    // If npmax <= MAX_GPU_PARTICLES, n_batches = 1, and all movement is done in one batch
    int n_batches = (part->npmax + MAX_GPU_PARTICLES - 1) / MAX_GPU_PARTICLES;
    for (int batch_no = 0; batch_no < n_batches; batch_no++) 
    {
        // Compute batch size/bounds
        long batch_start = batch_no * MAX_GPU_PARTICLES;
        long batch_end = std::min(batch_start + MAX_GPU_PARTICLES, part->npmax);  // max is part->npmax
        long batch_size = batch_end - batch_start;

        // Copy particles in batch to GPU (part in CPU to p_p on GPU)
        copy_mover_arrays(part, p_p, CPU_TO_GPU, batch_start, batch_end);

        // Launch the kernel to perform on the batch
        g_move_particle<<<(batch_size+TPB-1)/TPB, TPB>>>(0, batch_size, part->n_sub_cycles, part->NiterMover, 
                                                         *grd, *param, dt_inf, p_p, f_p, g_p);
        cudaDeviceSynchronize();

        // Copy moved particles back (p_p in GPU back to part in CPU).
        copy_mover_arrays(part, p_p, GPU_TO_CPU, batch_start, batch_end);

        std::cout << "====== In [mover_PC]: batch " << (batch_no + 1) << " of " << n_batches << ": done." << std::endl;
    }
    
    return(0); // exit successfully
}


/** GPU kernel to interpolate for a single particle during one subcycle. */
__global__ void g_interp_particle(int stream_offset, int nop, struct grid grd,
                                  const particles_pointers part, ids_pointers ids, const grd_pointers grd_p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x + stream_offset;
    if (i - stream_offset >= nop) return;

    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];

    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int (floor((part.x[i] - grd.xStart) * grd.invdx));
    iy = 2 + int (floor((part.y[i] - grd.yStart) * grd.invdy));
    iz = 2 + int (floor((part.z[i] - grd.zStart) * grd.invdz));

    // distances from node
    xi[0]   = part.x[i] - grd_p.XN_flat[get_idx(ix-1, iy, iz, grd.nyn, grd.nzn)];
    eta[0]  = part.y[i] - grd_p.YN_flat[get_idx(ix, iy-1, iz, grd.nyn, grd.nzn)];
    zeta[0] = part.z[i] - grd_p.ZN_flat[get_idx(ix, iy, iz-1, grd.nyn, grd.nzn)];
    xi[1]   = grd_p.XN_flat[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - part.x[i];
    eta[1]  = grd_p.YN_flat[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - part.y[i];
    zeta[1] = grd_p.ZN_flat[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - part.z[i];
    {
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part.q[i] * xi[ii] * eta[jj] * zeta[kk] * grd.invVOL;

        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    atomicAdd(&ids.rhon_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);


        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part.u[i] * weight[ii][jj][kk];

        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    atomicAdd(&ids.Jx_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);


        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part.v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    atomicAdd(&ids.Jy_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);



        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part.w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    atomicAdd(&ids.Jz_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);


        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part.u[i] * part.u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    atomicAdd(&ids.pxx_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);


        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part.u[i] * part.v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    atomicAdd(&ids.pxy_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);



        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part.u[i] * part.w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    atomicAdd(&ids.pxz_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);


        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part.v[i] * part.v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    atomicAdd(&ids.pyy_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);


        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part.v[i] * part.w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    atomicAdd(&ids.pyz_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);


        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part.w[i] * part.w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    atomicAdd(&ids.pzz_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);
    }
}

void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd,
               particles_pointers p_p, ids_pointers i_p, grd_pointers g_p, int grdSize, int rhocSize)
{
    // Print species
    std::cout << std::endl << "***  In [interpP2G]: Interpolating "
              << " - species " << part->species_ID << " ***" << std::endl;

    /*
     * The following steps are taken:
     * 0. Assume grd is copied to GPU already
     * 1. Copy zero'd ids to GPU as initialization
     * 2. For each batch/full batch:
     *     1. Copy relevant Particles to GPU
     *     2. Launch kernels that modify ids in GPU memory
     * 3. Copy ids back to CPU memory
     */

    // Copy initial zero'd IDS
    copy_interp_initial_to_GPU(ids, i_p, grdSize, rhocSize);

    // If npmax <= MAX_GPU_PARTICLES, n_batches = 1, and whole interpolation is done in one batch
    int n_batches = (part->npmax + MAX_GPU_PARTICLES - 1) / MAX_GPU_PARTICLES;
    for (int batch_no = 0; batch_no < n_batches; batch_no++) 
    {
        // Compute batch size/bounds
        long batch_start = batch_no * MAX_GPU_PARTICLES;
        long batch_end = std::min(batch_start + MAX_GPU_PARTICLES, part->npmax);  // max is part->npmax
        long batch_size = batch_end - batch_start;

        // Copy particles in batch to GPU (part in CPU to p_p on GPU)
        copy_interp_particles(part, p_p, batch_start, batch_end);

        // Launch the kernel to perform on the batch
        g_interp_particle<<<(batch_size+TPB-1)/TPB, TPB>>>(0, batch_size, *grd, p_p, i_p, g_p);
        cudaDeviceSynchronize();

        std::cout << "====== In [interpP2G]: batch " << (batch_no + 1) << " of " << n_batches << ": done." << std::endl;
    }
    // Copy results back (i_p in GPU back to ids in CPU).
    copy_interp_results(ids, i_p, grdSize, rhocSize);
}

/* -------------------- *
 * CPU Serial Versions  *
 * -------------------- */

/** CPU serial function to move a single particle during one subcycle. */
__host__ void h_move_particle(int i, int part_NiterMover, struct grid* grd,
                              struct parameters* param, const dt_info dt_inf,
                              particles_pointers part, field_pointers field, grd_pointers grd_p) 
{
    // extracting the particle variables out of the auxiliary struct
    FPpart* part_x = part.x;
    FPpart* part_y = part.y;
    FPpart* part_z = part.z;
    FPpart* part_u = part.u;
    FPpart* part_v = part.v;
    FPpart* part_w = part.w;

    // extracting the field variables out of the auxiliary struct
    FPfield* field_Ex_flat = field.Ex_flat;
    FPfield* field_Ey_flat = field.Ey_flat;
    FPfield* field_Ez_flat = field.Ez_flat;
    FPfield* field_Bxn_flat = field.Bxn_flat;
    FPfield* field_Byn_flat = field.Byn_flat;
    FPfield* field_Bzn_flat = field.Bzn_flat;

    // extracting the grid variables out of the auxiliary struct
    FPfield* grd_XN_flat = grd_p.XN_flat;
    FPfield* grd_YN_flat = grd_p.YN_flat;
    FPfield* grd_ZN_flat = grd_p.ZN_flat;

    // auxiliary variables
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    xptilde = part_x[i];
    yptilde = part_y[i];
    zptilde = part_z[i];
    // calculate the average velocity iteratively
    for(int innter=0; innter < part_NiterMover; innter++){
        // interpolation G-->P
        ix = 2 +  int((part_x[i] - grd->xStart)*grd->invdx);
        iy = 2 +  int((part_y[i] - grd->yStart)*grd->invdy);
        iz = 2 +  int((part_z[i] - grd->zStart)*grd->invdz);
        
        // calculate weights
        xi[0]   = part_x[i] - grd_XN_flat[get_idx(ix-1, iy, iz, grd->nyn, grd->nzn)];
        eta[0]  = part_y[i] - grd_YN_flat[get_idx(ix, iy-1, iz, grd->nyn, grd->nzn)];
        zeta[0] = part_z[i] - grd_ZN_flat[get_idx(ix, iy, iz-1, grd->nyn, grd->nzn)];
        xi[1]   = grd_XN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part_x[i];
        eta[1]  = grd_YN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part_y[i];
        zeta[1] = grd_ZN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part_z[i];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        // set to zero local electric and magnetic field
        Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
        
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++){
                    Exl += weight[ii][jj][kk]*field_Ex_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                    Eyl += weight[ii][jj][kk]*field_Ey_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                    Ezl += weight[ii][jj][kk]*field_Ez_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                    Bxl += weight[ii][jj][kk]*field_Bxn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                    Byl += weight[ii][jj][kk]*field_Byn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                    Bzl += weight[ii][jj][kk]*field_Bzn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                }
        
        // end interpolation
        omdtsq = dt_inf.qomdt2*dt_inf.qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
        denom = 1.0/(1.0 + omdtsq);
        // solve the position equation
        ut= part_u[i] + dt_inf.qomdt2*Exl;
        vt= part_v[i] + dt_inf.qomdt2*Eyl;
        wt= part_w[i] + dt_inf.qomdt2*Ezl;
        udotb = ut*Bxl + vt*Byl + wt*Bzl;
        // solve the velocity equation
        uptilde = (ut+dt_inf.qomdt2*(vt*Bzl -wt*Byl + dt_inf.qomdt2*udotb*Bxl))*denom;
        vptilde = (vt+dt_inf.qomdt2*(wt*Bxl -ut*Bzl + dt_inf.qomdt2*udotb*Byl))*denom;
        wptilde = (wt+dt_inf.qomdt2*(ut*Byl -vt*Bxl + dt_inf.qomdt2*udotb*Bzl))*denom;
        // update position
        part_x[i] = xptilde + uptilde*dt_inf.dto2;
        part_y[i] = yptilde + vptilde*dt_inf.dto2;
        part_z[i] = zptilde + wptilde*dt_inf.dto2;


    } // end of iteration
    // update the final position and velocity
    part_u[i] = 2.0 * uptilde - part_u[i];
    part_v[i] = 2.0 * vptilde - part_v[i];
    part_w[i] = 2.0 * wptilde - part_w[i];
    part_x[i] = xptilde + uptilde * dt_inf.dt_sub_cycling;
    part_y[i] = yptilde + vptilde * dt_inf.dt_sub_cycling;
    part_z[i] = zptilde + wptilde * dt_inf.dt_sub_cycling;


    //////////
    //////////
    ////////// BC
                                
    // X-DIRECTION: BC particles
    if (part_x[i] > grd->Lx){
        if (param->PERIODICX==true){ // PERIODIC
            part_x[i] = part_x[i] - grd->Lx;
        } else { // REFLECTING BC
            part_u[i] = -part_u[i];
            part_x[i] = 2*grd->Lx - part_x[i];
        }
    }
                                                                
    if (part_x[i] < 0){
        if (param->PERIODICX==true){ // PERIODIC
           part_x[i] = part_x[i] + grd->Lx;
        } else { // REFLECTING BC
            part_u[i] = -part_u[i];
            part_x[i] = -part_x[i];
        }
    }
        

    // Y-DIRECTION: BC particles
    if (part_y[i] > grd->Ly){
        if (param->PERIODICY==true){ // PERIODIC
            part_y[i] = part_y[i] - grd->Ly;
        } else { // REFLECTING BC
            part_v[i] = -part_v[i];
            part_y[i] = 2*grd->Ly - part_y[i];
        }
    }
                                                                
    if (part_y[i] < 0){
        if (param->PERIODICY==true){ // PERIODIC
            part_y[i] = part_y[i] + grd->Ly;
        } else { // REFLECTING BC
            part_v[i] = -part_v[i];
            part_y[i] = -part_y[i];
        }
    }
                                                                
    // Z-DIRECTION: BC particles
    if (part_z[i] > grd->Lz){
        if (param->PERIODICZ==true){ // PERIODIC
            part_z[i] = part_z[i] - grd->Lz;
        } else { // REFLECTING BC
            part_w[i] = -part_w[i];
            part_z[i] = 2*grd->Lz - part_z[i];
        }
    }
                                                                
    if (part_z[i] < 0){
        if (param->PERIODICZ==true){ // PERIODIC
            part_z[i] = part_z[i] + grd->Lz;
        } else { // REFLECTING BC
            part_w[i] = -part_w[i];
            part_z[i] = -part_z[i];
        }
    }
}

/** CPU serial function to interpolate for a single particle during one subcycle. */
__host__ void h_interp_particle(register long long i, struct grid grd,
                                particles_pointers part, ids_pointers ids, grd_pointers grd_p)
{
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int (floor((part.x[i] - grd.xStart) * grd.invdx));
    iy = 2 + int (floor((part.y[i] - grd.yStart) * grd.invdy));
    iz = 2 + int (floor((part.z[i] - grd.zStart) * grd.invdz));

    // distances from node
    xi[0]   = part.x[i] - grd_p.XN_flat[get_idx(ix-1, iy, iz, grd.nyn, grd.nzn)];
    eta[0]  = part.y[i] - grd_p.YN_flat[get_idx(ix, iy-1, iz, grd.nyn, grd.nzn)];
    zeta[0] = part.z[i] - grd_p.ZN_flat[get_idx(ix, iy, iz-1, grd.nyn, grd.nzn)];
    xi[1]   = grd_p.XN_flat[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - part.x[i];
    eta[1]  = grd_p.YN_flat[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - part.y[i];
    zeta[1] = grd_p.ZN_flat[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - part.z[i];

    // calculate the weights for different nodes
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                weight[ii][jj][kk] = part.q[i] * xi[ii] * eta[jj] * zeta[kk] * grd.invVOL;

    //////////////////////////
    // add charge density
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                ids.rhon_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)] += weight[ii][jj][kk] * grd.invVOL;


    ////////////////////////////
    // add current density - Jx
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.u[i] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                ids.Jx_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)] += weight[ii][jj][kk] * grd.invVOL;


    ////////////////////////////
    // add current density - Jy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.v[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                ids.Jy_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)] += weight[ii][jj][kk] * grd.invVOL;



    ////////////////////////////
    // add current density - Jz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.w[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                ids.Jz_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)] += weight[ii][jj][kk] * grd.invVOL;


    ////////////////////////////
    // add pressure pxx
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.u[i] * part.u[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                ids.pxx_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)] += weight[ii][jj][kk] * grd.invVOL;


    ////////////////////////////
    // add pressure pxy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.u[i] * part.v[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                ids.pxy_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)] += weight[ii][jj][kk] * grd.invVOL;



    /////////////////////////////
    // add pressure pxz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.u[i] * part.w[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                ids.pxz_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)] += weight[ii][jj][kk] * grd.invVOL;


    /////////////////////////////
    // add pressure pyy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.v[i] * part.v[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                ids.pyy_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)] += weight[ii][jj][kk] * grd.invVOL;


    /////////////////////////////
    // add pressure pyz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.v[i] * part.w[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                ids.pyz_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)] += weight[ii][jj][kk] * grd.invVOL;


    /////////////////////////////
    // add pressure pzz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.w[i] * part.w[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                ids.pzz_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)] += weight[ii][jj][kk] * grd.invVOL;

}

/** Interpolation Particle --> Grid: This is for species */
// CPU Version
void h_interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    // Create argument structs
    particles_pointers p_p {
        part->x, part->y, part->z,
        part->u, part->v, part->w, part->q
    };
    ids_pointers i_p {
        ids->rhon_flat, ids->rhoc_flat,
        ids->Jx_flat, ids->Jy_flat, ids->Jz_flat,
        ids->pxx_flat, ids->pxy_flat, ids->pxz_flat,
        ids->pyy_flat, ids->pyz_flat, ids->pzz_flat
    };
    grd_pointers g_p {
        grd->XN_flat, grd->YN_flat, grd->ZN_flat
    };
    for (register long long i = 0; i < part->nop; i++) {
        h_interp_particle(i, *grd, p_p, i_p, g_p);
    }
}

