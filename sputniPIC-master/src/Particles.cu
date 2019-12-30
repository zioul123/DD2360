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


/** allocate particle arrays. If streaming is enabled, use cudaHostAlloc */
void particle_allocate(struct parameters* param, struct particles* part, int is, bool enableStreaming)
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
    if (!enableStreaming)
    {
        part->x = new FPpart[npmax];
        part->y = new FPpart[npmax];
        part->z = new FPpart[npmax];
        // allocate velocity
        part->u = new FPpart[npmax];
        part->v = new FPpart[npmax];
        part->w = new FPpart[npmax];
        // allocate charge = q * statistical weight
        part->q = new FPinterp[npmax];
        std::cout << "In [particle_allocate]: Allocation of CPU (non-pinned) memory for species " << is << " done" << std::endl;
    }
    else if (enableStreaming)
    {
        cudaHostAlloc(&part->x, sizeof(FPpart) * npmax, cudaHostAllocDefault);
        cudaHostAlloc(&part->y, sizeof(FPpart) * npmax, cudaHostAllocDefault);
        cudaHostAlloc(&part->z, sizeof(FPpart) * npmax, cudaHostAllocDefault);
        cudaHostAlloc(&part->u, sizeof(FPpart) * npmax, cudaHostAllocDefault);
        cudaHostAlloc(&part->v, sizeof(FPpart) * npmax, cudaHostAllocDefault);
        cudaHostAlloc(&part->w, sizeof(FPpart) * npmax, cudaHostAllocDefault);
        cudaHostAlloc(&part->q, sizeof(FPinterp) * npmax, cudaHostAllocDefault);
        std::cout << "In [particle_allocate]: Allocation of CPU (pinned) memory for species " << is << " done" << std::endl;
    }
    
}


/** deallocate. If streaming was enabled, use cudaFreeHost instead of delete[]. */
void particle_deallocate(struct particles* part, bool enableStreaming)
{
    // deallocate particle variables
    if (!enableStreaming)
    {
        delete[] part->x;
        delete[] part->y;
        delete[] part->z;
        delete[] part->u;
        delete[] part->v;
        delete[] part->w;
        delete[] part->q; 
        std::cout << "In [particle_deallocate]: Dellocation of CPU memory (non-pinned) done" << std::endl;
    }
    else if (enableStreaming)
    {
        cudaFreeHost(part->x);
        cudaFreeHost(part->y);
        cudaFreeHost(part->z);
        cudaFreeHost(part->u);
        cudaFreeHost(part->v);
        cudaFreeHost(part->w);
        cudaFreeHost(part->q); 
        std::cout << "In [particle_deallocate]: Dellocation of CPU memory (pinned) done" << std::endl;
    }
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
             particles_pointers p_p, field_pointers f_p, grd_pointers g_p, int grdSize, int field_size, 
             cudaStream_t* streams, bool enableStreaming, int streamSize)
{
    // print species and subcycling
    std::cout << std::endl << "***  In [mover_PC]: MOVER with SUBCYCLING "<< param->n_sub_cycles
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

        if (!enableStreaming)
        {
            // Copy particles in batch to GPU (part in CPU to p_p on GPU) without streaming
            copy_particles(part, p_p, CPU_TO_GPU_MOVER, batch_start, batch_end);
            // Launch the kernel to perform on the batch
            g_move_particle<<<(batch_size+TPB-1)/TPB, TPB>>>(0, batch_size, part->n_sub_cycles, part->NiterMover, 
                                                             *grd, *param, dt_inf, p_p, f_p, g_p);
            // Copy moved particles back (p_p in GPU back to part in CPU) without streaming
            copy_particles(part, p_p, GPU_TO_CPU_MOVER, batch_start, batch_end);
        }
        else if (enableStreaming)
        {
            // If batch_size <= streamSize, n_streams = 1, and whole batch is done in one stream
            int n_streams = (batch_size + streamSize - 1) / streamSize;
            for (int stream_no = 0; stream_no < n_streams; stream_no++) 
            {
                // Compute stream size/bounds RELATIVE TO BATCH_START. In other words, to access the
                // CPU array, the starting element is batch_start + stream_start. GPU accesses are done
                // with CPU index % GPU_MAX_PARTICLES, so there is no need to convert the indices back.
                long stream_start = stream_no * streamSize;
                long stream_end = std::min(stream_start + streamSize, batch_size);  // max is batch_size
                long stream_size = stream_end - stream_start;

                // Copy particles in stream to GPU (part in CPU to p_p on GPU) with streaming
                copy_particles_async(part, p_p, CPU_TO_GPU_MOVER, 
                                     batch_start + stream_start, batch_start + stream_end, 
                                     streams[stream_no]);
                // Launch the kernel to perform on the stream
                g_move_particle<<<(stream_size+TPB-1)/TPB, TPB, 0, streams[stream_no]>>>(
                        stream_start, stream_size, part->n_sub_cycles, part->NiterMover, 
                        *grd, *param, dt_inf, p_p, f_p, g_p);
                // Copy moved particles back (p_p in GPU back to part in CPU) with streaming
                copy_particles_async(part, p_p, GPU_TO_CPU_MOVER, 
                                     batch_start + stream_start, batch_start + stream_end,
                                     streams[stream_no]);
            }
        }
        cudaDeviceSynchronize();
        std::cout << "====== In [mover_PC]: batch " << (batch_no + 1) << " of " << n_batches 
                  << (enableStreaming ? " (with streaming)" : " (without streaming)") 
                  << ": done." << std::endl;
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
               particles_pointers p_p, ids_pointers i_p, grd_pointers g_p, int grdSize, int rhocSize,
               cudaStream_t* streams, bool enableStreaming, int streamSize)
{
    // Print species
    std::cout << std::endl << "***  In [interpP2G]: Interpolating "
              << " - species " << part->species_ID << " ***" << std::endl;

    /*
     * The following steps are taken:
     * 0. Assume grd is copied to GPU already
     * 1. Copy zeroed ids to GPU as initialization
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

        if (!enableStreaming) 
        {
            // Copy particles in batch to GPU (part in CPU to p_p on GPU) without streaming
            copy_particles(part, p_p, CPU_TO_GPU_INTERP, batch_start, batch_end);
            // Launch the kernel to perform on the batch
            g_interp_particle<<<(batch_size+TPB-1)/TPB, TPB>>>(0, batch_size, *grd, p_p, i_p, g_p);
        }
        else if (enableStreaming) 
        {
            // If batch_size <= streamSize, n_streams = 1, and whole batch is done in one stream
            int n_streams = (batch_size + streamSize - 1) / streamSize;
            for (int stream_no = 0; stream_no < n_streams; stream_no++) 
            {
                // Compute stream size/bounds RELATIVE TO BATCH_START. In other words, to access the
                // CPU array, the starting element is batch_start + stream_start. GPU accesses are done
                // with CPU index % GPU_MAX_PARTICLES, so there is no need to convert the indices back.
                long stream_start = stream_no * streamSize;
                long stream_end = std::min(stream_start + streamSize, batch_size);  // max is batch_size
                long stream_size = stream_end - stream_start;

                // Copy particles in stream to GPU (part in CPU to p_p on GPU) with streaming
                copy_particles_async(part, p_p, CPU_TO_GPU_INTERP, 
                                     batch_start + stream_start, batch_start + stream_end,
                                     streams[stream_no]);
                // Launch the kernel to perform on the stream
                g_interp_particle<<<(stream_size+TPB-1)/TPB, TPB, 0, streams[stream_no]>>>(
                        stream_start, stream_size, *grd, p_p, i_p, g_p);
            }
        }
        cudaDeviceSynchronize();
        std::cout << "====== In [interpP2G]: batch " << (batch_no + 1) << " of " << n_batches 
                  << (enableStreaming ? " (with streaming)" : " (without streaming)") 
                  << ": done." << std::endl;
    }
    // Copy results back (i_p in GPU back to ids in CPU).
    copy_interp_results(ids, i_p, grdSize, rhocSize);
}

/** Mover and Interp. kernels combined into one. */
__global__ void g_combined_kernel(
        int stream_offset, int nop, int n_sub_cycles, int part_NiterMover, struct grid grd,
        struct parameters param, const dt_info dt_inf,
        particles_pointers part, const field_pointers field, ids_pointers ids, const grd_pointers grd_p) 
{
    // getting thread ID
    const int i = blockIdx.x * blockDim.x + threadIdx.x + stream_offset;
    if (i - stream_offset >= nop) return;

    // ------------------------------------ //
    // BEGINNING OF ORIGINALLY-MOVER KERNEL //
    // ------------------------------------ //

    // auxiliary variables
    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPpart temp[2][2][2];
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

    // ------------------------------------- //
    // BEGINNING OF ORIGINALLY-INTERP KERNEL //
    // ------------------------------------- //
    
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

void combinedMoveInterp(struct particles* part, struct EMfield* field, struct grid* grd, 
             struct interpDensSpecies* ids, struct parameters* param, 
             particles_pointers p_p, field_pointers f_p, grd_pointers g_p, ids_pointers i_p, 
             int grdSize, int field_size, int rhocSize,
             cudaStream_t* streams, bool enableStreaming, int streamSize) 
{
    // print species and subcycling
    std::cout << std::endl << "***  In [combinedMoveInterp]: Moving with SUBCYCLING "<< param->n_sub_cycles
              << " and interpolating - species " << part->species_ID << " ***" << std::endl;

    // "global" environment variables
    FPpart dt_sub_cycling = (FPpart) param->dt / ((double) part->n_sub_cycles);
    FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;
    const dt_info dt_inf { dt_sub_cycling, dto2, qomdt2 };

    /*
     * The following steps are taken:
     * 1. Copy grd, field, and zeroed ids to GPU as initialization
     * 2. For each batch/full batch:
     *     1. Copy relevant Particles to GPU
     *     2. Launch kernels that modify Particles in GPU memory
     *     3. Launch kernels that modify ids in GPU memory
     * 3. Copy ids back to CPU memory
     */

    // Copy grid/field constants to GPU
    copy_mover_constants_to_GPU(field, grd, f_p, g_p, grdSize, field_size);
    // Copy initial zeroed IDS
    copy_interp_initial_to_GPU(ids, i_p, grdSize, rhocSize);

    // If npmax <= MAX_GPU_PARTICLES, n_batches = 1, and all movement is done in one batch
    int n_batches = (part->npmax + MAX_GPU_PARTICLES - 1) / MAX_GPU_PARTICLES;
    for (int batch_no = 0; batch_no < n_batches; batch_no++) 
    {
        // Compute batch size/bounds
        long batch_start = batch_no * MAX_GPU_PARTICLES;
        long batch_end = std::min(batch_start + MAX_GPU_PARTICLES, part->npmax);  // max is part->npmax
        long batch_size = batch_end - batch_start;

        if (!enableStreaming)
        {
            // Copy particles in batch to GPU (part in CPU to p_p on GPU) without streaming
            copy_particles(part, p_p, CPU_TO_GPU_INTERP, batch_start, batch_end);
            // Launch the movement kernel to perform on the batch
            g_combined_kernel<<<(batch_size+TPB-1)/TPB, TPB>>>(0, batch_size, part->n_sub_cycles, part->NiterMover, 
                                                               *grd, *param, dt_inf, p_p, f_p, i_p, g_p);
            // Copy moved particles back (p_p in GPU back to part in CPU) without streaming
            copy_particles(part, p_p, GPU_TO_CPU_MOVER, batch_start, batch_end);
        }
        else if (enableStreaming)
        {
            // If batch_size <= streamSize, n_streams = 1, and whole batch is done in one stream
            int n_streams = (batch_size + streamSize - 1) / streamSize;
            for (int stream_no = 0; stream_no < n_streams; stream_no++) 
            {
                // Compute stream size/bounds RELATIVE TO BATCH_START. In other words, to access the
                // CPU array, the starting element is batch_start + stream_start. GPU accesses are done
                // with CPU index % GPU_MAX_PARTICLES, so there is no need to convert the indices back.
                long stream_start = stream_no * streamSize;
                long stream_end = std::min(stream_start + streamSize, batch_size);  // max is batch_size
                long stream_size = stream_end - stream_start;

                // Copy particles in stream to GPU (part in CPU to p_p on GPU) with streaming
                copy_particles_async(part, p_p, CPU_TO_GPU_INTERP, 
                                     batch_start + stream_start, batch_start + stream_end, 
                                     streams[stream_no]);
                // Launch the kernel to perform on the stream
                g_combined_kernel<<<(stream_size+TPB-1)/TPB, TPB, 0, streams[stream_no]>>>(
                        stream_start, stream_size, part->n_sub_cycles, part->NiterMover, 
                        *grd, *param, dt_inf, p_p, f_p, i_p, g_p);
                // Copy moved particles back (p_p in GPU back to part in CPU) with streaming
                copy_particles_async(part, p_p, GPU_TO_CPU_MOVER, 
                                     batch_start + stream_start, batch_start + stream_end,
                                     streams[stream_no]);
            }
        }
        cudaDeviceSynchronize();
        std::cout << "====== In [combinedMoveInterp]: batch " << (batch_no + 1) << " of " << n_batches 
                  << (enableStreaming ? " (with streaming)" : " (without streaming)") 
                  << ": done." << std::endl;
    }

    // Copy results back (i_p in GPU back to ids in CPU).
    copy_interp_results(ids, i_p, grdSize, rhocSize);
}

/* -------------------- *
 * CPU Serial Versions  *
 * -------------------- */
/** particle mover */
int h_mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "*** In [h_mover_PC]: MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
    
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover


/** Interpolation Particle --> Grid: This is for species */
void h_interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    // Print species
    std::cout << std::endl << "***  In [h_interpP2G]: Interpolating "
              << " - species " << part->species_ID << " ***" << std::endl;
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}

