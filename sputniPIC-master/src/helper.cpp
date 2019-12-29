#include "helper.h"


void print(std::string str) {
    std::cout << str << std::endl;
}

/** 
 * This function allocates the GPU memory needed for the kernels in the mover_pc and interp2G functions. If the number of
 * particles is more than the maximum defined, it only allocates the maximum number of particles already defined,
 * and then operations are done in batches of particles. 
 */
void allocate_gpu_memory(struct particles* part, int grdSize, int fieldSize, 
                                particles_pointers* p_p, ids_pointers* i_p,
                                grd_pointers* g_p, field_pointers* f_p) 
{
    
    FPpart* part_copies[6]; FPinterp* part_copy_q;
    FPinterp* ids_copies[11];
    FPfield* grd_copies[3];
    FPfield* field_copies[6];
    
    long num_gpu_particles = part->npmax;
    if (part->npmax > MAX_GPU_PARTICLES) {
        num_gpu_particles = MAX_GPU_PARTICLES;
        std::cout << "In [allocate_interp_gpu_memory]: part->nop is greater than MAX_GPU_PARTICLES. "
                     "Allocating only up to MAX_GPU_PARTICLES particles..." << std::endl;
    }

    // Allocate GPU arrays 
    {
        cudaMalloc(&part_copy_q, num_gpu_particles * sizeof(FPinterp));

        for (int i = 0; i < 6; ++i)
            cudaMalloc(&part_copies[i], num_gpu_particles * sizeof(FPpart));

        for (int i = 0; i < 11; ++i)
            cudaMalloc(&ids_copies[i], grdSize * sizeof(FPinterp));

        for (int i = 0; i < 3; ++i)
            cudaMalloc(&grd_copies[i], grdSize * sizeof(FPfield));

        for (int i = 0; i < 6; i++)
            cudaMalloc(&field_copies[i], fieldSize * sizeof(FPfield));
    }

    // Put GPU array pointers into structs
    p_p->x = part_copies[0];
    p_p->y = part_copies[1];
    p_p->z = part_copies[2];
    p_p->u = part_copies[3];
    p_p->v = part_copies[4];
    p_p->w = part_copies[5];
    p_p->q = part_copy_q;

    i_p->rhon_flat = ids_copies[0];
    i_p->rhoc_flat = ids_copies[1];
    i_p->Jx_flat = ids_copies[2];
    i_p->Jy_flat = ids_copies[3];
    i_p->Jz_flat = ids_copies[4];
    i_p->pxx_flat = ids_copies[5];
    i_p->pxy_flat = ids_copies[6];
    i_p->pxz_flat = ids_copies[7];
    i_p->pyy_flat = ids_copies[8];
    i_p->pyz_flat = ids_copies[9];
    i_p->pzz_flat = ids_copies[10];

    g_p->XN_flat = grd_copies[0];
    g_p->YN_flat = grd_copies[1];
    g_p->ZN_flat = grd_copies[2];

    f_p->Ex_flat = field_copies[0];
    f_p->Ey_flat = field_copies[1];
    f_p->Ez_flat = field_copies[2];
    f_p->Bxn_flat = field_copies[3];
    f_p->Byn_flat = field_copies[4];
    f_p->Bzn_flat = field_copies[5];
}

/**
 * Copy the field and grid arrays that are constant throughout all mover_PC kernels. This
 * should be executed before any mover_PC kernels are launched. Note that grid pointers
 * must also have been copied over prior to calling interpP2G kernels as well. 
 */
void copy_mover_constants_to_GPU(struct EMfield* field, struct grid* grd, 
                                 field_pointers f_p, grd_pointers g_p, 
                                 int grdSize, int field_size)
{
    // copy field variables
    cudaMemcpy(f_p.Ex_flat, field->Ex_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(f_p.Ey_flat, field->Ey_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(f_p.Ez_flat, field->Ez_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(f_p.Bxn_flat, field->Bxn_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(f_p.Byn_flat, field->Byn_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(f_p.Bzn_flat, field->Bzn_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);

    // copy grid variables
    cudaMemcpy(g_p.XN_flat, grd->XN_flat, grdSize * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(g_p.YN_flat, grd->YN_flat, grdSize * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(g_p.ZN_flat, grd->ZN_flat, grdSize * sizeof(FPfield), cudaMemcpyHostToDevice);
}

/** 
 * This function copies the Particle data needed for running the kernel in the mover_pc function.
 * NOTE: to is exclusive => mini-batch size is (to - from) 
 */
void copy_particles(struct particles* part, particles_pointers p_p, PICMode mode, 
                       long from, long to, bool verbose) {
    // if batch_size is not -1, it means that mini-batching should be done
    long batch_size = to - from;
    if (verbose) std::cout << "In [copy_particles]: copying with batch size of " << batch_size << std::endl;

    // Copy CPU Particles arrays to GPU
    if (mode == CPU_TO_GPU_MOVER || mode == CPU_TO_GPU_INTERP) {
        cudaMemcpy(&p_p.x[from % MAX_GPU_PARTICLES], &part->x[from], batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
        cudaMemcpy(&p_p.y[from % MAX_GPU_PARTICLES], &part->y[from], batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
        cudaMemcpy(&p_p.z[from % MAX_GPU_PARTICLES], &part->z[from], batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
        cudaMemcpy(&p_p.u[from % MAX_GPU_PARTICLES], &part->u[from], batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
        cudaMemcpy(&p_p.v[from % MAX_GPU_PARTICLES], &part->v[from], batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
        cudaMemcpy(&p_p.w[from % MAX_GPU_PARTICLES], &part->w[from], batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
        
        if (mode == CPU_TO_GPU_INTERP) 
            cudaMemcpy(&p_p.q[from % MAX_GPU_PARTICLES], &part->q[from], batch_size * sizeof(FPinterp), cudaMemcpyHostToDevice);
        
        if (verbose) std::cout << "In [copy_particles]: batch copy to GPU done..." << std::endl;
    }
    // Copy GPU arrays back to CPU
    else if (mode == GPU_TO_CPU_MOVER) {
        cudaMemcpy(&part->x[from], &p_p.x[from % MAX_GPU_PARTICLES], batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
        cudaMemcpy(&part->y[from], &p_p.y[from % MAX_GPU_PARTICLES], batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
        cudaMemcpy(&part->z[from], &p_p.z[from % MAX_GPU_PARTICLES], batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
        cudaMemcpy(&part->u[from], &p_p.u[from % MAX_GPU_PARTICLES], batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
        cudaMemcpy(&part->v[from], &p_p.v[from % MAX_GPU_PARTICLES], batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
        cudaMemcpy(&part->w[from], &p_p.w[from % MAX_GPU_PARTICLES], batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
        if (verbose) std::cout << "In [copy_particles]: copy back to CPU done..." << std::endl;
    }

}

/**
 * Copy the initial interpolation densities for interpolation kernels. This
 * should be executed before any interpP2G kernels are launched.
 * Assumes that grd arrays have already been set by mover_PC, and does not copy them again.
 */
void copy_interp_initial_to_GPU(struct interpDensSpecies* ids, ids_pointers i_p,
                                int grdSize, int rhocSize)
{
    cudaMemcpy(i_p.rhon_flat, ids->rhon_flat, grdSize*sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(i_p.rhoc_flat, ids->rhoc_flat, rhocSize*sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(i_p.Jx_flat, ids->Jx_flat, grdSize*sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(i_p.Jy_flat, ids->Jy_flat, grdSize*sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(i_p.Jz_flat, ids->Jz_flat, grdSize*sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(i_p.pxx_flat, ids->pxx_flat, grdSize*sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(i_p.pxy_flat, ids->pxy_flat, grdSize*sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(i_p.pxz_flat, ids->pxz_flat, grdSize*sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(i_p.pyy_flat, ids->pyy_flat, grdSize*sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(i_p.pyz_flat, ids->pyz_flat, grdSize*sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(i_p.pzz_flat, ids->pzz_flat, grdSize*sizeof(FPinterp), cudaMemcpyHostToDevice);
}

/** 
 * This function copies from specified Particles from CPU to GPU. 
 */
void copy_interp_particles(struct particles* part, particles_pointers p_p, long from, long to, bool verbose) 
{
    long batch_size = to - from;
    if (verbose) std::cout << "In [copy_interp_arrays]: copying with batch size of " << batch_size << std::endl;
    cudaMemcpy(&p_p.x[from % MAX_GPU_PARTICLES], &part->x[from], batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(&p_p.y[from % MAX_GPU_PARTICLES], &part->y[from], batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(&p_p.z[from % MAX_GPU_PARTICLES], &part->z[from], batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(&p_p.u[from % MAX_GPU_PARTICLES], &part->u[from], batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(&p_p.v[from % MAX_GPU_PARTICLES], &part->v[from], batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(&p_p.w[from % MAX_GPU_PARTICLES], &part->w[from], batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(&p_p.q[from % MAX_GPU_PARTICLES], &part->q[from], batch_size * sizeof(FPinterp), cudaMemcpyHostToDevice);
    if (verbose) std::cout << "In [copy_interp_arrays]: copy to GPU: done." << std::endl;
}

/** 
 * This function copies the results of interpolation from GPU to CPU.
 */
void copy_interp_results(struct interpDensSpecies* ids, ids_pointers i_p, int grdSize, int rhocSize) 
{
    cudaMemcpy(ids->rhon_flat, i_p.rhon_flat, grdSize*sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->rhoc_flat, i_p.rhoc_flat, rhocSize*sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jx_flat, i_p.Jx_flat, grdSize*sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jy_flat, i_p.Jy_flat, grdSize*sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jz_flat, i_p.Jz_flat, grdSize*sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxx_flat, i_p.pxx_flat, grdSize*sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxy_flat, i_p.pxy_flat, grdSize*sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxz_flat, i_p.pxz_flat, grdSize*sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyy_flat, i_p.pyy_flat, grdSize*sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyz_flat, i_p.pyz_flat, grdSize*sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pzz_flat, i_p.pzz_flat, grdSize*sizeof(FPinterp), cudaMemcpyDeviceToHost);
}

/** 
 * This function frees all the memory allocated on the GPU for both kernels.
 */
void free_gpu_memory(particles_pointers* p_p, ids_pointers* i_p, grd_pointers* g_p, field_pointers* f_p) {
    cudaFree(p_p->x);
    cudaFree(p_p->y);
    cudaFree(p_p->z);
    cudaFree(p_p->u);
    cudaFree(p_p->v);
    cudaFree(p_p->w);
    cudaFree(p_p->q);

    cudaFree(i_p->rhon_flat);
    cudaFree(i_p->rhoc_flat);
    cudaFree(i_p->Jx_flat);
    cudaFree(i_p->Jy_flat);
    cudaFree(i_p->Jz_flat);
    cudaFree(i_p->pxx_flat);
    cudaFree(i_p->pxy_flat);
    cudaFree(i_p->pxz_flat);
    cudaFree(i_p->pyy_flat);
    cudaFree(i_p->pyz_flat);
    cudaFree(i_p->pzz_flat);

    cudaFree(g_p->XN_flat);
    cudaFree(g_p->YN_flat);
    cudaFree(g_p->ZN_flat);

    cudaFree(f_p->Ex_flat);
    cudaFree(f_p->Ey_flat);
    cudaFree(f_p->Ez_flat);
    cudaFree(f_p->Bxn_flat);
    cudaFree(f_p->Byn_flat);
    cudaFree(f_p->Bzn_flat);

    std::cout << "In [free_gpu_memory]: all GPU memory freed.." << std::endl;
}
