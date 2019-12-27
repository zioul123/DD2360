#include "helper.h"


void print(std::string str) {
    std::cout << str << std::endl;
}


void allocate_batch(FPpart*& batch_x, FPpart*& batch_y, FPpart*& batch_z,
                    FPpart*& batch_u, FPpart*& batch_v, FPpart*& batch_w,
                    FPpart*& batch_q, long batch_size, PICMode mode) {
    /** This function allocates auxiliary batch variables that contain the data of a batch of particles, used to
     * transfer batch data between CPU and GPU.
     * NOTE: depending on the 'mode' variable, batch_q will be ignored (if mode is "mover_PC") */

    batch_x = (FPpart*) malloc(batch_size * sizeof(FPpart));
    batch_y = (FPpart*) malloc(batch_size * sizeof(FPpart));
    batch_z = (FPpart*) malloc(batch_size * sizeof(FPpart));
    batch_u = (FPpart*) malloc(batch_size * sizeof(FPpart));
    batch_v = (FPpart*) malloc(batch_size * sizeof(FPpart));
    batch_w = (FPpart*) malloc(batch_size * sizeof(FPpart));

    if (mode == INTERP2G)  // batch_q only used for interp2G, otherwise is ignored
        batch_q = (FPpart*) malloc(batch_size * sizeof(FPpart));
}


void deallocate_batch(FPpart*& batch_x, FPpart*& batch_y, FPpart*& batch_z,
                      FPpart*& batch_u, FPpart*& batch_v, FPpart*& batch_w,
                      FPpart*& batch_q, PICMode mode) {
    /** Since the batch variables are temporary, they are immediately deallocated once the data is copied to the GPU
     * or copied to the original particles variable
     * NOTE: depending on the 'mode' variable, batch_q will be ignored (if mode is "mover_PC") */

    free(batch_x);
    free(batch_y);
    free(batch_z);
    free(batch_u);
    free(batch_v);
    free(batch_w);

    if (mode == INTERP2G)
        free(batch_q);
}


void batch_copy(FPpart*& batch_x, FPpart*& batch_y, FPpart*& batch_z, FPpart*& batch_u, FPpart*& batch_v,
                FPpart*& batch_w, FPpart*& batch_q, FPpart*& part_x, FPpart*& part_y, FPpart*& part_z,
                FPpart*& part_u, FPpart*& part_v, FPpart*& part_w, FPpart*& part_q, long from, long to,
                PICMode mode, PICMode direction) {

    /** This function copies the data of a batch either from particles to batch variables (to be copied to GPU then)
     * or from the batch variables (which contain the kernel results on the batch) back to the particles
     * NOTE: depending on the 'mode' variable, batch_q will be ignored (if mode is "mover_PC") */

    for (long i = from; i < to; i++) {  // could it be more efficient?
        // iter * MAX_GPU_PARTICLES ==> 0, iter * MAX_GPU_PARTICLES + 1 ==> 1, iter * MAX_GPU_PARTICLES + 2 ==> 2
        long relative_i = i % MAX_GPU_PARTICLES;  // preventing segmentation fault, relative index in the batch

        if (direction == PARTICLE_TO_BATCH) {
            batch_x[relative_i] = part_x[i];
            batch_y[relative_i] = part_y[i];
            batch_z[relative_i] = part_z[i];
            batch_u[relative_i] = part_u[i];
            batch_v[relative_i] = part_v[i];
            batch_w[relative_i] = part_w[i];

            if (mode == INTERP2G)  // part_q only for inter2G, otherwise ignored
                batch_q[relative_i] = part_q[i];
        }

        else {  // direction == "batch_to_particle"
            part_x[i] = batch_x[relative_i];
            part_y[i] = batch_y[relative_i];
            part_z[i] = batch_z[relative_i];
            part_u[i] = batch_u[relative_i];
            part_v[i] = batch_v[relative_i];
            part_w[i] = batch_w[relative_i];

            if (mode == INTERP2G)  // part_q only for inter2G, otherwise ignored
                 part_q[i] = batch_q[relative_i];
        }
    }
}

/** 
 * This function allocates the GPU memory needed for the kernels in the mover_pc and interp2G functions. If the number of
 * particles is more than the maximum defined, it only allocates the maximum number of particles already defined,
 * and then operations are done in batches of particles. 
 */
void allocate_gpu_memory(struct particles* part, int grdSize, int fieldSize, 
                                particles_pointers* p_p, ids_pointers* i_p,
                                grd_pointers* g_p, field_pointers* f_p) {
    
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

    // Allocate GPU arrays for interpP2G
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

    // Put GPU array pointers into structs for interpP2G
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


void copy_interp_arrays(struct particles* part, struct interpDensSpecies* ids, struct grid* grd,
                        particles_pointers p_p, ids_pointers i_p, grd_pointers g_p, int grdSize,
                        int rhocSize, PICMode mode, long from, long to, bool verbose) {
    /** This function copies from CPU to GPU the data needed for running the kernel in the interp2G function.
     * For more info see the copy_mover_arrays function. */

    // if batch_size is not -1, it means that mini-batching should be done
    long batch_size = -1;
    if (from != -1 && to != -1) {  // determine size of mini-batch, if from and to are specified
        batch_size = to - from;
        if (verbose) std::cout << "In [copy_interp_arrays]: copying with batch size of " << batch_size << std::endl;
    }

    // prepare mini-batch structures if we mini-batching should be performed
    FPpart* batch_x; FPpart* batch_y; FPpart* batch_z; FPpart* batch_u; FPpart* batch_v; FPpart* batch_w; FPpart* batch_q;
    if (batch_size != -1) {  // mini-batch copying
        allocate_batch(batch_x, batch_y, batch_z, batch_u, batch_v, batch_w,
                       batch_q, batch_size, INTERP2G);
        if (verbose) std::cout << "In [copy_interp_arrays]: batch variables created..." << std::endl;
    }


    // Copy CPU arrays to GPU
    if (mode == CPU_TO_GPU) {
        // mini-batching
        if (batch_size != -1){
            // copy from particles to batch variables
            batch_copy(batch_x, batch_y, batch_z, batch_u, batch_v, batch_w, batch_q,
                       part->x, part->y, part->z, part->u, part->v, part->w, part->q,
                          from, to, INTERP2G, PARTICLE_TO_BATCH);
            if (verbose) std::cout << "In [copy_interp_arrays]: copy from part to batch done." << std::endl;

            // copy from batch variables to GPU
            cudaMemcpy(p_p.x, batch_x, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_p.y, batch_y, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_p.z, batch_z, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_p.u, batch_u, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_p.v, batch_v, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_p.w, batch_w, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_p.q, batch_q, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);

            if (verbose) std::cout << "In [copy_interp_arrays]: copy to GPU: done." << std::endl;


            // deallocate the memory once the data is copied to the GPU memory
            deallocate_batch(batch_x, batch_y, batch_z, batch_u, batch_v, batch_w,
                             batch_q, INTERP2G);
            if (verbose) std::cout << "In [copy_interp_arrays]: batch variables freed..." << std::endl;
        }

        // copy all the particles at once
        else {
            cudaMemcpy(p_p.x, part->x, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_p.y, part->y, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_p.z, part->z, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_p.u, part->u, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_p.v, part->v, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_p.w, part->w, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);

            cudaMemcpy(p_p.q, part->q, part->npmax*sizeof(FPinterp), cudaMemcpyHostToDevice);
        }

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

        cudaMemcpy(g_p.XN_flat, grd->XN_flat, grdSize*sizeof(FPfield), cudaMemcpyHostToDevice);
        cudaMemcpy(g_p.YN_flat, grd->YN_flat, grdSize*sizeof(FPfield), cudaMemcpyHostToDevice);
        cudaMemcpy(g_p.ZN_flat, grd->ZN_flat, grdSize*sizeof(FPfield), cudaMemcpyHostToDevice);
    }

    // Copy GPU arrays back to CPU
    else {
        // mini-batching
        if (batch_size != -1) {
            cudaMemcpy(batch_x, p_p.x, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(batch_y, p_p.y, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(batch_z, p_p.z, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(batch_u, p_p.u, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(batch_v, p_p.v, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(batch_w, p_p.w, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);

            cudaMemcpy(batch_q, p_p.q, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
            if (verbose) std::cout << "In [copy_interp_arrays]: copy from GPU to batch variables: done." << std::endl;

            batch_copy(batch_x, batch_y, batch_z, batch_u, batch_v, batch_w, batch_q,
                       part->x, part->y, part->z, part->u, part->v, part->w, part->q,
                          from, to, INTERP2G, BATCH_TO_PARTICLE);

            if (verbose) std::cout << "In [copy_interp_arrays]: copy from batch to particles: done." << std::endl;

            // deallocate the memory once the data is copied to the original particles
            deallocate_batch(batch_x, batch_y, batch_z, batch_u, batch_v, batch_w,
                             batch_q, INTERP2G);
            if (verbose) std::cout << "In [copy_interp_arrays]: batch variables freed: done." << std::endl;
        }

        // all particles at once
        else {
            cudaMemcpy(part->x, p_p.x, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(part->y, p_p.y, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(part->z, p_p.z, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(part->u, p_p.u, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(part->v, p_p.v, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(part->w, p_p.w, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);

            cudaMemcpy(part->q, p_p.q, part->npmax*sizeof(FPinterp), cudaMemcpyDeviceToHost);
        }

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

        cudaMemcpy(grd->XN_flat, g_p.XN_flat, grdSize*sizeof(FPfield), cudaMemcpyDeviceToHost);
        cudaMemcpy(grd->YN_flat, g_p.YN_flat, grdSize*sizeof(FPfield), cudaMemcpyDeviceToHost);
        cudaMemcpy(grd->ZN_flat, g_p.ZN_flat, grdSize*sizeof(FPfield), cudaMemcpyDeviceToHost);
    }
}

void copy_mover_arrays(struct particles* part, struct EMfield* field, struct grid* grd, particles_pointers p_info,
                       field_pointers f_p, grd_pointers g_p, int grdSize, int field_size,
                       PICMode mode, long from, long to, bool verbose) {
    /** This function copies from CPU to GPU the data needed for running the kernel in the interp2G function
     * long 'from' and long 'to' (if specified) determine which elements of the particle arrays should be copied to GPU.
     * If not specified (default), all the particles will be copied.
     * NOTE: to is exclusive => mini-batch size is (to - from) */

    // if batch_size is not -1, it means that mini-batching should be done
    long batch_size = -1;
    if (from != -1 && to != -1) {  // determine size of mini-batch, if from and to are specified
        batch_size = to - from;
        if (verbose) std::cout << "In [copy_mover_arrays]: copying with batch size of " << batch_size << std::endl;
    }

    // prepare mini-batch structures if we mini-batching should be performed
    FPpart* batch_x; FPpart* batch_y; FPpart* batch_z; FPpart* batch_u; FPpart* batch_v; FPpart* batch_w; FPpart* batch_q;
    if (batch_size != -1) {  // mini-batch copying
        allocate_batch(batch_x, batch_y, batch_z, batch_u, batch_v, batch_w,
                       batch_q, batch_size, MOVER_PC);  // by specifying "mover_PC", batch_q will be ignored

        if (verbose) std::cout << "In [copy_mover_arrays]: batch variables created..." << std::endl;
    }

    // Copy CPU arrays to GPU
    if (mode == CPU_TO_GPU) {
        // copy the mini-batch
        if (batch_size != -1) {
            // copy from particles to batch variables
            batch_copy(batch_x, batch_y, batch_z, batch_u, batch_v, batch_w, batch_q,
                       part->x, part->y, part->z, part->u, part->v, part->w, part->q,
                       from, to, MOVER_PC, PARTICLE_TO_BATCH);

            if (verbose) std::cout << "In [copy_mover_arrays]: copy from part to batch done." << std::endl;

            // copy from batch variables to GPU
            cudaMemcpy(p_info.x, batch_x, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.y, batch_y, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.z, batch_z, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.u, batch_u, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.v, batch_v, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.w, batch_w, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);

            if (verbose) std::cout << "In [copy_mover_arrays]: copy to GPU done..." << std::endl;

            // deallocate the memory once the data is copied to the GPU memory
            deallocate_batch(batch_x, batch_y, batch_z, batch_u, batch_v, batch_w,
                             batch_q, MOVER_PC);
            if (verbose) std::cout << "In [copy_mover_arrays]: batch variables freed..." << std::endl;
        }

        // copy all the particles at once
        else {
            cudaMemcpy(p_info.x, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.y, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.z, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.u, part->u, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.v, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.w, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
        }

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

    // Copy GPU arrays back to CPU
    else {
        // copy the mini-batch
        if (batch_size != -1) {
            cudaMemcpy(batch_x, p_info.x, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(batch_y, p_info.y, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(batch_z, p_info.z, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(batch_u, p_info.u, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(batch_v, p_info.v, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(batch_w, p_info.w, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);

            if (verbose) std::cout << "In [copy_mover_arrays]: copy back to CPU done..." << std::endl;

            batch_copy(batch_x, batch_y, batch_z, batch_u, batch_v, batch_w, batch_q,
                       part->x, part->y, part->z, part->u, part->v, part->w, part->q,
                          from, to, MOVER_PC, BATCH_TO_PARTICLE);

            // deallocate the memory once the data is copied to the original particles
            deallocate_batch(batch_x, batch_y, batch_z, batch_u, batch_v, batch_w,
                             batch_q, MOVER_PC);
            if (verbose) std::cout << "In [copy_mover_arrays]: batch variables freed..." << std::endl;
        }

        // copy all particles at once
        else {
            cudaMemcpy(part->x, p_info.x, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(part->y, p_info.y, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(part->z, p_info.z, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(part->u, p_info.u, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(part->v, p_info.v, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(part->w, p_info.w, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
        }

        // copy field variables
        cudaMemcpy(field->Ex_flat, f_p.Ex_flat, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
        cudaMemcpy(field->Ey_flat, f_p.Ey_flat, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
        cudaMemcpy(field->Ez_flat, f_p.Ez_flat, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
        cudaMemcpy(field->Bxn_flat, f_p.Bxn_flat, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
        cudaMemcpy(field->Byn_flat, f_p.Byn_flat, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
        cudaMemcpy(field->Bzn_flat, f_p.Bzn_flat, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);

        // copy grid variables
        cudaMemcpy(grd->XN_flat, g_p.XN_flat, grdSize * sizeof(FPfield), cudaMemcpyDeviceToHost);
        cudaMemcpy(grd->YN_flat, g_p.YN_flat, grdSize * sizeof(FPfield), cudaMemcpyDeviceToHost);
        cudaMemcpy(grd->ZN_flat, g_p.ZN_flat, grdSize * sizeof(FPfield), cudaMemcpyDeviceToHost);
    }
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
