#include "helper.h"


void print(std::string str) {
    std::cout << str << std::endl;
}


void allocate_interp_gpu_memory(struct particles* part, int grdSize, particles_pointers* p_p, ids_pointers* i_p,
                                grd_pointers* g_p) {
    FPpart* part_copies[6];
    FPinterp* part_copy_q;
    FPinterp* ids_copies[11];
    FPfield* grd_copies[3];

    // Allocate GPU arrays for interpP2G
    {
        cudaMalloc(&part_copy_q, part->npmax*sizeof(FPinterp));
        for (int i = 0; i < 6; ++i)
            cudaMalloc(&part_copies[i], part->npmax*sizeof(FPpart));
        for (int i = 0; i < 11; ++i)
            cudaMalloc(&ids_copies[i], grdSize*sizeof(FPinterp));
        for (int i = 0; i < 3; ++i)
            cudaMalloc(&grd_copies[i], grdSize*sizeof(FPfield));
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
}


void copy_interp_arrays(struct particles* part, struct interpDensSpecies* ids, struct grid* grd,
                        particles_pointers p_p, ids_pointers i_p, grd_pointers g_p, int grdSize,
                        int rhocSize, std::string mode) {

    if (mode == "cpu_to_gpu") {
        // std::cout << "In [copy_interp_arrays]: copying arrays to GPU" << std::endl;
        // Copy CPU arrays to GPU
        cudaMemcpy(p_p.x, part->x, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
        cudaMemcpy(p_p.y, part->y, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
        cudaMemcpy(p_p.z, part->z, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
        cudaMemcpy(p_p.u, part->u, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
        cudaMemcpy(p_p.v, part->v, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
        cudaMemcpy(p_p.w, part->w, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);

        cudaMemcpy(p_p.q, part->q, part->npmax*sizeof(FPinterp), cudaMemcpyHostToDevice);

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

    else {
        // Copy GPU arrays back to CPU
        // std::cout << "In [copy_interp_arrays]: copying arrays back to CPU" << std::endl;
        cudaMemcpy(part->x, p_p.x, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
        cudaMemcpy(part->y, p_p.y, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
        cudaMemcpy(part->z, p_p.z, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
        cudaMemcpy(part->u, p_p.u, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
        cudaMemcpy(part->v, p_p.v, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
        cudaMemcpy(part->w, p_p.w, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);

        cudaMemcpy(part->q, p_p.q, part->npmax*sizeof(FPinterp), cudaMemcpyDeviceToHost);

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


void allocate_mover_gpu_memory(struct particles* part, int grdSize, int field_size, particle_info* p_info,
                               field_pointers* f_pointers, grd_pointers* g_pointers) {
    // Declare GPU copies of arrays for mover_PC
    FPpart* part_info_copies[6];
    FPfield* f_pointer_copies[6];
    FPfield* g_pointer_copies[3];

    long num_gpu_particles = part->npmax;
    if (part->npmax > MAX_GPU_PARTICLES) {
        num_gpu_particles = MAX_GPU_PARTICLES;
        std::cout << "In [allocate_mover_gpu_memory]: part->nop is greater than MAX_GPU_PARTICLES. "
                     "Allocating only up to MAX_GPU_PARTICLES particles..." << std::endl;
        // getchar();
    }

    // Allocate GPU arrays for mover_PC
    {
        for (int i = 0; i < 6; i++)
            cudaMalloc(&part_info_copies[i], num_gpu_particles * sizeof(FPpart));

        for (int i = 0; i < 6; i++)
            cudaMalloc(&f_pointer_copies[i], field_size * sizeof(FPfield));

        for (int i = 0; i < 3; i++)
            cudaMalloc(&g_pointer_copies[i], grdSize * sizeof(FPfield));
    }

    // Put GPU array pointers into structs for mover_PC
    p_info->x = part_info_copies[0];
    p_info->y = part_info_copies[1];
    p_info->z = part_info_copies[2];
    p_info->u = part_info_copies[3];
    p_info->v = part_info_copies[4];
    p_info->w = part_info_copies[5];

    f_pointers->Ex_flat = f_pointer_copies[0];
    f_pointers->Ey_flat = f_pointer_copies[1];
    f_pointers->Ez_flat = f_pointer_copies[2];
    f_pointers->Bxn_flat = f_pointer_copies[3];
    f_pointers->Byn_flat = f_pointer_copies[4];
    f_pointers->Bzn_flat = f_pointer_copies[5];

    g_pointers->XN_flat = g_pointer_copies[0];
    g_pointers->YN_flat = g_pointer_copies[1];
    g_pointers->ZN_flat = g_pointer_copies[2];
}


void copy_mover_arrays(struct particles* part, struct EMfield* field, struct grid* grd, particle_info p_info,
                       field_pointers f_pointers, grd_pointers g_pointers, int grdSize, int field_size,
                       std::string mode, long from, long to) {
    /** long from and long to (if specified) determine which elements of the particle arrays should be copied to GPU.
     * If not specified (default), all the particles will be copied.
     * NOTE: to is exclusive => mini-batch size is (to - from) */
    long batch_size = -1;
    if (from != -1 && to != -1) {  // determine size of mini-batch, if from and to are specified
        batch_size = to - from;
        std::cout << "In [copy_mover_arrays]: copying with batch size of " << batch_size << std::endl;
        // getchar();
    }


    // prepare mini-batch structures if we mini-batching should be performed
    FPpart* x_tmp; FPpart* y_tmp; FPpart* z_tmp; FPpart* u_tmp; FPpart* v_tmp; FPpart* w_tmp;
    if (batch_size != -1) {  // mini-batch copying
        x_tmp = (FPpart*) malloc(batch_size * sizeof(FPpart));
        y_tmp = (FPpart*) malloc(batch_size * sizeof(FPpart));
        z_tmp = (FPpart*) malloc(batch_size * sizeof(FPpart));
        u_tmp = (FPpart*) malloc(batch_size * sizeof(FPpart));
        v_tmp = (FPpart*) malloc(batch_size * sizeof(FPpart));
        w_tmp = (FPpart*) malloc(batch_size * sizeof(FPpart));

        std::cout << "In [copy_mover_arrays]: temp variables created..." << std::endl;
        // getchar();

    }

    if (mode == "cpu_to_gpu") {
        // Copy CPU arrays to GPU
        // std::cout << "In [copy_mover_arrays]: copying arrays to GPU" << std::endl;

        if (batch_size != -1) {  // copy the mini-batch
            // copy the corresponding mini-batch of particles to tmp variables
            for (long i = from; i < to; i++) {  // could it be more efficient?
                // iter * MAX_GPU_PARTICLES ==> 0, iter * MAX_GPU_PARTICLES + 1 ==> 1, iter * MAX_GPU_PARTICLES + 2 ==> 2
                long relative_i = i % MAX_GPU_PARTICLES;  // preventing segmentation fault, relative index in the batch
                // std::cout << relative_i << std::endl;
                x_tmp[relative_i] = part->x[i];
                y_tmp[relative_i] = part->y[i];
                z_tmp[relative_i] = part->z[i];
                u_tmp[relative_i] = part->u[i];
                v_tmp[relative_i] = part->v[i];
                w_tmp[relative_i] = part->w[i];
            }
            std::cout << "In [copy_mover_arrays]: copy from part to temp done." << std::endl;
            /*std::cout << "\n\n x_tmp" << std::endl;
            for (int ii = 0; ii < 10; ii++) {
                std::cout << x_tmp[ii] << std::endl;
            }

            // getchar();*/

            cudaMemcpy(p_info.x, x_tmp, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.y, y_tmp, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.z, z_tmp, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.u, u_tmp, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.v, v_tmp, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.w, w_tmp, batch_size * sizeof(FPpart), cudaMemcpyHostToDevice);

            std::cout << "In [copy_mover_arrays]: copy to GPU done..." << std::endl;
            //getchar();
        }

        else {  // copy all the particles
            cudaMemcpy(p_info.x, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.y, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.z, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.u, part->u, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.v, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
            cudaMemcpy(p_info.w, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
        }

        cudaMemcpy(f_pointers.Ex_flat, field->Ex_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
        cudaMemcpy(f_pointers.Ey_flat, field->Ey_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
        cudaMemcpy(f_pointers.Ez_flat, field->Ez_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
        cudaMemcpy(f_pointers.Bxn_flat, field->Bxn_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
        cudaMemcpy(f_pointers.Byn_flat, field->Byn_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
        cudaMemcpy(f_pointers.Bzn_flat, field->Bzn_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);

        cudaMemcpy(g_pointers.XN_flat, grd->XN_flat, grdSize * sizeof(FPfield), cudaMemcpyHostToDevice);
        cudaMemcpy(g_pointers.YN_flat, grd->YN_flat, grdSize * sizeof(FPfield), cudaMemcpyHostToDevice);
        cudaMemcpy(g_pointers.ZN_flat, grd->ZN_flat, grdSize * sizeof(FPfield), cudaMemcpyHostToDevice);
    }

    // Copy GPU arrays back to CPU
    else {
        if (batch_size != -1) {  // copy the mini-batch
            cudaMemcpy(x_tmp, p_info.x, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(y_tmp, p_info.y, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(z_tmp, p_info.z, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(u_tmp, p_info.u, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(v_tmp, p_info.v, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(w_tmp, p_info.w, batch_size * sizeof(FPpart), cudaMemcpyDeviceToHost);

            std::cout << "In [copy_mover_arrays]: copy back to CPU done..." << std::endl;
            /*std::cout << "\n\n x_tmp AFTER GPU" << std::endl;
            for (int ii = 0; ii < 10; ii++) {
                std::cout << x_tmp[ii] << std::endl;
            }*/

            //getchar();

            for (long i = from; i < to; i++) {  // could it be more efficient?
                // iter * MAX_GPU_PARTICLES ==> 0, iter * MAX_GPU_PARTICLES + 1 ==> 1, iter * MAX_GPU_PARTICLES + 2 ==> 2
                long relative_i = i % MAX_GPU_PARTICLES;  // preventing segmentation fault, relative index in the batch
                part->x[i] = x_tmp[relative_i];
                part->y[i] = y_tmp[relative_i];
                part->z[i] = z_tmp[relative_i];
                part->u[i] = u_tmp[relative_i];
                part->v[i] = v_tmp[relative_i];
                part->w[i] = w_tmp[relative_i];

                // p_info.x[i] =1;
                /*p_info.x[i] = part->x[i];
                p_info.y[i] = part->y[i];
                p_info.z[i] = part->z[i];
                p_info.u[i] = part->u[i];
                p_info.v[i] = part->v[i];
                p_info.w[i] = part->w[i];*/

            }
            std::cout << "In [copy_mover_arrays]: copy from temp to part done." << std::endl;
            //getchar();
        }
        else {
            // std::cout << "In [copy_mover_arrays]: copying arrays back to CPU" << std::endl;
            cudaMemcpy(part->x, p_info.x, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(part->y, p_info.y, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(part->z, p_info.z, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(part->u, p_info.u, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(part->v, p_info.v, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
            cudaMemcpy(part->w, p_info.w, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
        }

        cudaMemcpy(field->Ex_flat, f_pointers.Ex_flat, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
        cudaMemcpy(field->Ey_flat, f_pointers.Ey_flat, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
        cudaMemcpy(field->Ez_flat, f_pointers.Ez_flat, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
        cudaMemcpy(field->Bxn_flat, f_pointers.Bxn_flat, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
        cudaMemcpy(field->Byn_flat, f_pointers.Byn_flat, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
        cudaMemcpy(field->Bzn_flat, f_pointers.Bzn_flat, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);

        cudaMemcpy(grd->XN_flat, g_pointers.XN_flat, grdSize * sizeof(FPfield), cudaMemcpyDeviceToHost);
        cudaMemcpy(grd->YN_flat, g_pointers.YN_flat, grdSize * sizeof(FPfield), cudaMemcpyDeviceToHost);
        cudaMemcpy(grd->ZN_flat, g_pointers.ZN_flat, grdSize * sizeof(FPfield), cudaMemcpyDeviceToHost);
    }
}


void free_gpu_memory(particles_pointers* p_p, ids_pointers* i_p, grd_pointers* g_p,
                     particle_info* p_info, field_pointers* f_pointers, grd_pointers* g_pointers) {

    cudaFree(p_p->x);
    cudaFree(p_p->y);
    cudaFree(p_p->z);
    cudaFree(p_p->u);
    cudaFree(p_p->v);
    cudaFree(p_p->w);
    cudaFree(p_p->q);
    cudaFree(p_p);  // freeing the top-level pointer

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
    cudaFree(i_p);

    cudaFree(g_p->XN_flat);
    cudaFree(g_p->YN_flat);
    cudaFree(g_p->ZN_flat);
    cudaFree(g_p);

    cudaFree(p_info->x);
    cudaFree(p_info->y);
    cudaFree(p_info->z);
    cudaFree(p_info->u);
    cudaFree(p_info->v);
    cudaFree(p_info->w);
    cudaFree(p_info);

    cudaFree(f_pointers->Ex_flat);
    cudaFree(f_pointers->Ey_flat);
    cudaFree(f_pointers->Ez_flat);
    cudaFree(f_pointers->Bxn_flat);
    cudaFree(f_pointers->Byn_flat);
    cudaFree(f_pointers->Bzn_flat);
    cudaFree(f_pointers);

    cudaFree(g_pointers->XN_flat);
    cudaFree(g_pointers->YN_flat);
    cudaFree(g_pointers->ZN_flat);
    cudaFree(g_pointers);

    std::cout << "In [free_gpu_memory]: all GPU memory freed.." << std::endl;
}