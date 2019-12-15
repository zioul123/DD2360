#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

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


/** CPU serial function to move a single particle during one subcycle. */
__host__ void h_move_particle(int i, int part_NiterMover, struct grid* grd, struct parameters* param,
        const dt_info dt_inf, particle_info p_info, field_pointers f_pointers, grd_pointers g_pointers) {
    // extracting the particle variables out of the auxiliary struct
    FPpart* part_x = p_info.x;
    FPpart* part_y = p_info.y;
    FPpart* part_z = p_info.z;
    FPpart* part_u = p_info.u;
    FPpart* part_v = p_info.v;
    FPpart* part_w = p_info.w;

    // extracting the field variables out of the auxiliary struct
    FPfield* field_Ex_flat = f_pointers.Ex_flat;
    FPfield* field_Ey_flat = f_pointers.Ey_flat;
    FPfield* field_Ez_flat = f_pointers.Ez_flat;
    FPfield* field_Bxn_flat = f_pointers.Bxn_flat;
    FPfield* field_Byn_flat = f_pointers.Byn_flat;
    FPfield* field_Bzn_flat = f_pointers.Bzn_flat;

    // extracting the grid variables out of the auxiliary struct
    FPfield* grd_XN_flat = g_pointers.XN_flat;
    FPfield* grd_YN_flat = g_pointers.YN_flat;
    FPfield* grd_ZN_flat = g_pointers.ZN_flat;

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


/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    // "global" environment variables
    FPpart dt_sub_cycling = (FPpart) param->dt / ((double) part->n_sub_cycles);
    FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;
    const dt_info dt_inf { dt_sub_cycling, dto2, qomdt2 };

    // start subcycling
    for (int i_sub=0; i_sub < part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i < part->nop; i++){
            /*h_move_particle(i, part->NiterMover,
                grd, param, dt_inf,
                part->x, part->y, part->z,
                part->u, part->v, part->w,
                field->Ex_flat, field->Ey_flat, field->Ez_flat,
                field->Bxn_flat, field->Byn_flat, field->Bzn_flat,
                grd->XN_flat, grd->YN_flat, grd->ZN_flat);  */

            particle_info p_info {
                part->x, part->y, part->z,
                part->u, part->v, part->w
            };

            field_pointers f_pointers {
                field->Ex_flat, field->Ey_flat, field->Ez_flat,
                field->Bxn_flat, field->Byn_flat, field->Bzn_flat
             };

            grd_pointers g_pointers{
                grd->XN_flat, grd->YN_flat, grd->ZN_flat
            };
            h_move_particle(i, part->NiterMover, grd, param, dt_inf, p_info, f_pointers, g_pointers);

        }  // end of one particle
    } // end of subcycling
                                                                        
    return(0); // exit successfully
} // end of the mover


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


/** GPU kernel to interpolate for a single particle during one subcycle. */
__global__ void g_interp_particle(int nop, struct grid grd,
    particles_pointers part, ids_pointers ids, grd_pointers grd_p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > nop) return;

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


/** Interpolation Particle --> Grid: This is for species
 *  TODO: Move the malloc and free to sputniPIC.cpp instead of here.
 */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd,
    particles_pointers p_p, ids_pointers i_p, grd_pointers g_p, int grdSize, int rhocSize)
{
    // Copy CPU arrays to GPU
    {
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

    // Launch interpolation kernel
    g_interp_particle<<<(part->nop+TPB-1)/TPB, TPB>>>(part->nop, *grd, p_p, i_p, g_p);
    cudaDeviceSynchronize();

    // Copy GPU arrays back to CPU
    {
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
