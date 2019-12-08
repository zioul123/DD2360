#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

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

/** Global auxilliary variables necessary to run h_move_particle(), values must be initialized
    by mover_PC() before calling it. */
FPpart dt_sub_cycling, dto2, qomdt2;

/** CPU serial function to move a single particle during one subcycle. */
__host__ void h_move_particle(int i, int part_NiterMover, struct grid* grd, struct parameters* param,
    FPpart* part_x, FPpart* part_y, FPpart* part_z, 
    FPpart* part_u, FPpart* part_v, FPpart* part_w,
    FPfield* field_Ex_flat, FPfield* field_Ey_flat, FPfield* field_Ez_flat,
    FPfield* field_Bxn_flat, FPfield* field_Byn_flat, FPfield* field_Bzn_flat,
    FPfield* grd_XN_flat, FPfield* grd_YN_flat, FPfield* grd_ZN_flat) 
{
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
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    weight[i][j][k] = xi[i] * eta[j] * zeta[k] * grd->invVOL;
        
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
        omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
        denom = 1.0/(1.0 + omdtsq);
        // solve the position equation
        ut= part_u[i] + qomdt2*Exl;
        vt= part_v[i] + qomdt2*Eyl;
        wt= part_w[i] + qomdt2*Ezl;
        udotb = ut*Bxl + vt*Byl + wt*Bzl;
        // solve the velocity equation
        uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
        vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
        wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
        // update position
        part_x[i] = xptilde + uptilde*dto2;
        part_y[i] = yptilde + vptilde*dto2;
        part_z[i] = zptilde + wptilde*dto2;
        
        
    } // end of iteration
    // update the final position and velocity
    part_u[i]= 2.0*uptilde - part_u[i];
    part_v[i]= 2.0*vptilde - part_v[i];
    part_w[i]= 2.0*wptilde - part_w[i];
    part_x[i] = xptilde + uptilde*dt_sub_cycling;
    part_y[i] = yptilde + vptilde*dt_sub_cycling;
    part_z[i] = zptilde + wptilde*dt_sub_cycling;


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
 
    // global auxiliary variables, updated before running mover_PC
    dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    
    // start subcycling
    for (int i_sub=0; i_sub < part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i < part->nop; i++){
            h_move_particle(i, part->NiterMover, grd, param,
                part->x, part->y, part->z,
                part->u, part->v, part->w, 
                field->Ex_flat, field->Ey_flat, field->Ez_flat,
                field->Bxn_flat, field->Byn_flat, field->Bzn_flat,
                grd->XN_flat, grd->YN_flat, grd->ZN_flat);
        }  // end of one particle
    } // end of subcycling
                                                                        
    return(0); // exit succcesfully
} // end of the mover

/** CPU serial function to interpolate for a single particle during one subcycle. */
__host__ void h_interp_particle(register long long i, struct grid* grd,
    FPpart* part_x, FPpart* part_y, FPpart* part_z, 
    FPpart* part_u, FPpart* part_v, FPpart* part_w, FPinterp* part_q,
    FPinterp *ids_rhon_flat, FPinterp *ids_rhoc_flat, 
    FPinterp *ids_Jx_flat, FPinterp *ids_Jy_flat, FPinterp *ids_Jz_flat, 
    FPinterp *ids_pxx_flat, FPinterp *ids_pxy_flat, FPinterp *ids_pxz_flat, 
    FPinterp *ids_pyy_flat, FPinterp *ids_pyz_flat, FPinterp *ids_pzz_flat, 
    FPfield* grd_XN_flat, FPfield* grd_YN_flat, FPfield* grd_ZN_flat) 
{
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int (floor((part_x[i] - grd->xStart) * grd->invdx));
    iy = 2 + int (floor((part_y[i] - grd->yStart) * grd->invdy));
    iz = 2 + int (floor((part_z[i] - grd->zStart) * grd->invdz));
    
    // distances from node
    xi[0]   = part_x[i] - grd_XN_flat[get_idx(ix-1, iy, iz, grd->nyn, grd->nzn)];
    eta[0]  = part_y[i] - grd_YN_flat[get_idx(ix, iy-1, iz, grd->nyn, grd->nzn)];
    zeta[0] = part_z[i] - grd_ZN_flat[get_idx(ix, iy, iz-1, grd->nyn, grd->nzn)];
    xi[1]   = grd_XN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part_x[i];
    eta[1]  = grd_YN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part_y[i];
    zeta[1] = grd_ZN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part_z[i];
    
    // calculate the weights for different nodes
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                weight[ii][jj][kk] = part_q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
    
    //////////////////////////
    // add charge density
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                ids_rhon_flat[get_idx(ix-i, iy-j, iz-k, grd->nyn, grd->nzn)] += weight[i][j][k] * grd->invVOL;
    
    
    ////////////////////////////
    // add current density - Jx
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_u[i] * weight[ii][jj][kk];
    
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                ids_Jx_flat[get_idx(ix-i, iy-j, iz-k, grd->nyn, grd->nzn)] += weight[i][j][k] * grd->invVOL;
    
    
    ////////////////////////////
    // add current density - Jy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_v[i] * weight[ii][jj][kk];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                ids_Jy_flat[get_idx(ix-i, iy-j, iz-k, grd->nyn, grd->nzn)] += weight[i][j][k] * grd->invVOL;
    
    
    
    ////////////////////////////
    // add current density - Jz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_w[i] * weight[ii][jj][kk];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                ids_Jz_flat[get_idx(ix-i, iy-j, iz-k, grd->nyn, grd->nzn)] += weight[i][j][k] * grd->invVOL;
    
    
    ////////////////////////////
    // add pressure pxx
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_u[i] * part_u[i] * weight[ii][jj][kk];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                ids_pxx_flat[get_idx(ix-i, iy-j, iz-k, grd->nyn, grd->nzn)] += weight[i][j][k] * grd->invVOL;
    
    
    ////////////////////////////
    // add pressure pxy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_u[i] * part_v[i] * weight[ii][jj][kk];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                ids_pxy_flat[get_idx(ix-i, iy-j, iz-k, grd->nyn, grd->nzn)] += weight[i][j][k] * grd->invVOL;
    
    
    
    /////////////////////////////
    // add pressure pxz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_u[i] * part_w[i] * weight[ii][jj][kk];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                ids_pxz_flat[get_idx(ix-i, iy-j, iz-k, grd->nyn, grd->nzn)] += weight[i][j][k] * grd->invVOL;
    
    
    /////////////////////////////
    // add pressure pyy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_v[i] * part_v[i] * weight[ii][jj][kk];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                ids_pyy_flat[get_idx(ix-i, iy-j, iz-k, grd->nyn, grd->nzn)] += weight[i][j][k] * grd->invVOL;
    
    
    /////////////////////////////
    // add pressure pyz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_v[i] * part_w[i] * weight[ii][jj][kk];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                ids_pyz_flat[get_idx(ix-i, iy-j, iz-k, grd->nyn, grd->nzn)] += weight[i][j][k] * grd->invVOL;
    
    
    /////////////////////////////
    // add pressure pzz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_w[i] * part_w[i] * weight[ii][jj][kk];
    for (int i=0; i < 2; i++)
        for (int j=0; j < 2; j++)
            for(int k=0; k < 2; k++)
                ids_pzz_flat[get_idx(ix-i, iy-j, iz-k, grd->nyn, grd->nzn)]= weight[i][j][k] * grd->invVOL;

}

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    for (register long long i = 0; i < part->nop; i++) {
        h_interp_particle(i, grd,
            part->x, part->y, part->z, 
            part->u, part->v, part->w, part->q,
            ids->rhon_flat, ids->rhoc_flat,
            ids->Jx_flat, ids->Jy_flat, ids->Jz_flat,
            ids->pxx_flat, ids->pxy_flat, ids->pxz_flat,
            ids->pyy_flat, ids->pyz_flat, ids->pzz_flat,
            grd->XN_flat, grd->YN_flat, grd->ZN_flat);
    }  
}
