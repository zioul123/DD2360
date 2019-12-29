#ifndef PARTICLES_H
#define PARTICLES_H

#include <math.h>

#include "Alloc.h"
#include "Parameters.h"
#include "PrecisionTypes.h"
#include "Grid.h"
#include "EMfield.h"
#include "InterpDensSpecies.h"


/** Structs containing the arrays necessary to run h_interp_particle*/
typedef struct {
    FPpart* x; FPpart* y; FPpart* z;
    FPpart* u; FPpart* v; FPpart* w; 
    FPinterp* q; // Only for interp2g
} particles_pointers;


typedef struct {
    FPinterp* rhon_flat; FPinterp* rhoc_flat;
    FPinterp* Jx_flat; FPinterp* Jy_flat; FPinterp* Jz_flat;
    FPinterp* pxx_flat; FPinterp* pxy_flat; FPinterp* pxz_flat;
    FPinterp* pyy_flat; FPinterp* pyz_flat; FPinterp* pzz_flat;
} ids_pointers;


/** NOTE: This struct is used in both moving and interpolating functions */
typedef struct {
    FPfield* XN_flat; FPfield* YN_flat; FPfield* ZN_flat;
} grd_pointers;


/** Structs for mover_PC */

typedef struct {
    FPfield* Ex_flat; FPfield* Ey_flat; FPfield* Ez_flat;
    FPfield* Bxn_flat; FPfield* Byn_flat; FPfield* Bzn_flat;
} field_pointers;


struct particles {
    
    /** species ID: 0, 1, 2 , ... */
    int species_ID;
    
    /** maximum number of particles of this species on this domain. used for memory allocation */
    long npmax;
    /** number of particles of this species on this domain */
    long nop;
    
    /** Electron and ions have different number of iterations: ions moves slower than ions */
    int NiterMover;
    /** number of particle of subcycles in the mover */
    int n_sub_cycles;
    
    
    /** number of particles per cell */
    int npcel;
    /** number of particles per cell - X direction */
    int npcelx;
    /** number of particles per cell - Y direction */
    int npcely;
    /** number of particles per cell - Z direction */
    int npcelz;
    
    
    /** charge over mass ratio */
    FPpart qom;
    
    /* drift and thermal velocities for this species */
    FPpart u0, v0, w0;
    FPpart uth, vth, wth;
    
    /** particle arrays: 1D arrays[npmax] */
    FPpart* x; FPpart*  y; FPpart* z; FPpart* u; FPpart* v; FPpart* w;
    /** q must have precision of interpolated quantities: typically double. Not used in mover */
    FPinterp* q;
    
    
    
};

/** allocate particle arrays */
void particle_allocate(struct parameters*, struct particles*, int, bool enableStreaming=true);


/** deallocate */
void particle_deallocate(struct particles*, bool enableStreaming=true);


/** particle mover */
// int mover_PC(struct particles*, struct EMfield*, struct grid*, struct parameters*);
int mover_PC(struct particles*, struct EMfield*, struct grid*, struct parameters*,
             particles_pointers, field_pointers, grd_pointers, int, int, 
             cudaStream_t* streams, bool enableStreaming=true);


/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd,
               particles_pointers, ids_pointers, grd_pointers, int, int, 
               cudaStream_t* streams, bool enableStreaming=true);

#endif
