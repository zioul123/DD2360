#ifndef HELPER_H
#define HELPER_H

#include "Particles.h"

#define MAX_GPU_PARTICLES 14155776  // the number of particles per species in the GEM_3D file is chosen for maximum


// Modes for running the functions
enum PICMode 
{
    INTERP2G,
    MOVER_PC,
    PARTICLE_TO_BATCH,
    BATCH_TO_PARTICLE,
    CPU_TO_GPU,
    GPU_TO_CPU
};

void print(std::string str);


void allocate_batch(FPpart*& batch_x, FPpart*& batch_y, FPpart*& batch_z,
                    FPpart*& batch_u, FPpart*& batch_v, FPpart*& batch_w,
                    FPpart*& batch_q, long batch_size, PICMode mode);


void deallocate_batch(FPpart*& batch_x, FPpart*& batch_y, FPpart*& batch_z,
                      FPpart*& batch_u, FPpart*& batch_v, FPpart*& batch_w,
                      FPpart*& batch_q, PICMode mode);


void batch_copy(FPpart*& batch_x, FPpart*& batch_y, FPpart*& batch_z, FPpart*& batch_u, FPpart*& batch_v,
                FPpart*& batch_w, FPpart*& batch_q, FPpart*& part_x, FPpart*& part_y, FPpart*& part_z,
                FPpart*& part_u, FPpart*& part_v, FPpart*& part_w, FPpart*& part_q,
                long from, long to, PICMode mode, PICMode direction);


void allocate_gpu_memory(struct particles* part, int grdSize, int fieldSize, 
                         particles_pointers* p_p, ids_pointers* i_p, 
                         grd_pointers* g_p, field_pointers* f_p);


void copy_interp_arrays(struct particles* part, struct interpDensSpecies* ids, struct grid* grd,
                        particles_pointers p_p, ids_pointers i_p, grd_pointers g_p, 
                        int grdSize, int rhocSize, PICMode mode, 
                        long from=-1, long to=-1, bool verbose=false);


void copy_mover_arrays(struct particles* part, struct EMfield* field, struct grid* grd, 
                       particles_pointers p_p, field_pointers f_p, grd_pointers g_p, 
                       int grdSize, int field_size, PICMode mode, 
                       long from=-1, long to=-1, bool verbose=false);


void free_gpu_memory(particles_pointers* p_p, ids_pointers* i_p, grd_pointers* g_p, field_pointers* f_p);

#endif
