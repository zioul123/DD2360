#ifndef HELPER_H
#define HELPER_H

#include "Particles.h"

#define MAX_GPU_PARTICLES 14155776  // the number of particles per species in the GEM_3D file is chosen for maximum


void print(std::string str);


void allocate_batch(FPpart*& batch_x, FPpart*& batch_y, FPpart*& batch_z,
                    FPpart*& batch_u, FPpart*& batch_v, FPpart*& batch_w,
                    FPpart*& batch_q, long batch_size, std::string mode);


void deallocate_batch(FPpart*& batch_x, FPpart*& batch_y, FPpart*& batch_z,
                      FPpart*& batch_u, FPpart*& batch_v, FPpart*& batch_w,
                      FPpart*& batch_q, std::string mode);


void batch_copy(FPpart*& batch_x, FPpart*& batch_y, FPpart*& batch_z, FPpart*& batch_u, FPpart*& batch_v,
                FPpart*& batch_w, FPpart*& batch_q, FPpart*& part_x, FPpart*& part_y, FPpart*& part_z,
                FPpart*& part_u, FPpart*& part_v, FPpart*& part_w, FPpart*& part_q,long from, long to,
                std::string mode, std::string direction);



void allocate_interp_gpu_memory(struct particles* part, int grdSize, particles_pointers* p_p,
                                ids_pointers* i_p, grd_pointers* g_p);


void copy_interp_arrays(struct particles* part, struct interpDensSpecies* ids, struct grid* grd,
                        particles_pointers p_p, ids_pointers i_p, grd_pointers g_p, int grdSize,
                        int rhocSize, std::string mode, long from=-1, long to=-1, bool verbose=false);


void allocate_mover_gpu_memory(struct particles* part, int grdSize, int field_size, particle_info* p_info,
                               field_pointers* f_pointers, grd_pointers* g_pointers);


void copy_mover_arrays(struct particles* part, struct EMfield* field, struct grid* grd, particle_info p_info,
                       field_pointers f_pointers, grd_pointers g_pointers, int grdSize, int field_size,
                       std::string mode, long from=-1, long to=-1, bool verbose=false);


void free_gpu_memory(particles_pointers* p_p, ids_pointers* i_p, grd_pointers* g_p,
                     particle_info* p_info, field_pointers* f_pointers, grd_pointers* g_pointers);


void create_particle_batch(long batch_size, particle_info p_info_batch);

#endif
