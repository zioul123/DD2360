#ifndef HELPER_H
#define HELPER_H

#include "Particles.h"

/*void allocate_interp_gpu_memory(particles *part, FPpart *part_copies, FPinterp *part_copy_q,
                                FPinterp *ids_copies, FPfield *grd_copies, int grdSize);*/
void allocate_interp_gpu_memory(struct particles* part, int grdSize, particles_pointers* p_p,
                                ids_pointers* i_p, grd_pointers* g_p);


void copy_interp_arrays(struct particles* part, struct interpDensSpecies* ids, struct grid* grd,
                        particles_pointers p_p, ids_pointers i_p, grd_pointers g_p, int grdSize,
                        int rhocSize, std::string mode);


void allocate_mover_gpu_memory(struct particles* part, int grdSize, int field_size, particle_info* p_info,
                               field_pointers* f_pointers, grd_pointers* g_pointers);

void copy_mover_arrays(struct particles* part, struct EMfield* field, struct grid* grd, particle_info p_info,
                       field_pointers f_pointers, grd_pointers g_pointers, int grdSize, int field_size,
                       std::string mode);


void free_gpu_memory(particles_pointers* p_p, ids_pointers* i_p, grd_pointers* g_p,
                     particle_info* p_info, field_pointers* f_pointers, grd_pointers* g_pointers);

#endif
