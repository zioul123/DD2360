#ifndef HELPER_H
#define HELPER_H

#include "Particles.h"

#define MAX_GPU_PARTICLES 14155776  // the number of particles per species in the GEM_3D file is chosen for maximum

// Modes for running the copy_particles function
enum PICMode 
{
    CPU_TO_GPU,
    GPU_TO_CPU
};

void print(std::string str);


void allocate_gpu_memory(struct particles* part, int grdSize, int fieldSize, 
                         particles_pointers* p_p, ids_pointers* i_p, 
                         grd_pointers* g_p, field_pointers* f_p);


void createStreams(cudaStream_t** streams, int nStreams);


void copy_mover_constants_to_GPU(struct EMfield* field, struct grid* grd, 
                                 field_pointers f_p, grd_pointers g_p,
                                 int grdSize, int field_size);


void copy_particles(struct particles* part, particles_pointers p_p, PICMode mode, 
                    long from, long to, bool verbose=false);


void copy_particles_async(struct particles* part, particles_pointers p_p, PICMode mode, 
                          long from, long to, cudaStream_t stream);


void copy_interp_initial_to_GPU(struct interpDensSpecies* ids, ids_pointers i_p,
                                int grdSize, int rhocSize);


void copy_interp_results(struct interpDensSpecies* ids, ids_pointers i_p, int grdSize, int rhocSize);


void free_gpu_memory(particles_pointers* p_p, ids_pointers* i_p, grd_pointers* g_p, field_pointers* f_p);


void destroyStreams(cudaStream_t* streams, int nStreams);


#endif
