/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"

#include "helper.h"  // our helper functions

int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);
    
    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp= 0.0;
    
    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield field;
    field_allocate(&grd,&field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd,&ids[is],is);
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    
    // Allocate Particles
    particles *part = new particles[param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&part[is],is);
    }
    
    // Initialization
    initGEM(&param,&grd,&field,&field_aux,part,ids);

    // -------------------------------------------------------------- //
    // ------ Additions for GPU version ----------------------------- //
    // Declare GPU copies of arrays
    int grdSize = grd.nxn * grd.nyn * grd.nzn;
    int rhocSize = grd.nxc * grd.nyc * grd.nzc;
    int field_size = grd.nxn * grd.nyn * grd.nzn;
    particles_pointers p_p; ids_pointers i_p; grd_pointers g_p; field_pointers f_p; // on the GPU memory
    allocate_gpu_memory(part, grdSize, field_size, &p_p, &i_p, &g_p, &f_p);  // Allocates maximum MAX_GPU_PARTICLES particles
    // Declare CUDA streams if enabled
    cudaStream_t* streams;
    if (STREAMS_ENABLED) createStreams(&streams);

     // on the GPU memory
    std::cout << "In [main]: All GPU memory allocation: done" << std::endl;


    // -------------------------------------------------------------- //

    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);
        
        // This version calls the mover_PC and interpP2G functions in sequence
        if (!KERNELS_COMBINED)
        {
            // implicit mover
            iMover = cpuSecond(); // start timer for mover
            for (int is=0; is < param.ns; is++)
                // mover_PC(&part[is],&field,&grd,&param);
                mover_PC(&part[is], &field, &grd, &param, p_p, f_p, g_p, grdSize, field_size, streams, STREAMS_ENABLED);
            eMover += (cpuSecond() - iMover); // stop timer for mover
            
            // interpolation particle to grid
            iInterp = cpuSecond(); // start timer for the interpolation step
            // interpolate species
            for (int is=0; is < param.ns; is++)
                interpP2G(&part[is],&ids[is],&grd, p_p, i_p, g_p, grdSize, rhocSize, streams, STREAMS_ENABLED);
            // Continue execution outside the if-else clause
        }
        // This version calls the function that combines movement and interp of particles
        else if (KERNELS_COMBINED)
        {
            // implicit mover
            iMover = cpuSecond(); // start timer for mover
            for (int is=0; is < param.ns; is++)
                // mover_PC(&part[is],&field,&grd,&param);
                combinedMoveInterp(&part[is], &field, &grd, &ids[is], &param, 
                                   p_p, f_p, g_p, i_p, grdSize, field_size, rhocSize, 
                                   streams, STREAMS_ENABLED);
            eMover += (cpuSecond() - iMover); // stop timer for mover
            
            // interpolation particle to grid was complete already
            iInterp = cpuSecond(); // start timer for the interpolation step
            // Continue execution outside the if-else clause
        }

        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&grd,&param);
        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&param);
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd,ids,&idn);
        }
    }  // end of one PIC cycle
    
    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }

    // -------------------------------------------------------------- //
    // ------ Additions for GPU version ----------------------------- //
    // Free GPU arrays
    free_gpu_memory(&p_p, &i_p, &g_p, &f_p);
    if (STREAMS_ENABLED) destroyStreams(streams);

    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    // exit
    return 0;
}


