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
    // Declare GPU copies of arrays for interpP2G
    int grdSize = grd.nxn * grd.nyn * grd.nzn;
    int rhocSize = grd.nxc * grd.nyc * grd.nzc;
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
    particles_pointers p_p {
        part_copies[0], part_copies[1], part_copies[2],
        part_copies[3], part_copies[4], part_copies[5], part_copy_q
    };
    ids_pointers i_p {
        ids_copies[0], ids_copies[1], ids_copies[2],
        ids_copies[3], ids_copies[4], ids_copies[5], ids_copies[6],
        ids_copies[7], ids_copies[8], ids_copies[9], ids_copies[10]
    };
    grd_pointers g_p {
        grd_copies[0], grd_copies[1], grd_copies[2]
    };


    // Declare GPU copies of arrays for mover_PC
    FPpart* part_info_copies[6];
    FPfield* f_pointer_copies[6];
    FPfield* g_pointer_copies[3];

    int field_size = grd.nxn * grd.nyn * grd.nzn;

    // Allocate GPU arrays for mover_PC
    {
        for (int i = 0; i < 6; i++)
            cudaMalloc(&part_info_copies[i], part->npmax * sizeof(FPpart));

        for (int i = 0; i < 6; i++)
            cudaMalloc(&f_pointer_copies[i], field_size * sizeof(FPfield));

        for (int i = 0; i < 3; i++)
            cudaMalloc(&g_pointer_copies[i], grdSize * sizeof(FPfield));
    }

    // Put GPU array pointers into structs for mover_PC
    particle_info p_info {
        part_info_copies[0], part_info_copies[1], part_info_copies[2],
        part_info_copies[3], part_info_copies[4], part_info_copies[5]
    };

    field_pointers f_pointers {
        f_pointer_copies[0], f_pointer_copies[1], f_pointer_copies[2],
        f_pointer_copies[3], f_pointer_copies[4], f_pointer_copies[5]
    };

    grd_pointers g_pointers {
        g_pointer_copies[0], g_pointer_copies[1], g_pointer_copies[2]
    };


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
        
        
        
        // implicit mover
        iMover = cpuSecond(); // start timer for mover
        for (int is=0; is < param.ns; is++)
            // mover_PC(&part[is],&field,&grd,&param);
            mover_PC(&part[is], &field, &grd, &param, p_info, f_pointers, g_pointers, grdSize, field_size);

        eMover += (cpuSecond() - iMover); // stop timer for mover
        
        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++)
            interpP2G(&part[is],&ids[is],&grd, p_p, i_p, g_p, grdSize, rhocSize);
        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&grd,&param);
        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&param);
        
        
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd,ids,&idn);
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
        
    
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
    {
        cudaFree(part_copy_q);
        for (int i = 0; i < 6; ++i)
            cudaFree(part_copies[i]);
        for (int i = 0; i < 11; ++i)
            cudaFree(ids_copies[i]);
        for (int i = 0; i < 3; ++i)
            cudaFree(grd_copies[i]);
    }
    // -------------------------------------------------------------- //

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


