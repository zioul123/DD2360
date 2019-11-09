#!/bin/bash

# The name of the script is myjob
#SBATCH -J runs

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 1:00:00
#SBATCH -A edu19.DD2360

# Number of nodes
#SBATCH --nodes=1
#SBATCH -e error_file.e

# Run the executable file
# and write the output into my_output_file

# experiments wiht CPU
num_iterations=1000
echo "========================= CPU EXPERIMENTS =========================" >> a_output
for num_particles in 10000, 50000, 100000
do
    srun -n 1 ./a.out $num_particles $num_iterations 32 include_cpu exclude_copy_time >> a_output
done


# experiments only with GPU
echo "========================= GPU EXPERIMENTS =========================" >> a_output
for num_particles in 10000, 50000, 100000
do
    for tpb in 16, 32, 64, 128, 256
    do
        for mem_time in "exclude_copy_time", "include_copy_time"
        do
            srun -n 1 ./a.out $num_particles $num_iterations $tpb exclude_cpu $mem_time >> a_output
        done
    done
done
