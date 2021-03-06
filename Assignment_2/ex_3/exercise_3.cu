#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

// #define NUM_PARTICLES 10000
// #define NUM_ITERATIONS 10000
// int TPB = 16;

#define SEED 10
#define EPSILON 1e-5

typedef struct {
    float3 position;
    float3 velocity;
} Particle;


// Deterministically generates a "random" float, provided a seed and 3 integers.
__host__ __device__ float gen_random(int seed, int a, int b, int c) {
    return (float)((seed * a + b) % c) / c;
}


// Given an array of particles and an index, print that particle.
void printParticle(Particle* particles, int index){
    printf("%f %f %f %f %f %f\n",
    particles[index].position.x, particles[index].position.y, particles[index].position.z,
    particles[index].velocity.x, particles[index].velocity.y, particles[index].velocity.z);
}


// Compare two arrays of Particles. If their position coordinates are all within EPSILON of each other,
// return true, else false.
__host__ bool arraysMatch(Particle* arr1, Particle* arr2, int num_particles)
{
    for (int i = 0; i < num_particles; i++) {
        if (fabs(arr1[i].position.x - arr2[i].position.x) > EPSILON ||
        fabs(arr1[i].position.y - arr2[i].position.y) > EPSILON ||
        fabs(arr1[i].position.z - arr2[i].position.z) > EPSILON)
        return false;
    }
    return true;
}


// Get the current time
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


// Replaces the x, y and z values in a float3 to random values between 0 and 1.
void randomizeFloat3(float3* f3) {
    f3->x = (float) rand() / RAND_MAX;
    f3->y = (float) rand() / RAND_MAX;
    f3->z = (float) rand() / RAND_MAX;
}


// Randomizes the position and velocity of all Particles in an array.
void randomizeParticles(Particle* particles, int num_particles) {
    srand(0);
    for (int i = 0; i < num_particles; i++) {
        randomizeFloat3(&particles[i].position);
        randomizeFloat3(&particles[i].velocity);
    }
}


// Updates a particle's position by its velocity, then updates its velocity
__host__ __device__ void updateParticle(Particle* particle, int id, int iter, int num_particles) {
    int dt = 1;

    // update position
    particle->position.x += dt * particle->velocity.x;
    particle->position.y += dt * particle->velocity.y;
    particle->position.z += dt * particle->velocity.z;

    // update the velocity randomly
    particle->velocity.x += gen_random(SEED, id, iter, num_particles);
    particle->velocity.y += gen_random(SEED, id, iter, num_particles);
    particle->velocity.z += gen_random(SEED, id, iter, num_particles);
}


// CPU function that updates a given particle.
void cpu_updatePositionAndVelocity(Particle* particle, int id, int iter, int num_particles) {
    updateParticle(particle, id, iter, num_particles);
}


// Kernel that finds a given Particle's ID then updates it if within range.
__global__ void gpu_updatePositionAndVelocity(Particle* particles, int iter, int num_particles) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= num_particles) // If out of bounds, ignore the Particle.
    return;
    else
    updateParticle(&particles[id], id, iter, num_particles);
}


// Perform the update step for all Particles in the array on CPU with a for loop.
void cpu_updateParticles(Particle* particles, int iter, int num_particles) {
    // srand(time(NULL))
    for (int i = 0; i < num_particles; i++) {
        cpu_updatePositionAndVelocity(&particles[i], i, iter, num_particles);
    }
}


// Perform the update step for all Particles in the array by launching GPU kernels.
void gpu_updateParticles(Particle* particles, int iter, int num_particles, int tpb) {
    gpu_updatePositionAndVelocity<<<(num_particles + tpb - 1)/tpb, tpb>>>(particles, iter, num_particles);
}


int main(int argc, char** argv) {
    printf("Running the simulations with the following params:\n");

    // reading the command line arguments, without any kind of error checking
    const int num_particles = (int) strtol(argv[1], NULL, 10); // e.g. 10000 - NULL is the endpointer and 10 is the base
    const int num_iterations = (int) strtol(argv[2], NULL, 10); // e.g. 10000
    const int tpb = (int) strtol(argv[3], NULL, 10); // e.g. 32
    const char* include_cpu = argv[4];
    const char* include_copy_time = argv[5];
    printf("======== %s: %d, %s: %d, %s: %d\n\n", "num_particles", num_particles, "num_iterations", num_iterations, "tpb", tpb);


    // Declare variables
    Particle *c_particles, *g_particles, *g_result;
    double iStart, iElaps;

    // Initialize array for CPU
    c_particles = (Particle*) malloc(num_particles*sizeof(Particle));
    randomizeParticles(c_particles, num_particles);

    // Initialize array for GPU - particle positions/velocities in device memory are a copy of those in host memory
    g_result = (Particle*) malloc(num_particles*sizeof(Particle)); // Used to store the result of GPU simulation
    cudaMalloc(&g_particles, num_particles*sizeof(Particle));

    iStart = cpuSecond();
    cudaMemcpy(g_particles, c_particles, num_particles*sizeof(Particle), cudaMemcpyHostToDevice);
    double copy_time = cpuSecond() - iStart;


    // CPU Version
    if (strcmp(include_cpu, "include_cpu") == 0) {  // perfrom CPU version if wanted by the user
        printf("CPU simulation started...\n"); fflush(stdout);
        iStart = cpuSecond();
        for (int i = 0; i < num_iterations; i++) {
            cpu_updateParticles(c_particles, i, num_particles);
        }
        iElaps = cpuSecond() - iStart;
        printf("Done in %f!\n\n", iElaps); fflush(stdout);
    }
    else
        printf("Excluded the CPU experiment...\n\n");

    // GPU Version
    printf("GPU simulation started...\n"); fflush(stdout);
    iStart = cpuSecond();
    for (int i = 0; i < num_iterations; i++) {
        gpu_updateParticles(g_particles, i, num_particles, tpb);
        cudaDeviceSynchronize();
    }
    iElaps = cpuSecond() - iStart;

    if (strcmp(include_copy_time, "include_copy_time") == 0) {
        printf("Done (including copy time) in %f!\n\n", iElaps + copy_time);
        fflush(stdout);
    }
    else {
        printf("Done (excluding copy time) in %f!\n\n", iElaps);
        fflush(stdout);
    }

    // copying the result back from the GPU memory to the CUP memory
    cudaMemcpy(g_result, g_particles, num_particles*sizeof(Particle), cudaMemcpyDeviceToHost);

    // if CPU version is perfromed, then compare it with GPU version
    if (strcmp(include_cpu, "include_cpu") == 0)
        printf(arraysMatch(g_result, c_particles, num_particles) ? "Results match!\n" : "Results are wrong!\n");
    printf("========================================================== \n\n\n");

    // Free arrays
    free(c_particles);
    cudaFree(g_particles);
}
