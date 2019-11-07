#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 10000
#define SEED 10
#define EPSILON 1e-5

int TPB = 32;

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
__host__ bool arraysMatch(Particle* arr1, Particle* arr2)
{
    for (int i = 0; i < NUM_PARTICLES; i++) {
        if (abs(arr1[i].position.x - arr2[i].position.x) > EPSILON ||
            abs(arr1[i].position.y - arr2[i].position.y) > EPSILON ||
            abs(arr1[i].position.z - arr2[i].position.z) > EPSILON)
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
void randomizeParticles(Particle* particles) {
    srand(0);
    for (int i = 0; i < NUM_PARTICLES; i++) {
      randomizeFloat3(&particles[i].position);
      randomizeFloat3(&particles[i].velocity);
    }
}

// Updates a particle's position by its velocity, then updates its velocity
__host__ __device__ void updateParticle(Particle* particle, int id, int iter) {
  int dt = 1;

  // update position
  particle->position.x += dt * particle->velocity.x;
  particle->position.y += dt * particle->velocity.y;
  particle->position.z += dt * particle->velocity.z;

  // update the velocity randomly
  particle->velocity.x += gen_random(SEED, id, iter, NUM_PARTICLES);
  particle->velocity.y += gen_random(SEED, id, iter, NUM_PARTICLES);
  particle->velocity.z += gen_random(SEED, id, iter, NUM_PARTICLES);
}

// CPU function that updates a given particle.
void cpu_updatePositionAndVelocity(Particle* particle, int id, int iter) {
  updateParticle(particle, id, iter);
}

// Kernel that finds a given Particle's ID then updates it if within range.
__global__ void gpu_updatePositionAndVelocity(Particle* particles, int iter) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id >= NUM_PARTICLES) // If out of bounds, ignore the Particle.
    return; 
  else 
    updateParticle(&particles[id], id, iter);
}

// Perform the update step for all Particles in the array on CPU with a for loop.
void cpu_updateParticles(Particle* particles, int iter) {
  // srand(time(NULL))
  for (int i = 0; i < NUM_PARTICLES; i++) {
    cpu_updatePositionAndVelocity(&particles[i], i, iter);
  }
}

// Perform the update step for all Particles in the array by launching GPU kernels.
void gpu_updateParticles(Particle* particles, int iter) {
  // srand(time(NULL))
  gpu_updatePositionAndVelocity<<<(NUM_PARTICLES + TPB - 1)/TPB, TPB>>>(particles, iter);
}

int main() {
  // Declare variables
  Particle *c_particles, *g_particles, *g_result;
  double iStart, iElaps;

  // Initialize array for CPU
  c_particles = (Particle*) malloc(NUM_PARTICLES*sizeof(Particle));
  randomizeParticles(c_particles);

  // Initialize array for GPU - particle positions/velocities in device memory are a copy of those in host memory
  g_result = (Particle*) malloc(NUM_PARTICLES*sizeof(Particle)); // Used to store the result of GPU simulation
  cudaMalloc(&g_particles, NUM_PARTICLES*sizeof(Particle));
  cudaMemcpy(g_particles, c_particles, NUM_PARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);

  // CPU Version
  printf("CPU simulation started...\n"); fflush(stdout);
  iStart = cpuSecond();
  for (int i = 0; i < NUM_ITERATIONS; i++) {
    cpu_updateParticles(c_particles, i);
  }
  iElaps = cpuSecond() - iStart;
  printf("Done in %f!\n\n", iElaps); fflush(stdout);
// printf("Final particle position: ");
// printParticle(c_particles, 5);

  // GPU Version
  printf("GPU simulation started...\n"); fflush(stdout);
  iStart = cpuSecond();
  for (int i = 0; i < NUM_ITERATIONS; i++) {
    gpu_updateParticles(g_particles, i);
    cudaDeviceSynchronize();
  }
  iElaps = cpuSecond() - iStart;
  printf("Done in %f!\n", iElaps); fflush(stdout);
  cudaMemcpy(g_result, g_particles, NUM_PARTICLES*sizeof(Particle), cudaMemcpyDeviceToHost);
// printf("Final particle position: ");
// printParticle(g_result, 5);

  printf(arraysMatch(g_result, c_particles) ? "Results match!\n" : "Results are wrong!\n");

  // Free arrays
  free(c_particles);
  cudaFree(g_particles);
}
