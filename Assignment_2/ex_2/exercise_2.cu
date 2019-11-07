#include <stdio.h>
#include <sys/time.h>
#define A 0.1234
#define TPB 256
#define INITIAL_N 10000
#define FINAL_N 100000000
#define EPSILON 1e-5
// #define ARRAY_SIZE 10000
int ARRAY_SIZE = INITIAL_N;
// Get the current time
double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

// Fill the array with random floats from 0 to 1
__host__ void fillArray(float* arr) 
{
    srand(0);
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        arr[i] = (float) rand() / RAND_MAX;
    }
}

__host__ void cpu_saxpy(float a, float* x, float* y) 
{
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void gpu_saxpy(float a, float* x, float* y)
{
    // Get thread ID
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    y[i] = a * x[i] + y[i];
}

// Compare two arrays. If the values are within EPSILON of each other,
// return true, else false.
__host__ bool arraysMatch(float* arr1, float* arr2)
{
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        if (abs(arr1[i] - arr2[i]) > EPSILON)
            return false;
    }
    return true;
}

int main()
{
    // Vary ARRAY_SIZE. To use a fixed array size, uncomment the define statement and
    // comment out the loop.
    printf("ARR SIZE  | CPU      | GPU      | Correctness\n");
    for (; ARRAY_SIZE < FINAL_N; ARRAY_SIZE *= 2) {
        printf("%9d | ", ARRAY_SIZE);

        // Create array pointers x and y on CPU and GPU
        float *c_x, *c_y, *g_x, *g_y, *g_res;
        c_x = (float*)malloc(ARRAY_SIZE*sizeof(float));
        c_y = (float*)malloc(ARRAY_SIZE*sizeof(float));
        g_res = (float*)malloc(ARRAY_SIZE*sizeof(float)); // To store result from GPU
        cudaMalloc(&g_x, ARRAY_SIZE*sizeof(float));
        cudaMalloc(&g_y, ARRAY_SIZE*sizeof(float));
        if (c_x == NULL || c_y == NULL || g_res == NULL || g_x == NULL || g_y == NULL) {
            printf("malloc failed.\n");
            return -1;
        }
        
        // Fill arrays
        fillArray(c_x);
        fillArray(c_y);
        cudaMemcpy(g_x, c_x, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(g_y, c_y, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

        // Create timing variables
        double iStart, iElaps;

        // Perform SAXPY on CPU
        // printf("Computing SAXPY on the CPU...");
        iStart = cpuSecond();
        cpu_saxpy(A, c_x, c_y);
        iElaps = cpuSecond() - iStart;
        // printf(" Done in %f!\n\n", iElaps);
        printf("%8.6f | ", iElaps);

        // Perform SAXPY on GPU
        // printf("Computing SAXPY on the GPU...");
        iStart = cpuSecond();
        gpu_saxpy<<<(ARRAY_SIZE+TPB-1)/TPB, TPB>>>(A, g_x, g_y);
        cudaDeviceSynchronize();
        iElaps = cpuSecond() - iStart;
        // printf(" Done in %f!\n\n", iElaps);
        printf("%8.6f | ", iElaps);

        // Compare results to ensure correctness
        // printf("Comparing the output for each implementation...");
        cudaMemcpy(g_res, g_y, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
        // printf(arraysMatch(c_y, g_res) ? " Correct!\n" : " Wrong!\n");
        printf(arraysMatch(c_y, g_res) ? "Correct\n" : " Wrong\n");
        fflush(stdout);

        // Free memory
        free(c_x);
        free(c_y);
        free(g_res);
        cudaFree(g_x);
        cudaFree(g_y);
    }
    
    return 0;
}