#include <stdio.h>
#define N 256
#define TPB 64

__global__ void printKernel()
{
    // Get thread ID
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    // Print message
    printf("Hello World! My threadId is %d\n\n", i);
}

int main()
{
    // Launch kernel to print
    printKernel<<<N/TPB, TPB>>>();
    cudaDeviceSynchronize();
    return 0;
}