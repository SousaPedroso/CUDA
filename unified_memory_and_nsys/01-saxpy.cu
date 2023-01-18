#include <stdio.h>
#include <assert.h>

#define N 2048 * 2048 // Number of elements in each vector

/*
 * Optimize this already-accelerated codebase. Work iteratively,
 * and use nsys to support your work.
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 20us.
 *
 * Some bugs have been placed in this codebase for your edification.
 */

__global__ void initWith(float * vector, float num){
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int gridStride = gridDim.x * blockDim.x;

    for (int i=threadId; i<N; i+=gridStride){
        if (i < N) vector[i] = num;
    }
}

__global__ void saxpy(float * a, float * b, float * c)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = gridDim.x * blockDim.x;

    for (int i=threadId; i<N; i+=gridStride){
        if ( i < N ) c[threadId] = 2 * a[threadId] + b[threadId];
    }
}

cudaError_t checkCuda(cudaError_t result){
    if (result != cudaSuccess){
        fprintf(stderr, "CUDA RUNTIME ERROR: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

int main()
{
    float *a, *b, *c;
    int deviceId, multiProcessorCount;
    cudaDeviceProp props;
    int size = N * sizeof (int); // The total number of bytes per vector

    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&props, deviceId);
    multiProcessorCount = props.multiProcessorCount;

    if (checkCuda(cudaMallocManaged(&a, size)) != cudaSuccess){
        exit(1);
    }
    if (checkCuda(cudaMallocManaged(&b, size)) != cudaSuccess){
        exit(1);
    }
    if (checkCuda(cudaMallocManaged(&c, size)) != cudaSuccess){
        exit(1);
    }

    int threads_per_block = 256;
    int number_of_blocks = multiProcessorCount;

    if (checkCuda(cudaMemPrefetchAsync(a, size, deviceId)) != cudaSuccess){
        exit(1);
    }
    if (checkCuda(cudaMemPrefetchAsync(b, size, deviceId)) != cudaSuccess){
        exit(1);
    }
    if (checkCuda(cudaMemPrefetchAsync(c, size, deviceId)) != cudaSuccess){
        exit(1);
    }

    // Initialize memory
    initWith<<<number_of_blocks, threads_per_block>>>(a, 2);

    initWith<<<number_of_blocks, threads_per_block>>>(b, 1);

    initWith<<<number_of_blocks, threads_per_block>>>(c, 0);

    if (checkCuda(cudaDeviceSynchronize()) != cudaSuccess){
        exit(1);
    }

    saxpy <<< number_of_blocks, threads_per_block >>> ( a, b, c );
    if (checkCuda(cudaDeviceSynchronize()) != cudaSuccess){
        exit(1);
    }

    cudaMemPrefetchAsync(a, size, cudaCpuDeviceId);

    // Print out the first and last 5 values of c for a quality check
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %f, ", i, c[i]);
    printf ("\n");

    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %f, ", i, c[i]);
    printf ("\n");

    if (checkCuda(cudaFree(a)) != cudaSuccess){
        exit(1);
    }

    if (checkCuda(cudaFree(b)) != cudaSuccess){
        exit(1);
    }

    if (checkCuda(cudaFree(c)) != cudaSuccess){
        exit(1);
    }
}
