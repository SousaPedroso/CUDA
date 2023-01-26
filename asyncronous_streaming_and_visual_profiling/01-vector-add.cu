#include <stdio.h>
#include <assert.h>

__global__ void initWith(float num, float *a, int N)
{
  int threadId, gridStride;
  threadId = threadIdx.x + blockIdx.x * blockDim.x;
  gridStride = blockDim.x * gridDim.x;

  for(int i = threadId; i < N; i+= gridStride){
    if (i < N) a[i] = num;
  }
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
  int threadId, gridStride;
  threadId = threadIdx.x + blockIdx.x * blockDim.x;
  gridStride = blockDim.x * gridDim.x;

  for(int i = threadId; i < N; i+=gridStride){
    if (i<N) result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *array, int N)
{

  for(int i = 0; i < N; i+=1){
    if(array[i] != target){
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
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
  int deviceId, processorCount, blocks, threadsPerBlock;
  cudaDeviceProp props;
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  if (checkCuda(cudaMallocManaged(&a, size)) != cudaSuccess){
    exit(1);
  }

  if (checkCuda(cudaMallocManaged(&b, size)) != cudaSuccess){
    exit(1);
  }

  if (checkCuda(cudaMallocManaged(&c, size)) != cudaSuccess){
    exit(1);
  }

  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&props, deviceId);

  // https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html

  // maxBlocksPerMultiProcessor is 12-above CUDA
  processorCount = props.multiProcessorCount;
  blocks = processorCount*2;
  threadsPerBlock = props.maxThreadsPerBlock;

  /*
    Experiments using cudaMemPrefetchAsync to understand its impact on
    page-faulting and memory migration
   * Using cudaMemPrefetchAsync each time reduced the total time of execution
   * of the program, decreasing from 4285931ns (97.5%) to 122655ns (52.6%)
   * the amount of time used by initWith method, due to not having anymore
   * data migration overhead

  */
  cudaMemPrefetchAsync(a, size, deviceId);
  cudaMemPrefetchAsync(b, size, deviceId);
  cudaMemPrefetchAsync(c, size, deviceId);

  cudaError_t addVectorsErr;
  cudaStream_t streamA;
  cudaStream_t streamB;
  cudaStream_t streamC;
  cudaStreamCreate(&streamA);
  cudaStreamCreate(&streamB);
  cudaStreamCreate(&streamC);

  initWith<<<blocks, threadsPerBlock, 0, streamA>>>(3, a, N);
  initWith<<<blocks, threadsPerBlock, 0, streamB>>>(4, b, N);
  initWith<<<blocks, threadsPerBlock, 0, streamC>>>(0, c, N);

  addVectorsInto<<<blocks, threadsPerBlock>>>(c, a, b, N);

  addVectorsErr = cudaGetLastError();
  if (checkCuda(addVectorsErr) != cudaSuccess){
    exit(1);
  }

  if (checkCuda(cudaDeviceSynchronize()) != cudaSuccess){
    exit(1);
  }

  cudaMemPrefetchAsync(c, size, cudaCpuDeviceId); // Prefetch c to CPU

  checkElementsAre(7, c, N);

  if (checkCuda(cudaStreamDestroy(streamA)) != cudaSuccess){
    exit(1);
  }

  if (checkCuda(cudaStreamDestroy(streamB)) != cudaSuccess){
    exit(1);
  }

  if (checkCuda(cudaStreamDestroy(streamC)) != cudaSuccess){
    exit(1);
  }

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
