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

  initWith<<<blocks, threadsPerBlock>>>(3, a, N);
  initWith<<<blocks, threadsPerBlock>>>(4, b, N);
  initWith<<<blocks, threadsPerBlock>>>(0, c, N);

  addVectorsInto<<<blocks, threadsPerBlock>>>(c, a, b, N);
  if (checkCuda(cudaDeviceSynchronize()) != cudaSuccess){
    exit(1);
  }

  checkElementsAre(7, c, N);

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
