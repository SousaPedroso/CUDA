#include <stdio.h>

__global__ void initializeElementsTo(int initialValue, int *a, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) a[i] = initialValue;
}

int main()
{

  int N = 1000;

  int *a;
  size_t size = N * sizeof(int);

  cudaMallocManaged(&a, size);

  size_t threads_per_block = 256;

  size_t number_of_blocks = 4;

  int initialValue = 6;

  initializeElementsTo<<<number_of_blocks, threads_per_block>>>(initialValue, a, N);
  cudaDeviceSynchronize();

  for (int i = 0; i < N; ++i)
  {
    if(a[i] != initialValue)
    {
      printf("FAILURE: target value: %d\t a[%d]: %d\n", initialValue, i, a[i]);
      cudaFree(a);
      exit(1);
    }
  }
  printf("SUCCESS!\n");

  cudaFree(a);
}
