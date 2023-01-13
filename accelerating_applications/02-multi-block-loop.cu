#include <stdio.h>

__global__ void loop()
{
    printf("This is iteration number %d\n", blockDim.x*blockIdx.x+threadIdx.x);
}

int main()
{

  int b=2, N = 10;
  
  loop<<<b, N/b>>>();
  cudaDeviceSynchronize();
}
