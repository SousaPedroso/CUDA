#include <stdio.h>

__global__ void loop()
{
    printf("This is iteration number %d\n", threadIdx.x);
}

int main()
{

  int N = 10;
  loop<<<1, N>>>();
  
  cudaDeviceSynchronize();
}
