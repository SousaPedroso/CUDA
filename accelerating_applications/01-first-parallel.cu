#include <stdio.h>

__global__ void firstParallel()
{
  printf("This should be running in parallel.\n");
}

int main()
{

  firstParallel<<<5,5>>>();


  cudaDeviceSynchronize();

}
