#include <stdio.h>

__global__ void printSuccessForCorrectExecutionConfiguration()
{

  if(threadIdx.x != 1023 && blockIdx.x != 255)
  {
    printf("Success!\n");
  } else {
    printf("Failure. Update the execution configuration as necessary.\n");
  }
}

int main()
{

  printSuccessForCorrectExecutionConfiguration<<<1, 1>>>();
  
  cudaDeviceSynchronize();
}
