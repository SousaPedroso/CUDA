__global__
void deviceKernel(int *a, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] = 1;
  }
}

void hostFunction(int *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 1;
  }
}

int main()
{

  int N = 2<<24;
  size_t size = N * sizeof(int);
  int *a;
  cudaMallocManaged(&a, size);

  /*
   * Conduct experiments to learn more about the behavior of
   * `cudaMallocManaged`.
   *
   * What happens when unified memory is accessed only by the GPU?
   * What happens when unified memory is accessed only by the CPU?
   * What happens when unified memory is accessed first by the GPU then the CPU?
   * What happens when unified memory is accessed first by the CPU then the GPU?
   *
   * Hypothesize about UM behavior, page faulting specificially, before each
   * experiment, and then verify by running `nsys`.
   */
  
  /* unified memory is accessed only by the CPU
   * There is no CUDA Memory Operation Statistics section
   * As expected, there is no evidence of memory migration or page faulting
  hostFunction(a, N);
  */

  /* unified memory is accessed only by the GPU
   * There is no CUDA Memory Operation Statistics section
   * Despite I not expected there was no evidence of memory migration
   * It makes sense since the CPU is only waiting for the GPU finish its
   * work
  deviceKernel<<<40, 256>>>(a, N);
  cudaDeviceSynchronize();
  */

  /* unified memory is accessed first by the GPU then the CPU
  deviceKernel<<<40, 256>>>(a, N);
   * There is CUDA Memory Operation Statistics section
   * As expected, there is evidence of memory migration [CUDA Unified Memory memcpy DtoH],
   * since after GPU finish its work, the data should be used by the host function
  deviceKernel<<<40, 256>>>(a, N);
  cudaDeviceSynchronize();
  hostFunction(a, N);
  */

  /* unified memory is accessed first by the CPU then the GPU
   * There is CUDA Memory Operation Statistics section
   * The same ocurred as the previos experiment, but with [CUDA Unified Memory memcpy DtoH]
  hostFunction(a, N);
  deviceKernel<<<40, 256>>>(a, N);
  cudaDeviceSynchronize();
  */

  cudaFree(a);
}
