#include <stdio.h>

int main()
{

  int deviceId;
  int computeCapabilityMajor;
  int computeCapabilityMinor;
  int multiProcessorCount;
  int warpSize;

  cudaGetDevice(&deviceId); // points to the id of the currently active GPU.

  cudaDeviceProp props; // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
  cudaGetDeviceProperties(&props, deviceId); // get the properties of the currently active GPU
  computeCapabilityMajor = props.major;
  computeCapabilityMinor = props.minor;
  multiProcessorCount = props.multiProcessorCount;
  warpSize = props.warpSize; // (warp) group of 32 threads inside a block
                            // managed by Streaming MultiProcessors (SMs)

  /*
   * There should be no need to modify the output string below.
   */

  printf("Device ID: %d\nNumber of SMs: %d\nCompute Capability Major: %d\nCompute Capability Minor: %d\nWarp Size: %d\n", deviceId, multiProcessorCount, computeCapabilityMajor, computeCapabilityMinor, warpSize);
}
