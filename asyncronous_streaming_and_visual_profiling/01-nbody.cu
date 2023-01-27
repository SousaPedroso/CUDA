#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"
#include <assert.h>

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;


__global__ void integrateCoords(Body *p, float dt, int n){
  int threadId = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = blockDim.x * gridDim.x;

  for(int i=threadId; i<n; i+=gridStride){
    if (i < n){
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }
  }
}

/*
 * Calculate the gravitational impact of all bodies in the system
 * on all others.
 */
__global__ void bodyForce(Body *p, float dt, int n) {

  int threadId = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = blockDim.x * gridDim.x;

  for(int i=threadId; i<n; i+=gridStride){
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    if (i < n){
      for(int j=0; j<n; j++){
        float dx = p[j].x - p[i].x;
        float dy = p[j].y - p[i].y;
        float dz = p[j].z - p[i].z;
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      }

      p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
    }
  }
}

// void checkElementsAre(float* target, float *array, int N)
// {

//   for(int i = 0; i < N; i+=1){
//     if(array[i] != target[i]){
//       printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target[i]);
//       exit(1);
//     }
//   }
//   printf("SUCCESS! All values added correctly.\n");
// }

cudaError_t checkCuda(cudaError_t result){
    if (result != cudaSuccess){
        fprintf(stderr, "CUDA RUNTIME ERROR: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

int main(const int argc, const char** argv) {

  int nBodies = 2<<11;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);

  const char * initialized_values;
  const char * solution_values;

  if (nBodies == 2<<11) {
    initialized_values = "09-nbody/files/initialized_4096";
    solution_values = "09-nbody/files/solution_4096";
  } else { // nBodies == 2<<15
    initialized_values = "09-nbody/files/initialized_65536";
    solution_values = "09-nbody/files/solution_65536";
  }

  if (argc > 2) initialized_values = argv[2];
  if (argc > 3) solution_values = argv[3];

  const float dt = 0.01f; // Time step
  const int nIters = 10;  // Simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf;

  int deviceId, processorCount, blocks, threadsPerBlock;
  cudaDeviceProp props;
  cudaError_t computationError;

  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&props, deviceId);
  processorCount = props.multiProcessorCount;
  blocks = processorCount*2;
  threadsPerBlock = props.maxThreadsPerBlock;

  if (checkCuda(cudaMallocManaged(&buf, bytes)) != cudaSuccess){
    exit(1);
  }

  Body *p = (Body*)buf;
  if (checkCuda(cudaMemPrefetchAsync(p, bytes, deviceId)) != cudaSuccess){
    exit(1);
  }

  read_values_from_file(initialized_values, buf, bytes);

  double totalTime = 0.0;

  /*
   * This simulation will run for 10 cycles of time, calculating gravitational
   * interaction amongst bodies, and adjusting their positions to reflect.
   */

  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();

    bodyForce<<<blocks, threadsPerBlock>>>(p, dt, nBodies); // compute interbody forces
    computationError = cudaGetLastError();
    if (checkCuda(computationError) != cudaSuccess){
      exit(1);
    }

    cudaDeviceSynchronize(); // wait bodyforce compute
    integrateCoords<<<blocks, threadsPerBlock>>>(p, dt, nBodies);
    computationError = cudaGetLastError();
    if (checkCuda(computationError) != cudaSuccess){
      exit(1);
    }
    cudaDeviceSynchronize(); // wait the integration

    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;
  }

  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
  write_values_to_file(solution_values, buf, bytes);

  printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

  cudaFree(buf);
}