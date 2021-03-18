#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_BLOCKS 2
#define BLOCK_WIDTH 16

__global__ void cuda_hello_test(){
  printf("Hello world! I'm thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

extern "C" void cuda_hello_world(){ 

  printf("Reached the second function. Attempting to run from GPU\n");


  int DeviceCount;
  cudaError_t dev_count = cudaGetDeviceCount(&DeviceCount);
  if (dev_count==0){
  printf("No Cuda supported GPUs detected.\n");
    }

  else{
  //cudaEvent_t start, stop;
  //float       elapsedTime;
  //cudaEventCreate(&start);
  //cudaEventRecord(start, 0);
  printf("Device count is %d\n", dev_count);
  printf("Running on CUDA GPUs\n");

  // launch the kernel
  cuda_hello_test<<<NUM_BLOCKS, BLOCK_WIDTH>>>();
  cudaDeviceSynchronize();

  printf("Done!\n");

  //cudaEventCreate(&stop);
  //cudaEventRecord(stop, 0);
  //cudaEventSynchronize(stop);
  //cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("Execution time: %f seconds\n", elapsedTime / 1000);
  }
}