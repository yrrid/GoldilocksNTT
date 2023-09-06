/*

Copyright (c) 2023 Yrrid Software, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define $CUDA(call) if((call)!=0) { printf("\nCall \"" #call "\" failed from %s, line %d, error=%d\n", __FILE__, __LINE__, cudaGetLastError()); exit(1); }

__managed__ uint32_t nextCounter[128]={0};

__device__ __forceinline__ void loadValues(uint64_t* values, uint64_t* gmem) {
  uint32_t warpThread=threadIdx.x & 0x1F;
  
  #pragma unroll
  for(uint32_t i=0;i<32;i++)
    values[i]=gmem[i*32 + warpThread];
}

__device__ __forceinline__ void storeValues(uint64_t* gmem, uint64_t* values) {
  uint32_t warpThread=threadIdx.x & 0x1F;
  
  #pragma unroll
  for(uint32_t i=0;i<32;i++)
    gmem[i*32 + warpThread]=values[i];
}

__device__ __forceinline__ void storeShared(uint64_t* smem, uint64_t* values) {
  uint32_t stride=blockDim.x + 1;

  #pragma unroll
  for(uint32_t i=0;i<32;i++)
    smem[threadIdx.x + stride*i]=values[i];
}

__device__ __forceinline__ void loadShared(uint64_t* values, uint64_t* smem) {
  uint32_t warpThread=threadIdx.x & 0x1F, warp=threadIdx.x>>5;
  uint32_t stride=blockDim.x + 1;
  
  #pragma unroll
  for(uint32_t i=0;i<32;i++)
    values[i]=smem[warpThread*stride + warp*32 + i];
}

__global__ void transposeKernel(uint64_t* data, uint32_t* next, uint32_t count) {
  uint32_t warpThread=threadIdx.x & 0x1F;
  uint32_t index=0;
  uint64_t values[32];
  extern __shared__ uint64_t smem[];
  
  while(true) {
    if(warpThread==0) 
      index=atomicAdd(next, 1);                   // grab the next chunk of 1024 numbers
    index=__shfl_sync(0xFFFFFFFF, index, 0);
    if(index>=count)
      break;
  
    // load the data 1024 uint64_t values from global
    loadValues(values, data+index*1024);
    
    #ifdef NO_SHARED
      #pragma unroll
      for(int i=0;i<32;i++) 
        values[i]+=blockIdx.x;
    #else
      // transpose them in shared
      storeShared(smem, values);                    // do the transpose
      loadShared(values, smem);
    #endif
    
    // store 1024 values back out to global
    storeValues(data+index*1024, values);   
  }

  // last warp out the door resets the counter
  if(index==count + (gridDim.x*blockDim.x>>5)-1)
    *next=0;
}

uint64_t random_sample() {
  uint64_t x;

  x=rand() & 0xFFFF;
  x=(x<<16) + (rand() & 0xFFFF);
  x=(x<<16) + (rand() & 0xFFFF);
  x=(x<<16) + (rand() & 0xFFFF);
  if(x>0xFFFFFFFF00000001ull)
    x=x + 0xFFFFFFFFull;
  return x;
}

void random_samples(uint64_t* res, uint32_t count) {
  for(int i=0;i<count;i++)
    res[i]=random_sample();
}

int main(int argc, const char** argv) {
  uint32_t            ntts, repeatCount;
  uint64_t*           cpuData;
  uint64_t*           gpuData;
  cudaEvent_t         start, stop;
  float               time;

  if(argc!=3) {
    fprintf(stderr, "Usage:  ./%s <nttCount> <repeatCount>\n", argv[0]);
    fprintf(stderr, "Where <nttCount> is the number of NTT iterations to run on each set of 1024 point\n");
    fprintf(stderr, "and <repeatCount> is the number of times to run the kernel\n");
    return 0;
  }

  ntts=atoi(argv[1]);
  repeatCount=atoi(argv[2]);

  cpuData=(uint64_t*)malloc(sizeof(uint64_t)*ntts*1024);
  
  if(cpuData==NULL) {
    fprintf(stderr, "Malloc failed\n");
    exit(1);
  }
  
  random_samples(cpuData, ntts*1024);

  $CUDA(cudaMalloc((void**)&gpuData, sizeof(uint64_t)*ntts*1024));
  $CUDA(cudaFuncSetAttribute(transposeKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 97*1024));  

  transposeKernel<<<60, 256, 97*1024>>>(gpuData, nextCounter, ntts);
  $CUDA(cudaDeviceSynchronize());

  $CUDA(cudaMemcpy(gpuData, cpuData, sizeof(uint64_t)*ntts*1024, cudaMemcpyHostToDevice));
     
  fprintf(stderr, "Running kernel\n");

  $CUDA(cudaEventCreate(&start));
  $CUDA(cudaEventCreate(&stop));
  $CUDA(cudaEventRecord(start, 0));
  for(int i=0;i<repeatCount;i++) 
    transposeKernel<<<60, 384, 97*1024>>>(gpuData, nextCounter, ntts);
  $CUDA(cudaEventRecord(stop, 0));
  $CUDA(cudaDeviceSynchronize());
  $CUDA(cudaEventElapsedTime(&time, start, stop));

  if(cudaGetLastError()!=0)
    fprintf(stderr, "Finished == %d\n", cudaGetLastError());
  fprintf(stderr, "Runtime=%0.3f MS   L2BW = %0.3f GB/sec\n", time, 2.0/time*ntts*repeatCount*8/1024.0);
  $CUDA(cudaMemcpy(cpuData, gpuData, sizeof(uint64_t)*ntts*1024, cudaMemcpyDeviceToHost));
}
